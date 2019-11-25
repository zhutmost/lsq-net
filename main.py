from pathlib import Path
import torch as t
import torchvision as tv
import logging

import util
from model import create_model
import process


def main():
    # Parse arguments
    args = util.get_parser().parse_args()

    script_dir = Path.cwd()
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, 'logging.conf')
    logger = logging.getLogger()
    pymonitor = util.PythonMonitor(logger)

    if args.cpu or not t.cuda.is_available() or args.gpu == '':
        args.device = 'cpu'
        args.gpu = []
    else:
        args.device = 'cuda'
        try:
            args.gpu = [int(s) for s in args.gpu.split(',')]
        except ValueError:
            logger.error('Argument --gpu must be a comma-separated list of integers only')
            exit(1)
        available_gpu = t.cuda.device_count()
        for dev_id in args.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list != 0
        t.cuda.set_device(args.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training
        t.backends.cudnn.benchmark = True

    # Currently only ImageNet dataset is supported
    args.dataset = 'imagenet'
    args.num_classes = 1000

    # Create the model
    model = create_model(args)
    # model = tv.models.resnet18(pretrained=args.pretrained).to(args.device)

    start_epoch = 0
    perf_scoreboard = []

    if args.resume:
        model, start_epoch, _ = util.load_checkpoint(model, args.resume, args.device)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device)

    optimizer = t.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}],
                            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, last_epoch=start_epoch - 1)

    # Initialize data loader
    train_loader, val_loader = util.load_data(args.dataset, args.dataset_dir,
                                              args.batch_size, args.load_workers)
    test_loader = val_loader
    logger.info('Dataset sizes:\n'
                '          training = %d\n'
                '        validation = %d\n'
                '              test = %d', len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.eval:
        process.validate(test_loader, model, criterion, -1, [pymonitor], args)
    else:  # training
        if args.resume or args.pretrained:
            top1, top5, vloss = process.validate(test_loader, model, criterion,
                                                 start_epoch - 1, [pymonitor], args)
            process.update_training_scoreboard(perf_scoreboard, model, top1, top5,
                                               start_epoch - 1, args.num_best_scores)
        for epoch in range(start_epoch, args.epochs):
            process.train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, [pymonitor], args)

            top1, top5, vloss = process.validate(test_loader, model, criterion, -1, [pymonitor], args)
            process.update_training_scoreboard(perf_scoreboard, model, top1, top5, epoch, args.num_best_scores)
            is_best = perf_scoreboard[0].epoch == epoch

            util.save_checkpoint(epoch, args.arch, model, {'top1': top1}, is_best, args.name, log_dir)


if __name__ == "__main__":
    main()
