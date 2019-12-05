import logging
from pathlib import Path

import torch as t

import process
import util
from model import create_model


def main():
    # Parse arguments
    args = util.get_parser().parse_args()

    script_dir = Path.cwd()
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, 'logging.conf')
    logger = logging.getLogger()

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

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
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True

    # Currently only ImageNet dataset is supported
    args.dataset = 'imagenet'
    args.num_classes = 1000

    # Create the model
    model = create_model(args)

    start_epoch = 0
    perf_scoreboard = process.PerformanceScoreboard(args.num_best_scores)

    if args.resume:
        model, start_epoch, _ = util.load_checkpoint(model, args.resume, args.device)

    # Initialize data loader
    train_loader, val_loader = util.load_data(args.dataset, args.dataset_dir,
                                              args.batch_size, args.load_workers)
    test_loader = val_loader
    logger.info('Dataset sizes:\n'
                '          training = %d\n'
                '        validation = %d\n'
                '              test = %d', len(train_loader), len(val_loader), len(test_loader))

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=5, verbose=True)
    # optimizer = t.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}],
    #                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, last_epoch=start_epoch - 1)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info(('LR scheduler: %s\n' % lr_scheduler).replace('\n', '\n' + ' ' * 11))

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume or args.pretrained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(test_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            process.train(train_loader, model, criterion, optimizer, epoch, monitors, args)

            top1, top5, val_loss = process.validate(test_loader, model, criterion, epoch, monitors, args)

            if lr_scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_loss, epoch)
                new_lr = optimizer.param_groups[0]['lr']
                logger.info('Learning rate updates from %.9f to %.9f' % (old_lr, new_lr))

            perf_scoreboard.update(top1, top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': top1}, is_best, args.name, log_dir)


if __name__ == "__main__":
    main()
