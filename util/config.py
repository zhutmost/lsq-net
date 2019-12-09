import argparse
import logging
import logging.config
import time


def get_parser():
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    # environment
    p.add_argument('--dataset-dir', metavar='DIR', required=True,
                   help='path to dataset directory (currently only ImageNet supported)')
    p.add_argument('--output-dir', metavar='DIR', default='out',
                   help='path to output directory')
    p.add_argument('--gpu', metavar='DEV_ID', default='',
                   help='comma-separated list of GPU device IDs to be used')
    p.add_argument('--cpu', action='store_true',
                   help='use CPU only, override the --gpu flag')
    p.add_argument('--load-workers', metavar='N', type=int, default=4,
                   help='number of data loading workers')
    p.add_argument('--name', '-n', metavar='NAME', default=None,
                   help='experiment name')
    p.add_argument('--load-serialized', action='store_true',
                   help='load the model without DataParallel wrapping it')
    p.add_argument('--print-freq', '-p', metavar='N', default=20, type=int,
                   help='print frequency (default: 20)')
    p.add_argument('--resume', default='', metavar='PATH',
                   help='path to latest checkpoint (default: none)')
    p.add_argument('--num-best-scores', metavar='N', default=1, type=int,
                   help='number of best scores to track and report (default: 1)')
    # model
    p.add_argument('--arch', default='resnet18',
                   help='model architecture (currently only ResNet supported)')
    p.add_argument('--pre-trained', dest='pretrained', action='store_true',
                   help='use pre-trained model')
    p.add_argument('--switchable-norm', action='store_true',
                   help='use switchable normalization layers instead of batch normalization')
    # quantization
    p.add_argument('--quan-bit-w', metavar='N', dest='nbit_w', type=int,
                   help='bit width of quantized weight')
    p.add_argument('--quan-bit-a', metavar='N', dest='nbit_a', type=int,
                   help='bit width of quantized activation')
    # training
    p.add_argument('--eval', action='store_true',
                   help='evaluate the model without training')
    p.add_argument('--batch-size', metavar='N', type=int, default=256,
                   help='mini-batch size')
    p.add_argument('--learning-rate', dest='lr', metavar='N', type=float, default=0.01,
                   help='learning rate')
    p.add_argument('--momentum', metavar='N', type=float, default=0.9, help='momentum')
    p.add_argument('--weight-decay', metavar='N', type=float, default=0.0001,
                   help='weight decay')
    p.add_argument('--epochs', metavar='N', type=int, default=90,
                   help='number of total epochs to run')
    return p


def init_logger(experiment_name, output_dir, cfg_file=None):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = time_str if experiment_name is None else experiment_name + '_' + time_str
    log_dir = output_dir / exp_full_name
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (exp_full_name + '.log')
    logging.config.fileConfig(cfg_file, defaults={'logfilename': log_file})
    logger = logging.getLogger()
    logger.info('Log file for this run: ' + str(log_file))
    return log_dir
