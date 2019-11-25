import torch as t
import logging
from .resnet import *


def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained, quan_bit_w=args.nbit_w, quan_bit_a=args.nbit_a)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained, quan_bit_w=args.nbit_w, quan_bit_a=args.nbit_a)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained, quan_bit_w=args.nbit_w, quan_bit_a=args.nbit_a)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained, quan_bit_w=args.nbit_w, quan_bit_a=args.nbit_a)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained, quan_bit_w=args.nbit_w, quan_bit_a=args.nbit_a)
    else:
        logger.error('Model architecture is not supported')
        exit(-1)

    logger.info('Initialized %s model for %s dataset' % (args.arch, args.dataset))

    if args.gpu and not args.load_serialized:
        model = t.nn.DataParallel(model, device_ids=args.gpu)

    return model.to(args.device)
