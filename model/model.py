import logging

import torch as t

from .resnet import *
from .switchable_norm import SwitchNorm2d


def create_model(args):
    logger = logging.getLogger()

    norm_layer = SwitchNorm2d if args.switchable_norm else None
    if norm_layer is not None and args.pretrained:
        raise ValueError("Pre-trained model only supports BatchNorm, but got %s" % norm_layer)

    model = None
    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained, quan_bit_w=args.nbit_w,
                         quan_bit_a=args.nbit_a, norm_layer=norm_layer)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained, quan_bit_w=args.nbit_w,
                         quan_bit_a=args.nbit_a, norm_layer=norm_layer)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained, quan_bit_w=args.nbit_w,
                         quan_bit_a=args.nbit_a, norm_layer=norm_layer)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained, quan_bit_w=args.nbit_w,
                          quan_bit_a=args.nbit_a, norm_layer=norm_layer)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained, quan_bit_w=args.nbit_w,
                          quan_bit_a=args.nbit_a, norm_layer=norm_layer)
    else:
        logger.error('Model architecture is not supported')
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pretrained
    msg += '\n            Normalization layer = %s' % norm_layer
    msg += '\n           Activation bit-width = %s' % args.nbit_a
    msg += '\n               Weight bit-width = %s' % args.nbit_w
    logger.info(msg)

    if args.gpu and not args.load_serialized:
        model = t.nn.DataParallel(model, device_ids=args.gpu)

    return model.to(args.device)
