import logging

import torch as t

from .resnet import *
from .switchable_norm import SwitchNorm2d


def create_model(args):
    logger = logging.getLogger()

    norm_layer = SwitchNorm2d if args.switchable_norm else None

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

    logger.info('Created `%s` model for `%s` dataset' % (args.arch, args.dataset))

    if args.gpu and not args.load_serialized:
        model = t.nn.DataParallel(model, device_ids=args.gpu)

    return model.to(args.device)
