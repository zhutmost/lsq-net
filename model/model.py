import logging

import torch as t

from .resnet import *
from .switchable_norm import SwitchNorm2d


def create_model(args):
    logger = logging.getLogger()

    norm_layer = SwitchNorm2d if args.switchable_norm else None
    if norm_layer is not None and args.pre_trained:
        raise ValueError("Pre-trained model only supports BatchNorm, but got %s" % norm_layer)

    model = None
    if args.dataloader.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = resnet18(pre_trained=args.pre_trained, norm_layer=norm_layer, quan_scheduler=args.quan)
        elif args.arch == 'resnet34':
            model = resnet34(pre_trained=args.pre_trained, norm_layer=norm_layer, quan_scheduler=args.quan)
        elif args.arch == 'resnet50':
            model = resnet50(pre_trained=args.pre_trained, norm_layer=norm_layer, quan_scheduler=args.quan)
        elif args.arch == 'resnet101':
            model = resnet101(pre_trained=args.pre_trained, norm_layer=norm_layer, quan_scheduler=args.quan)
        elif args.arch == 'resnet152':
            model = resnet152(pre_trained=args.pre_trained, norm_layer=norm_layer, quan_scheduler=args.quan)
    elif args.dataset.dataset == 'cifar10':
        pass

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    msg += '\n            Normalization layer = %s' % norm_layer
    msg += '\n           Activation bit-width = %s' % args.quan.bit_a
    msg += '\n               Weight bit-width = %s' % args.quan.bit_w
    logger.info(msg)

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    return model.to(args.device.type)
