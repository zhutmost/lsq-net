"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.hub import load_state_dict_from_url

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

model_urls = {
    'resnet20': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet20-12fca82f.th',
    'resnet32': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet32-d509ac18.th',
    'resnet44': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet44-014dd654.th',
    'resnet56': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet56-4bfd9763.th',
    'resnet110': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet110-1d1ed7c2.th.th',
    'resnet1202': 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet1202-f3b1deed.th',
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, (out.size()[3], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        s = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict = OrderedDict()
        for k, v in s['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
        model.load_state_dict(state_dict)
    return model


def resnet20(pretrained=False, progress=True):
    return _resnet('resnet20', BasicBlock, [3, 3, 3], pretrained, progress)


def resnet32(pretrained=False, progress=True):
    return _resnet('resnet32', BasicBlock, [5, 5, 5], pretrained, progress)


def resnet44(pretrained=False, progress=True):
    return _resnet('resnet44', BasicBlock, [7, 7, 7], pretrained, progress)


def resnet56(pretrained=False, progress=True):
    return _resnet('resnet56', BasicBlock, [9, 9, 9], pretrained, progress)


def resnet110(pretrained=False, progress=True):
    return _resnet('resnet110', BasicBlock, [18, 18, 18], pretrained, progress)


def resnet1202(pretrained=False, progress=True):
    return _resnet('resnet1202', BasicBlock, [200, 200, 200], pretrained, progress)
