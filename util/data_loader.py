import os

import numpy as np
import torch as t
import torch.utils.data
import torchvision as tv
from sklearn.model_selection import train_test_split


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = t.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = t.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def load_data(cfg):
    if cfg.val_split < 0 or cfg.val_split >= 1:
        raise ValueError('val_split should be in the range of [0, 1) but got %.3f' % cfg.val_split)

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    if cfg.dataset == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'train'), transform=train_transform)
        test_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'val'), transform=val_transform)

    elif cfg.dataset == 'cifar10':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, 4),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.CIFAR10(cfg.path, train=True, transform=train_transform, download=True)
        test_set = tv.datasets.CIFAR10(cfg.path, train=False, transform=val_transform, download=True)

    else:
        raise ValueError('load_data does not support dataset %s' % cfg.dataset)

    if cfg.val_split != 0:
        train_set, val_set = __balance_val_split(train_set, cfg.val_split)
    else:
        # In this case, use the test set for validation
        val_set = test_set

    worker_init_fn = None
    if cfg.deterministic:
        worker_init_fn = __deterministic_worker_init_fn

    train_loader = t.utils.data.DataLoader(
        train_set, cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = t.utils.data.DataLoader(
        val_set, cfg.batch_size, num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = t.utils.data.DataLoader(
        test_set, cfg.batch_size, num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader
