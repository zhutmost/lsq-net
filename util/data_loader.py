import torch as t
import torchvision as tv
import os


def load_data(dataset, data_dir, batch_size, workers):
    if dataset == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'), transform=train_transform)
        train_loader = t.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

        val_set = tv.datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'), transform=val_transform)
        val_loader = t.utils.data.DataLoader(
            val_set, batch_size=batch_size, num_workers=workers, pin_memory=True)

        return train_loader, val_loader
    else:
        raise ValueError('load_data does not support dataset %s' % dataset)
