from torch.utils.data import DataLoader
from . import ext_transforms as et
from .voc import VOCSegmentation

from typing import Tuple

def get_voc(data_root='./data', crop_size=512, crop_val=512, year='2012_aug', download=False):
    train_transform = et.ExtCompose([
        # et.ExtResize(size=crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(crop_size),
            et.ExtCenterCrop(crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = VOCSegmentation(root=data_root, year=year,
                                image_set='train', download=download, transform=train_transform)
    val_dst = VOCSegmentation(root=data_root, year=year,
                                image_set='val', download=False, transform=val_transform)

    return train_dst, val_dst

def build_data_loader(batch_size=1, num_workers=0, max_training_samples=-1) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = get_voc()
    if max_training_samples > 0:    # for testing
        num_samples = len(train_dataset)
        train_dataset.image_set
        max_training_samples = min(max_training_samples, num_samples)
        train_dataset.images = train_dataset.images[:max_training_samples]
        train_dataset.masks = train_dataset.masks[:max_training_samples]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader