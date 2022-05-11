from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as F

from . import ext_transforms as et
from .voc import VOCSegmentation

from typing import Tuple

SIZE = 512
def get_voc(data_root='./data', crop_size=SIZE, crop_val=SIZE, year='2012_aug', download=False):
    train_transform = et.ExtCompose([
        # et.ExtResize(size=crop_size),
        et.ExtRandomScale((0.5, 2.0), interpolation=F.InterpolationMode.BILINEAR),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if crop_val:
        val_transform = et.ExtCompose([
            # F.InterpolationMode.
            et.ExtResize(crop_size, interpolation=F.InterpolationMode.BILINEAR),
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

def build_data_loader(data_root='./data', batch_size=1, num_workers=0, max_training_samples=-1, crop_size=512) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = get_voc(data_root, crop_size=crop_size, crop_val=crop_size)
    if max_training_samples > 0:    # for testing
        num_samples = len(train_dataset)
        train_dataset.image_set
        max_training_samples = min(max_training_samples, num_samples)
        train_dataset.images = train_dataset.images[:max_training_samples]
        train_dataset.masks = train_dataset.masks[:max_training_samples]
        max_val_samples = min(max_training_samples, len(val_dataset))
        val_dataset.images = val_dataset.images[:max_val_samples]
        val_dataset.masks = val_dataset.masks[:max_val_samples]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader