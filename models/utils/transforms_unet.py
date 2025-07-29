"""
transforms_unet.py
==================
Paired transforms for U-Net segmentation of text scratches.

• Resizes both image and mask to 224×224 (no padding, no letterbox)
• Applies *identical* spatial augmentation to image & mask (crucial)
• Normalises images (ImageNet), leaves masks as single-channel 0/1

Exports:
    get_train_transforms_pair() → (image, mask) pair with random aug
    get_val_transforms_pair()   → (image, mask) pair with deterministic preproc

"""

from typing import Callable, Tuple
import albumentations as A             # pip install albumentations
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transforms_pair() -> Callable:
    """
    Returns an Albumentations transform that, when applied as:
        out = tf(image=image_array, mask=mask_array)
    yields:
        out['image'] → float tensor [3,H,W] (normalised)
        out['mask']  → float tensor [1,H,W] (0 or 1)
    Spatial flips/rotations are applied to both image & mask in sync;
    colour/contrast only to image.
    """
    return A.Compose([
        # --- Spatial (applies to both im & mask) ---
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),        # 0/90/180/270 rotation
        A.Rotate(limit=10, border_mode=0, value=0, mask_value=0, p=0.5),
        # --- Photometric (image only) ---
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, p=0.5),
        # --- Convert ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ])

def get_val_transforms_pair() -> Callable:
    """
    Deterministic transforms for val/test: only resize, normalise, to tensor.
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, always_apply=True),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ])

#  --- How to use in dataset.py ---
# from transforms_unet import get_train_transforms_pair, get_val_transforms_pair
# tf = get_train_transforms_pair()
# out = tf(image=img_np, mask=mask_np)
# image_tensor, mask_tensor = out['image'], out['mask']
