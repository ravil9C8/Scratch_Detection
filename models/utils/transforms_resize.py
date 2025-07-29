"""
transforms.py
=============
Reusable image-transforms for the “scratch-on-text” project
(now **plain-resize** instead of letter-boxing).

• Network input fixed at 224×224 (DenseNet-friendly).
• Public helpers:
      get_train_transforms()  – training set
      get_val_transforms()    – validation / test / inference
"""

from typing import Tuple
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# ---------------------------------------------------------
# 1.  Constants
# ---------------------------------------------------------
IMG_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------
# 2.  Augmentation blocks
# ---------------------------------------------------------
AUGMENTATIONS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),
    T.RandomRotation(degrees=10, expand=False, fill=(0, 0, 0)),
    T.RandomApply(
        [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)],
        p=0.5
    ),
])

# ---------------------------------------------------------
# 3.  Public helper functions
# ---------------------------------------------------------
def _resize(img: Image.Image) -> Image.Image:
    """Uniform resize to IMG_SIZE × IMG_SIZE (no padding, keeps 3-channel PIL)."""
    return F.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)

def get_train_transforms() -> T.Compose:
    """Training pipeline: resize → light aug → tensor → normalise."""
    return T.Compose([
        _resize,
        AUGMENTATIONS,
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms() -> T.Compose:
    """Validation / test pipeline: deterministic resize + normalise."""
    return T.Compose([
        _resize,
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ---------------------------------------------------------
# 4.  Quick sanity check (run: python transforms.py)
# ---------------------------------------------------------
if __name__ == "__main__":
    import pathlib, matplotlib.pyplot as plt

    SAMPLE_PATH = "/home/ravil/assignment_mowito/data/anomaly_detection_test_data/train/bad/03_08_2024_16_59_07.698036_classifier_input.png"
    img = Image.open(SAMPLE_PATH).convert("RGB")

    for mode, tf in [("train", get_train_transforms()),
                     ("val",   get_val_transforms())]:
        out = tf(img)
        vis = out.clone()
        for c in range(3):
            vis[c] = vis[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
        vis = vis.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        plt.figure(figsize=(3, 3))
        plt.title(f"{mode} pipeline result")
        plt.axis("off")
        plt.imshow(vis)
    plt.show()
