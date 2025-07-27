"""
transforms.py
=============
Reusable image-transforms for the “scratch-on-text” project.

• The network input size is fixed at 224×224 (DenseNet-friendly).
• Two public helpers are exposed:
      get_train_transforms()  – used only on the training set
      get_val_transforms()    – used on validation / test / inference

Author : <your-name>
Date   : 2025-07-27
"""

from typing import Tuple
import random

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# ---------------------------------------------------------
# 1.  Constants
# ---------------------------------------------------------
IMG_SIZE: int = 224                           # final edge length
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------
# 2.  Letter-box utility (keeps full FoV, no cropping)
# ---------------------------------------------------------
class LetterBox:
    """
    Resize the input PIL image so that the longer side equals `size`
    while preserving aspect ratio, then pad the shorter side with
    `fill` colour so that the final image is exactly (size, size).
    """

    def __init__(self, size: int = IMG_SIZE, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    @staticmethod
    def _get_new_dims(w: int, h: int, size: int) -> Tuple[int, int]:
        scale = float(size) / max(h, w)
        return int(w * scale), int(h * scale)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image")

        w, h = img.size
        new_w, new_h = self._get_new_dims(w, h, self.size)
        img = F.resize(img, (new_h, new_w), antialias=True)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left   = pad_w // 2
        right  = pad_w - left
        top    = pad_h // 2
        bottom = pad_h - top

        return F.pad(img, padding=(left, top, right, bottom), fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill={self.fill})"

# ---------------------------------------------------------
# 3.  Augmentation blocks
# ---------------------------------------------------------
AUGMENTATIONS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),
    T.RandomRotation(degrees=10, expand=False, fill=(0, 0, 0)),
    T.RandomApply([T.ColorJitter(brightness=0.15,
                                 contrast=0.15,
                                 saturation=0.15)], p=0.5),
])

# ---------------------------------------------------------
# 4.  Public helper functions
# ---------------------------------------------------------
def get_train_transforms() -> T.Compose:
    """
    Transform chain for the training set.
    1. Letter-box to 224×224   (keeps entire scratch & text region)
    2. Light spatial + colour augmentation
    3. ToTensor + ImageNet normalisation
    """
    return T.Compose([
        LetterBox(IMG_SIZE, fill=0),
        AUGMENTATIONS,
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> T.Compose:
    """
    Deterministic transform for validation / test / inference:
    only letter-boxing and normalisation (no randomness).
    """
    return T.Compose([
        LetterBox(IMG_SIZE, fill=0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------------------------------------------
# 5.  Quick sanity check (run `python transforms.py` directly)
# -----------------------------------------------------------------
if __name__ == "__main__":
    import pathlib, matplotlib.pyplot as plt

    SAMPLE_PATH = next(pathlib.Path("dataset/good").glob("*"))  # pick one file
    img = Image.open(SAMPLE_PATH).convert("RGB")

    for mode, tf in [("train", get_train_transforms()),
                     ("val",   get_val_transforms())]:
        out = tf(img)                           # tensor [C,H,W]
        # Undo normalisation for visual debug
        vis = out.clone()
        for c in range(3):
            vis[c] = vis[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
        vis = vis.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        plt.figure(figsize=(3, 3))
        plt.title(f"{mode} pipeline result")
        plt.axis('off')
        plt.imshow(vis)
    plt.show()
