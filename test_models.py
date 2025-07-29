"""
run_all_models.py
─────────────────────────────────────────────────────────
Feed ONE image + its ground-truth mask through

1. DenseNet-121 classifier   – letterbox preprocessing  
2. DenseNet-121 classifier   – plain-resize preprocessing  
3. U-Net (ResNet-34 encoder) – segmentation + mask output

Then:

• prints, in a table, the probability of “bad” and the predicted
  label (good / bad) for **both classifiers**  
• stores a side-by-side PNG showing
      original | ground-truth | U-Net prediction
  in the same spatial resolution as the ground truth.

Edit the USER CONFIG section, then run:
    python run_all_models.py
"""

# ────────── imports ───────────────────────────────────────
from pathlib import Path
from typing import Tuple, Callable
import numpy as np
import torch, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ────────── USER CONFIG ───────────────────────────────────
IMG_PATH     = Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test/bad/09_08_2024_18_34_16.316934_classifier_input.png")
MASK_PATH    = Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test/masks/09_08_2024_18_34_16.316934_classifier_input.png")         # GT mask (binary)
CKPT_DENSE_LB= Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_best.pt")
CKPT_DENSE_RS= Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_without_letterboxing.pt")
CKPT_UNET    = Path("/home/ravil/assignment_mowito/weights/unet/unet_best_loss40_60.pt")
OUT_FIG      = Path("viz_unet_output.png")                  # saved figure
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────

IMG_SIZE       = 224
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# ══════════════════════════════════════════════════════════
# 1.  TRANSFORMS (defined locally as requested)
# ══════════════════════════════════════════════════════════
# 1-A ► Letter-box utility
class LetterBox:
    def __init__(self, size: int = IMG_SIZE, fill: int = 0) -> None:
        self.size, self.fill = size, fill
    @staticmethod
    def _get_new_dims(w: int, h: int, size: int) -> Tuple[int, int]:
        scale = float(size) / max(h, w)
        return int(w * scale), int(h * scale)
    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image")
        w, h = img.size
        new_w, new_h = self._get_new_dims(w, h, self.size)
        img = TF.resize(img, (new_h, new_w), antialias=True)
        pad_w, pad_h = self.size - new_w, self.size - new_h
        left, right  = pad_w // 2, pad_w - pad_w // 2
        top,  bottom = pad_h // 2, pad_h - pad_h // 2
        return TF.pad(img, padding=(left, top, right, bottom), fill=self.fill)

# 1-B ► Letterbox validation transform
lb_val_tf = T.Compose([
    LetterBox(IMG_SIZE, fill=0),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 1-C ► Plain-resize validation transform
def _resize(img: Image.Image) -> Image.Image:
    return TF.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)

rs_val_tf = T.Compose([
    _resize,
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 1-D ► U-Net paired transform (Albumentations)
unet_val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(transpose_mask=True),
])

# ══════════════════════════════════════════════════════════
# 2.  LOAD IMAGE & MASK
# ══════════════════════════════════════════════════════════
img_np  = np.array(Image.open(IMG_PATH).convert("RGB"))
mask_np = np.array(Image.open(MASK_PATH).convert("L"))
mask_np = (mask_np > 128).astype(np.uint8)                 # GT binary

# ══════════════════════════════════════════════════════════
# 3.  CLASSIFIER 1 – DenseNet (letterbox)
# ══════════════════════════════════════════════════════════
from torchvision import models
dense_lb = models.densenet121(weights=None)
dense_lb.classifier = torch.nn.Linear(dense_lb.classifier.in_features, 1)
dense_lb.load_state_dict(torch.load(CKPT_DENSE_LB, map_location="cpu")["model"])
dense_lb.to(DEVICE).eval()

x_lb = lb_val_tf(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
prob_lb = torch.sigmoid(dense_lb(x_lb)).item()
label_lb = "BAD" if prob_lb > 0.5 else "GOOD"

# ══════════════════════════════════════════════════════════
# 4.  CLASSIFIER 2 – DenseNet (resize)
# ══════════════════════════════════════════════════════════
dense_rs = models.densenet121(weights=None)
dense_rs.classifier = torch.nn.Linear(dense_rs.classifier.in_features, 1)
dense_rs.load_state_dict(torch.load(CKPT_DENSE_RS, map_location="cpu")["model"])
dense_rs.to(DEVICE).eval()

x_rs = rs_val_tf(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
prob_rs = torch.sigmoid(dense_rs(x_rs)).item()
label_rs = "BAD" if prob_rs > 0.5 else "GOOD"

# ══════════════════════════════════════════════════════════
# 5.  U-NET SEGMENTATION
# ══════════════════════════════════════════════════════════
unet = smp.Unet("resnet34", encoder_weights=None,
                in_channels=3, classes=1, activation=None).to(DEVICE)
state = torch.load(CKPT_UNET, map_location="cpu")["model"]
unet.load_state_dict(state)
unet.eval()

pair = unet_val_tf(image=img_np, mask=mask_np)
x_unet = pair["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs_224 = torch.sigmoid(unet(x_unet))                 # (1,1,224,224)

# resize prediction back to GT size
Hgt, Wgt = mask_np.shape
probs_gt = F.interpolate(probs_224, size=(Hgt, Wgt),
                         mode="bilinear", align_corners=False)
pred_mask = (probs_gt.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

# ══════════════════════════════════════════════════════════
# 6.  PRINT CLASSIFIER RESULTS
# ══════════════════════════════════════════════════════════
print("\n────────── Classifier Results ──────────")
print(f"{'Model':<20} | {'Prob_bad':>8} | Pred")
print("-"*46)
print(f"{'DenseNet-LB':<20} | {prob_lb:8.3f} | {label_lb}")
print(f"{'DenseNet-Resize':<20} | {prob_rs:8.3f} | {label_rs}")

# ══════════════════════════════════════════════════════════
# 7.  SAVE SIDE-BY-SIDE FIGURE
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle(f"U-Net Prediction – {IMG_PATH.name}", fontweight="bold")

ax[0].imshow(img_np);       ax[0].set_title("Original");      ax[0].axis("off")
ax[1].imshow(mask_np, cmap="gray"); ax[1].set_title("Ground Truth"); ax[1].axis("off")
ax[2].imshow(pred_mask, cmap="gray"); ax[2].set_title("Predicted");   ax[2].axis("off")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=200)
print(f"\nU-Net visualisation saved to {OUT_FIG}\n")
