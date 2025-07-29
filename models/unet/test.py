"""
viz_unet_prediction.py
──────────────────────
Visualise a single sample with:

1. original RGB image
2. ground-truth mask
3. U-Net predicted mask

Usage
-----
edit the four variables below, then run:
    python viz_unet_prediction.py
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from ..utils.transforms_unet import get_val_transforms_pair   # paired resize-224 + normalise

# ────────────── USER CONFIG ──────────────────────────────────
IMG_PATH   = Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test/bad/16_08_2024_13_34_25.067846_classifier_input.png")           # ← set your image
MASK_PATH  = Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test/masks/16_08_2024_13_34_25.067846_classifier_input.png")            # ← set your GT mask
CKPT_PATH  = Path("/home/ravil/assignment_mowito/weights/unet/unet_best_loss40_60.pt")
TITLE      = "Sample scratch detected - unet alpha 40"            # ← edit freely
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────


# 1. Load raw image & mask as numpy
img_np  = np.array(Image.open(IMG_PATH).convert("RGB"))
mask_np = np.array(Image.open(MASK_PATH).convert("L"))
mask_np = (mask_np > 128).astype(np.uint8)             # binarise

# 2. Apply the SAME validation transform used at training time
tf = get_val_transforms_pair()
pair = tf(image=img_np, mask=mask_np)
img_tensor = pair["image"].unsqueeze(0).to(DEVICE)     # (1,3,224,224)

# 3. Rebuild U-Net and load weights
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

state = torch.load(CKPT_PATH, map_location="cpu")["model"]
model.load_state_dict(state)
model.eval()

# 4. Forward pass → probability map (224×224)
with torch.no_grad():
    probs_224 = torch.sigmoid(model(img_tensor))       # (1,1,224,224)

# 5. Resize prediction **back to ground-truth shape**
Hgt, Wgt = mask_np.shape                                # original dims
probs_gt = F.interpolate(
    probs_224, size=(Hgt, Wgt), mode="bilinear", align_corners=False
)                                                       # (1,1,Hgt,Wgt)
pred_mask = (probs_gt.squeeze() > 0.5).cpu().numpy()    # (Hgt,Wgt) binary

# 6. Visualise
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle(TITLE, fontsize=14, fontweight="bold")

ax[0].imshow(img_np)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(mask_np, cmap="gray")
ax[1].set_title("Ground Truth")
ax[1].axis("off")

ax[2].imshow(pred_mask, cmap="gray")
ax[2].set_title("Predicted")
ax[2].axis("off")

plt.tight_layout()
plt.show()