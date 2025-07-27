# visualizer.py
# ------------------------------------------------------------
# Generate a Grad-CAM heat-map for a DenseNet-121 binary
# scratch-detection model trained with a 1-logit classifier head.
# ------------------------------------------------------------
import pathlib, cv2, torch
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.transforms import functional as F
from ..utils.transforms import get_val_transforms            # your existing module

# --------------------- USER INPUT -----------------------------------
IMG_PATH   = pathlib.Path("/home/ravil/assignment_mowito/data/visualize_images/03_08_2024_18_02_56.504574_classifier_input.png")       # ← edit
CKPT_FILE  = pathlib.Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_80per.pt")  # ← edit
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------

# --------------------- LOAD MODEL -----------------------------------
model = models.densenet121(weights=None)
in_feat = model.classifier.in_features
model.classifier = torch.nn.Linear(in_feat, 1)            # 1-logit head
state = torch.load(CKPT_FILE, map_location="cpu")["model"]
model.load_state_dict(state); model.to(DEVICE).eval()

# We’ll hook the **last convolutional feature map**: DenseNet’s `features`
target_layer = model.features

# --------------------- GRAD-CAM UTILS -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._hook_activations)
        target_layer.register_backward_hook(self._hook_gradients)

    def _hook_activations(self, _, __, output):
        self.activations = output.detach()               # shape (B,C,H,W)

    def _hook_gradients(self, _, grad_input, __):
        self.gradients = grad_input[0].detach()          # same shape

    def __call__(self, x):
        logit = self.model(x)                            # (B,1)
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))           # dlogit/dA
        # Global-avg-pool the gradients -> weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()                # (H,W)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)     # 0-1
        return cam, torch.sigmoid(logit).item()          # heat-map & P(bad)

# --------------------- PREPROCESS IMAGE -----------------------------
tf = get_val_transforms()
image_rgb = Image.open(IMG_PATH).convert("RGB")
x = tf(image_rgb).unsqueeze(0).to(DEVICE)                # (1,3,224,224)

# --------------------- RUN GRAD-CAM ---------------------------------
gradcam = GradCAM(model, target_layer)
heatmap, prob_bad = gradcam(x)

# --------------------- OVERLAY & SAVE -------------------------------
# Original (un-normalised) image for overlay
orig = np.array(image_rgb.resize((224, 224)))            # (H,W,3) uint8
heat = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0)

out_path = IMG_PATH.with_name(IMG_PATH.stem + "_gradcam.png")
cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"Grad-CAM saved to {out_path}")
print(f"Predicted P(bad) = {prob_bad:.3f}")
