# test.py
# -------------------------------------------------------
# Usage: python test.py          (no CLI args needed)
# -------------------------------------------------------
import pathlib, torch
from PIL import Image
from torchvision import models
from ..utils.transforms import get_val_transforms   # <- from your existing module

# -------------------------------------------------------
# 1. Hard-code two image paths here
#    (absolute or relative to this script)
# -------------------------------------------------------
IMG_PATH_1 = "/home/ravil/assignment_mowito/data/anomaly_detection_test_data/bad/03_08_2024_16_54_38.244099_classifier_input.png"
IMG_PATH_2 = "/home/ravil/assignment_mowito/data/anomaly_detection_test_data/good/03_08_2024_17_21_22.750544_classifier_input.png"
IMG_PATHS  = [IMG_PATH_1, IMG_PATH_2]

# -------------------------------------------------------
# 2. Load checkpoint & rebuild model
# -------------------------------------------------------
CKPT_FILE = pathlib.Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_best.pt")
assert CKPT_FILE.exists(), f"checkpoint not found: {CKPT_FILE}"

model = models.densenet121(weights=None)
in_feat = model.classifier.in_features
model.classifier = torch.nn.Linear(in_feat, 1)        # 1-logit head (as in final training)

state = torch.load(CKPT_FILE, map_location="cpu")
model.load_state_dict(state["model"])
model.eval()

# -------------------------------------------------------
# 3. Single-image inference helper
# -------------------------------------------------------
tf   = get_val_transforms()          # deterministic resize + normalise
sigm = torch.nn.Sigmoid()

def predict(path: str) -> tuple[int, float]:
    """Return (pred_label, prob_bad) for one image."""
    img = Image.open(path).convert("RGB")
    x   = tf(img).unsqueeze(0)        # shape (1, 3, 224, 224)
    with torch.no_grad():
        logit = model(x).squeeze()    # shape ()
        prob  = sigm(logit).item()    # scalar
    label = int(prob > 0.5)           # 0 = good, 1 = bad
    return label, prob

# -------------------------------------------------------
# 4. Run predictions
# -------------------------------------------------------
for p in IMG_PATHS:
    lbl, pr = predict(p)
    cls = "bad" if lbl else "good"
    print(f"{p}:  class = {cls}   prob_bad = {pr:.3f}")
