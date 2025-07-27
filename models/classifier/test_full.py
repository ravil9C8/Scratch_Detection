# test_full.py
# ------------------------------------------------------------
# Evaluate a trained DenseNet-121 scratch-detection model on a
# *folder* of test images and report Precision, Recall & Specificity.
# ------------------------------------------------------------
import pathlib, time, torch, json
from PIL import Image
from torchvision import models
from tqdm import tqdm                              # ← Added
from ..utils.transforms import get_val_transforms  # ← your existing module

# --------------------------- CONFIGURATION ---------------------------
TEST_ROOT   = pathlib.Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test")  # ← EDIT
CKPT_FILE   = pathlib.Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_80per.pt")                    # ← EDIT
BATCH_SIZE  = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

assert TEST_ROOT.exists(), f"Test path not found: {TEST_ROOT}"
assert CKPT_FILE.exists(), f"Checkpoint not found: {CKPT_FILE}"

# --------------------------- DATA LOADING ----------------------------
good_imgs = sorted((TEST_ROOT / "good").glob("*"))
bad_imgs  = sorted((TEST_ROOT / "bad").glob("*"))
assert good_imgs and bad_imgs, "good/ or bad/ folder is empty."

all_paths = good_imgs + bad_imgs
all_lbls  = [0] * len(good_imgs) + [1] * len(bad_imgs)  # 0=good, 1=bad
tf        = get_val_transforms()

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --------------------------- MODEL SET-UP ----------------------------
model = models.densenet121(weights=None)
in_feat = model.classifier.in_features
model.classifier = torch.nn.Linear(in_feat, 1)  # single logit
state = torch.load(CKPT_FILE, map_location="cpu")
model.load_state_dict(state["model"])
model.to(DEVICE).eval()
sigm = torch.nn.Sigmoid()

# --------------------------- INFERENCE -------------------------------
TP = FP = TN = FN = 0
t0 = time.time()

with torch.no_grad():
    for idx_batch in tqdm(list(chunks(range(len(all_paths)), BATCH_SIZE)), desc="Evaluating"):
        imgs = [tf(Image.open(all_paths[i]).convert("RGB")) for i in idx_batch]
        batch = torch.stack(imgs).to(DEVICE)                 # (B,3,224,224)
        probs = sigm(model(batch)).squeeze(1).cpu()          # (B,)
        preds = (probs > 0.5).int()                          # 0/1
        labels = torch.tensor([all_lbls[i] for i in idx_batch])

        TP += int(((preds == 1) & (labels == 1)).sum())
        FP += int(((preds == 1) & (labels == 0)).sum())
        TN += int(((preds == 0) & (labels == 0)).sum())
        FN += int(((preds == 0) & (labels == 1)).sum())

# --------------------------- METRICS ---------------------------------
precision    = TP / (TP + FP + 1e-8)
recall       = TP / (TP + FN + 1e-8)          # = Sensitivity
specificity  = TN / (TN + FP + 1e-8)

print("\nEvaluation complete ───────────────")
print(f"Samples     : {len(all_paths)}  (good={len(good_imgs)}, bad={len(bad_imgs)})")
print(f"TP / FP / TN / FN : {TP} / {FP} / {TN} / {FN}")
print(f"Precision   : {precision:.3f}")
print(f"Recall      : {recall:.3f}")
print(f"Specificity : {specificity:.3f}")
print(f"Elapsed     : {time.time() - t0:.1f}s")

# --------------------------- OPTIONAL: SAVE RAW RESULTS --------------
# Uncomment to write filenames, labels, probabilities
# out_json = TEST_ROOT/"predictions.json"
# with open(out_json, "w") as f:
#     json.dump(
#         [
#             {"file": str(p), "label": int(l), "prob_bad": float(pb)}
#             for p,l,pb in zip(all_paths, all_lbls, probs.numpy())
#         ],
#         f, indent=2
#     )
# print(f"Raw predictions saved to {out_json}")
