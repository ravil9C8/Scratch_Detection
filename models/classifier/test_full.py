import pathlib, time, torch, json
from PIL import Image
from torchvision import models
from tqdm import tqdm                              # ← Added
from ..utils.transforms import get_val_transforms  # ← your existing module

# --------------------------- CONFIGURATION ---------------------------
TEST_ROOT   = pathlib.Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data/test")  # ← EDIT
CKPT_FILE   = pathlib.Path("/home/ravil/assignment_mowito/weights/classifier/densenet121_without_letterboxing.pt")                    # ← EDIT
BATCH_SIZE  = 16
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

fn_paths = []                               # <<< NEW (1): container for FN image paths

# --------------------------- INFERENCE -------------------------------
with torch.no_grad():
    for idx_batch in chunks(range(len(all_paths)), BATCH_SIZE):
        imgs   = [tf(Image.open(all_paths[i]).convert("RGB")) for i in idx_batch]
        batch  = torch.stack(imgs).to(DEVICE)
        probs  = sigm(model(batch)).squeeze(1).cpu()
        preds  = (probs > 0.5).int()
        labels = torch.tensor([all_lbls[i] for i in idx_batch])

        # bookkeeping
        TP += int(((preds==1) & (labels==1)).sum())
        FP += int(((preds==1) & (labels==0)).sum())
        TN += int(((preds==0) & (labels==0)).sum())
        FN += int(((preds==0) & (labels==1)).sum())

        # store FN image paths
        for i, p, l in zip(idx_batch, preds, labels):
            if (p == 0) and (l == 1):      # predicted good but actually bad
                fn_paths.append(all_paths[i])   # <<< NEW (2)

# --------------------------- METRICS ---------------------------------
precision    = TP / (TP+FP+1e-8)
recall       = TP / (TP+FN+1e-8)
specificity  = TN / (TN+FP+1e-8)

print("\nEvaluation complete")
print(f"Samples     : {len(all_paths)}  (good={len(good_imgs)}, bad={len(bad_imgs)})")
print(f"TP / FP / TN / FN : {TP} / {FP} / {TN} / {FN}")
print(f"Precision   : {precision:.3f}")
print(f"Recall      : {recall:.3f}")
print(f"Specificity : {specificity:.3f}")

# --------------------------- FN PRINTOUT ------------------------------
if fn_paths:
    print("\nFalse-Negative images (predicted good but actually bad):")
    for p in fn_paths:
        print(f"  {p}")
else:
    print("\nNo False-Negatives 🎉")
