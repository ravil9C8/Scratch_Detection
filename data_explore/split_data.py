# split_data_folders.py
# ------------------------------------------------------------
# Creates train/ and test/ folders (80 / 20 stratified) under
# /home/ravil/assignment_mowito/data/anomaly_detection_test_data
# ------------------------------------------------------------
import shutil, random, pathlib
from sklearn.model_selection import train_test_split

# 1.  Config
ROOT = pathlib.Path("/home/ravil/assignment_mowito/data/anomaly_detection_test_data")
TRAIN_RATIO = 0.80
SEED = 42

# 2.  Gather paths
good_imgs = sorted([p for p in (ROOT / "good").iterdir() if p.is_file()])
bad_imgs  = sorted([p for p in (ROOT / "bad").iterdir()  if p.is_file()])

assert good_imgs and bad_imgs, "good/ or bad/ folder is empty."

files   = good_imgs + bad_imgs
labels  = [0]*len(good_imgs) + [1]*len(bad_imgs)      # 0 = good, 1 = bad

# 3.  Stratified split
train_idx, test_idx = train_test_split(
    range(len(files)),
    train_size=TRAIN_RATIO,
    stratify=labels,
    random_state=SEED
)

split_map = {**{i: "train" for i in train_idx}, **{i: "test" for i in test_idx}}

# 4.  Helper to copy image (and mask if bad)
def copy_sample(img_path: pathlib.Path, split: str):
    cls = img_path.parent.name            # "good" or "bad"
    dest_dir = ROOT / split / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest_dir / img_path.name)

    # copy mask only for bad images
    if cls == "bad":
        mask_path = ROOT / "mask" / img_path.name
        if mask_path.exists():
            dest_mask_dir = ROOT / split / "mask"
            dest_mask_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, dest_mask_dir / mask_path.name)

# 5.  Execute copy loop
for idx, img_path in enumerate(files):
    copy_sample(img_path, split_map[idx])

# 6.  Report
def count(dir_path):
    return sum(1 for _ in dir_path.iterdir()) if dir_path.exists() else 0

train_good = count(ROOT / "train" / "good")
train_bad  = count(ROOT / "train" / "bad")
test_good  = count(ROOT / "test"  / "good")
test_bad   = count(ROOT / "test"  / "bad")

print("Split complete ✅")
print(f"Train : {train_good} good, {train_bad} bad")
print(f"Test  : {test_good} good, {test_bad} bad")
