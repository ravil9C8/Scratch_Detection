# test_unet.py
# ─────────────────────────────────────────────────────────────
# Evaluate a trained U-Net on a folder of test images.
#   • expects test_root/good  and  test_root/bad (+ masks for bad)
#   • checkpoint must come from train_unet.py (1-logit output)
#   • reports: Dice, IoU, Precision, Recall, Specificity
#   • lists all false-negative images (bad→pred good)
# ─────────────────────────────────────────────────────────────
import argparse, json, time, numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from ..utils.transforms_unet import get_val_transforms_pair

# ───────────── CLI ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root",   required=True,
                   help="test folder with good/, bad/, masks/")
    p.add_argument("--ckpt",   required=True, help="path to unet_best.pt")
    p.add_argument("--bs",     type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ───────────── Dataset ───────────────────────────────────────
class TestSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.imgs  = img_paths
        self.masks = mask_paths
        self.tf    = get_val_transforms_pair()

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        mpath= self.masks[idx]
        if mpath is None:                       # good image → blank mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:                                   # bad image → real mask
            mask = np.array(Image.open(mpath).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        out = self.tf(image=img, mask=mask)
        x   = out["image"]
        y   = out["mask"].float()               # (1,H,W)
        return x, y, str(self.imgs[idx])        # keep path for error log

def build_loader(root: Path, bs: int):
    good_dir, bad_dir, mask_dir = root/"good", root/"bad", root/"masks"
    assert good_dir.exists() and bad_dir.exists(), "good/ or bad/ missing"

    img_paths, mask_paths = [], []
    # good: blank masks
    for p in sorted(good_dir.glob("*")):
        img_paths.append(p); mask_paths.append(None)
    # bad: real masks (skip if missing)
    for p in sorted(bad_dir.glob("*")):
        m = mask_dir/p.name
        if not m.exists(): print(f"⚠︎ mask missing for {p.name} – skipped"); continue
        img_paths.append(p); mask_paths.append(m)

    return DataLoader(
        TestSegDataset(img_paths, mask_paths),
        batch_size=bs, shuffle=False, num_workers=4, pin_memory=True
    )

# ───────────── Metrics helpers ───────────────────────────────
def dice_coef(p, t, eps=1e-7):
    inter = (p & t).float().sum((1,2,3))
    union = p.float().sum((1,2,3)) + t.float().sum((1,2,3))
    return ((2*inter+eps)/(union+eps)).mean().item()

def iou_coef (p, t, eps=1e-7):
    inter = (p & t).float().sum((1,2,3))
    union = (p | t).float().sum((1,2,3))
    return ((inter+eps)/(union+eps)).mean().item()

# ───────────── Main evaluation ───────────────────────────────
def main():
    args = parse_args(); root = Path(args.root)
    loader = build_loader(root, args.bs)

    # rebuild model
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3,
                     classes=1, activation=None).to(args.device)
    ckpt  = torch.load(args.ckpt, map_location="cpu")['model']
    model.load_state_dict(ckpt); model.eval()

    TP=FP=TN=FN=0; total_dice=[]; total_iou=[]; fn_paths=[]
    t0=time.time()
    with torch.no_grad():
        for imgs, masks, paths in loader:
            imgs, masks = imgs.to(args.device), masks.to(args.device)
            if masks.ndim == 3:  # if mask is [B,H,W] → add channel dim
                masks = masks.unsqueeze(1)  # [B,1,H,W]
            logits = model(imgs); probs = torch.sigmoid(logits)
            preds  = (probs>0.5).float()

            # pixel metrics
            total_dice.append(dice_coef(preds.bool(), masks.bool()))
            total_iou .append(iou_coef (preds.bool(), masks.bool()))

            # image-level good/bad
            pred_bad = (preds.sum(dim=(1,2,3)) > 0)      # tensor[B]
            true_bad = (masks.sum(dim=(1,2,3)) > 0)
            TP += int((pred_bad &  true_bad).sum())
            FP += int((pred_bad & ~true_bad).sum())
            TN += int((~pred_bad & ~true_bad).sum())
            FN += int((~pred_bad &  true_bad).sum())

            # collect FN paths
            for p_flag, t_flag, pth in zip(pred_bad, true_bad, paths):
                if (p_flag==0) and (t_flag==1): fn_paths.append(pth)

    # aggregate
    precision   = TP/(TP+FP+1e-8)
    recall      = TP/(TP+FN+1e-8)
    specificity = TN/(TN+FP+1e-8)
    mean_dice   = np.mean(total_dice)
    mean_iou    = np.mean(total_iou)

    # report
    print("\n────────── Evaluation Summary ──────────")
    print(f"Images        : {len(loader.dataset)}")
    print(f"Pixel Dice    : {mean_dice:.3f}")
    print(f"Pixel IoU     : {mean_iou:.3f}")
    print(f"Precision     : {precision:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"Specificity   : {specificity:.3f}")
    print(f"Elapsed       : {time.time()-t0:.1f}s")

    # false-negatives
    if fn_paths:
        print("\nFalse-negative images (bad → predicted good):")
        for p in fn_paths: print("  ", p)
    else:
        print("\nNo false-negatives 🎉")

if __name__ == "__main__":
    main()
