# train_unet.py
# ------------------------------------------------------------
# Train a U-Net for scratch segmentation using folder paths.
# ------------------------------------------------------------
import argparse, os, random, time, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torch.nn.functional as F
import segmentation_models_pytorch as smp             # pip install segmentation-models-pytorch albumentations
from tqdm import tqdm

from ..utils.transforms_unet import (
    get_train_transforms_pair,
    get_val_transforms_pair,
)

# ------------------------------------------------------------
# 1.  CLI ─────────────────────────────────────────────────────
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", required=True,
                   help="dataset root containing good/, bad/ and mask/ folders")
    p.add_argument("--out", default="weights_unet", help="where to store checkpoints")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs",     type=int, default=8)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------
# 2.  Dataset ────────────────────────────────────────────────
# ------------------------------------------------------------
class ScratchSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, train=True):
        self.imgs  = img_paths
        self.masks = mask_paths
        self.tf    = get_train_transforms_pair() if train else get_val_transforms_pair()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB"))

        mask_path = self.masks[idx]
        if mask_path.name == "dummy_blank.png":          # good image → blank mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:                                            # bad image → real mask
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 128).astype(np.uint8)

        # ‼️  These two lines were missing ‼️
        out = self.tf(image=img, mask=mask)              # paired augmentation
        mask_tensor = out["mask"].float()          # shape (H,W) or (1,H,W)
        if mask_tensor.dim() == 2:                 # (H,W) → insert channel
            mask_tensor = mask_tensor.unsqueeze(0)
        return out["image"], mask_tensor  


# ------------------------------------------------------------
# 3.  Helpers ─────────────────────────────────────────────────
# ------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_dataset(root: Path):
    good_dir, bad_dir, mask_dir = root / "good", root / "bad", root / "masks"
    assert good_dir.exists() and bad_dir.exists(), "good/ or bad/ folder missing"

    img_paths, mask_paths = [], []

    # Good images → blank mask path duplicates to keep list lengths equal
    for p in sorted(good_dir.glob("*")):
        img_paths.append(p)
        # create a dummy all-black mask in memory later
        mask_paths.append(mask_dir / "dummy_blank.png")

    # Bad images → must have matching mask
    for p in sorted(bad_dir.glob("*")):
        m = mask_dir / p.name
        if not m.exists():
            print(f"Warning: mask missing for {p.name} — sample skipped")
            continue
        img_paths.append(p)
        mask_paths.append(m)

    labels = [0]*len(list(good_dir.glob("*"))) + [1]*len(list(bad_dir.glob("*")))
    return img_paths, mask_paths, labels


def build_loaders(root, bs, seed):
    imgs, masks, labels = discover_dataset(root)

    train_idx, val_idx = train_test_split(
        range(len(imgs)),
        train_size=0.80,
        stratify=labels,
        random_state=seed,
    )

    def subset(indices, train):
        sub_imgs  = [imgs[i]  for i in indices]
        sub_masks = [masks[i] for i in indices]
        ds = ScratchSegDataset(sub_imgs, sub_masks, train=train)
        return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=4, pin_memory=True)

    return subset(train_idx, True), subset(val_idx, False)


def dice_loss(pred, target, smooth=1):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice   = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

# … previous imports & code remain unchanged …

# ─────────────────────────────────────────────────────────────
# 4.  Main training loop
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(Path(args.root), args.bs, args.seed)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    opt           = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_dice     = 0.0
    alpha         = 0.4                    # weight for Dice in the mixed loss

    for epoch in range(1, args.epochs + 1):
        # ───── TRAIN ───────────────────────────────────────
        model.train()
        tr_loss, tr_bce, tr_dice = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = model(x)
            dice   = dice_loss(logits, y)
            bce    = F.binary_cross_entropy_with_logits(logits, y)
            loss   = alpha * dice + (1 - alpha) * bce
            loss.backward()
            opt.step()

            # accumulate batch metrics
            bs            = x.size(0)
            tr_loss += loss.item() * bs
            tr_bce  += bce.item()  * bs
            tr_dice += (1 - dice.item()) * bs         # covert loss back to Dice score

            pbar.set_postfix(loss=loss.item())

        # ───── VALIDATE ────────────────────────────────────
        model.eval()
        val_loss, val_dice_list, val_bce = 0.0, [], 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False)

        with torch.no_grad():
            for x, y in val_bar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                dice   = dice_loss(logits, y)
                bce    = F.binary_cross_entropy_with_logits(logits, y)
                loss   = alpha * dice + (1 - alpha) * bce

                bs          = x.size(0)
                val_loss   += loss.item() * bs
                val_bce    += bce.item()  * bs

                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * y).sum((1, 2, 3))
                union = preds.sum((1, 2, 3)) + y.sum((1, 2, 3))
                val_dice_list += list(((2 * inter + 1) / (union + 1)).cpu().numpy())

                val_bar.set_postfix(dice=np.mean(val_dice_list))

        # epoch-level averages
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)

        mean_tr_loss = tr_loss / n_train
        mean_tr_bce  = tr_bce  / n_train
        mean_tr_dice = tr_dice / n_train

        mean_val_loss = val_loss / n_val
        mean_val_bce  = val_bce  / n_val
        mean_val_dice = float(np.mean(val_dice_list))

        print(
            f"Epoch {epoch:02d} │ "
            f"Train loss {mean_tr_loss:.4f} (BCE {mean_tr_bce:.4f}, Dice {mean_tr_dice:.3f}) │ "
            f"Val loss {mean_val_loss:.4f} (BCE {mean_val_bce:.4f}, Dice {mean_val_dice:.3f})"
        )

        # checkpoint on best validation Dice
        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice},
                Path(args.out) / "unet_best_loss40_60.pt",
            )
            print(f"  ↳ New best model saved (Dice {best_dice:.3f})")

        # save final weights at end of training
        if epoch == args.epochs:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice},
                Path(args.out) / "unet_final.pt",
            )
            print(f"  ↳ Final model saved (Dice {best_dice:.3f})")

    print(f"Training complete — best Dice {best_dice:.3f} stored in unet_best_loss40_60.pt")

if __name__ == "__main__":
    main()