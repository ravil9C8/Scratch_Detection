# train_densenet.py
# -------------------------------------------------------
# End-to-end training script for scratch / no-scratch
# binary classification using DenseNet-121.
# -------------------------------------------------------
import argparse, json, pathlib, random, time

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from ..utils.transforms import get_train_transforms, get_val_transforms   # ← our module
from tqdm import tqdm
# -------------------------------------------------------
# 1.  Hyper-parameters & CLI
# -------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="dataset",         help="dataset folder with good/, bad/, mask/")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs",     type=int, default=32,    help="batch size")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--out",    default="weights",       help="checkpoint dir")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()

# -------------------------------------------------------
# 2.  Utility
# -------------------------------------------------------
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class ScratchDataset(Dataset):
    def __init__(self, paths, labels, train=True):
        self.paths  = paths
        self.labels = labels
        self.tf     = get_train_transforms() if train else get_val_transforms()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[idx], dtype=torch.long)

# -------------------------------------------------------
# 3.  Build loaders with stratified split
# -------------------------------------------------------
def make_loaders(root: str, bs: int, seed: int):
    good = sorted((pathlib.Path(root)/"good").glob("*"))
    bad  = sorted((pathlib.Path(root)/"bad").glob("*"))
    X = good + bad
    y = [0]*len(good) + [1]*len(bad)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, val_idx = next(sss.split(X, y))

    def subset(idxs, train):
        paths  = [X[i] for i in idxs]
        labels = [y[i] for i in idxs]
        ds  = ScratchDataset(paths, labels, train=train)

        if train:
            # class weight sampler to counter 1:4 imbalance
            cls_ct = torch.tensor([labels.count(0), labels.count(1)], dtype=torch.float)
            weights = 1. / cls_ct[labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    return subset(tr_idx, True), subset(val_idx, False)

def main():
    args = parse_args(); set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pathlib.Path(args.out).mkdir(exist_ok=True, parents=True)

    train_loader, val_loader = make_loaders(args.root, args.bs, args.seed)

    model = models.densenet121(weights="DEFAULT")
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, 1)  # binary head
    model.to(device)

    pos_weight = torch.tensor([train_loader.dataset.labels.count(0) /
                               train_loader.dataset.labels.count(1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # imbalance aware

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_recall = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y.float())
            optimizer.zero_grad(); loss.backward(); optimizer.step(); sched.step()
            epoch_loss += loss.item() * x.size(0)
            loop.set_postfix(loss=loss.item())

        # ----- validation -----
        model.eval(); TP = FP = FN = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = torch.sigmoid(model(x.to(device))).squeeze() > 0.5
                y = y.to(device).bool()
                TP += (pred & y).sum().item()
                FP += (pred & ~y).sum().item()
                FN += (~pred & y).sum().item()

        recall = TP / (TP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        print(f"Epoch {epoch:02d}: loss {epoch_loss / len(train_loader.dataset):.4f}  "
              f"recall {recall:.3f}  precision {precision:.3f}")

        # checkpoint best-recall model
        if recall > best_recall:
            best_recall = recall
            ckpt = {"model": model.state_dict(), "epoch": epoch,
                    "precision": precision, "recall": recall}
            torch.save(ckpt, f"{args.out}/densenet121_best.pt")

    print(f"Training done. Best recall {best_recall:.3f} saved to {args.out}/")

if __name__ == "__main__":
    main()
