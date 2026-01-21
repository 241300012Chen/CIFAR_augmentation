import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

def stratified_subsample_indices(dataset, fraction: float, seed: int):
    targets = np.array(dataset.targets)
    num_classes = int(targets.max() + 1)
    rng = np.random.default_rng(seed)
    indices = []
    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        k = max(1, int(len(cls_idx) * fraction))
        indices.extend(cls_idx[:k].tolist())
    rng.shuffle(indices)
    return indices

@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_dir = os.path.dirname(__file__)
    data_root = os.path.join(proj_dir, "data")
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    out_path = os.path.join(proj_dir, "results", "clean_gap_epoch40.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    clean_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # clean datasets (IMPORTANT: same clean transform for train and test)
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=clean_tf)
    test_set   = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=clean_tf)

    fracs = [0.1, 0.2]
    seeds = [0, 1, 2]
    exps  = ["baseline", "geometric", "photometric", "composite"]

    rows = [["exp", "fraction", "seed", "train_acc_clean", "test_acc_clean", "gap_clean"]]

    for frac in fracs:
        for seed in seeds:
            # reconstruct the SAME subset indices used in training
            subset_idx = stratified_subsample_indices(train_full, frac, seed)
            train_subset = Subset(train_full, subset_idx)

            train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
            test_loader  = DataLoader(test_set,     batch_size=256, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

            for exp in exps:
                tag = f"{exp}_frac{int(frac*100)}_seed{seed}"
                ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")
                if not os.path.exists(ckpt_path):
                    print(f"[WARN] missing ckpt: {ckpt_path}")
                    continue

                model = resnet18(num_classes=10).to(device)
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)

                tr = evaluate_acc(model, train_loader, device)
                te = evaluate_acc(model, test_loader, device)
                gap = tr - te

                rows.append([exp, frac, seed, f"{tr*100:.2f}", f"{te*100:.2f}", f"{gap*100:.2f}"])
                print(f"{tag}: train_clean={tr*100:.2f} test_clean={te*100:.2f} gap_clean={gap*100:.2f}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
