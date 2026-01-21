import os
import time
import random
import argparse
import csv
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.utils import make_grid, save_image


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Low-resource (stratified) sampling
# ---------------------------
def stratified_subsample_indices(dataset, fraction: float, seed: int):
    """
    Stratified sampling for CIFAR-10: sample the same fraction per class.
    dataset.targets is a list of class labels (0..9).
    """
    assert 0 < fraction <= 1.0
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


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    gap: float


# ---------------------------
# Transforms
# ---------------------------
def get_transforms(exp_name: str):
    # Standard CIFAR-10 normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    common = [T.ToTensor(), T.Normalize(mean, std)]

    if exp_name == "baseline":
        train_tf = T.Compose(common)

    elif exp_name == "geometric":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ] + common)

    elif exp_name == "photometric":
        train_tf = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ] + common)

    elif exp_name == "composite":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ] + common)

    else:
        raise ValueError(f"Unknown exp_name: {exp_name}")

    test_tf = T.Compose(common)
    return train_tf, test_tf


# ---------------------------
# Train / Eval (Modified for Loss Tracking)
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Returns (accuracy, avg_loss)"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        
        # Sum up batch loss
        total_loss += loss.item() * x.size(0)
        
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """Returns (accuracy, avg_loss)"""
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss


def save_aug_demo_image(
    data_root: str,
    out_dir: str,
    tb_writer: SummaryWriter | None,
    seed: int,
    example_index: int = 0,
):
    """
    Save a 2x4 grid: baseline/geometric/photometric/composite applied to the SAME image.
    NO Normalization for visualization.
    """
    os.makedirs(out_dir, exist_ok=True)

    raw_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=None
    )
    img, label = raw_ds[example_index]

    set_seed(seed) 

    # Visualization transforms (No Normalize!)
    vis_base = T.Compose([T.ToTensor()])
    vis_geo = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    vis_pho = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), T.ToTensor()])
    vis_com = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor()
    ])

    imgs = torch.stack([vis_base(img), vis_geo(img), vis_pho(img), vis_com(img)], dim=0)

    # Make it 1 row, 4 columns
    grid = make_grid(imgs, nrow=4, padding=2)
    out_path = os.path.join(out_dir, "augmentation_demo_grid.png")
    save_image(grid, out_path)

    if tb_writer is not None:
        tb_writer.add_image("augmentation_demo/grid", grid, global_step=0)

    return out_path, int(label)


def run_single_experiment(args) -> tuple[list[EpochStats], str | None, nn.Module]:
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = args.num_workers

    train_tf, test_tf = get_transforms(args.exp)

    proj_dir = os.path.dirname(__file__)
    data_root = os.path.join(proj_dir, "data")

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=args.download, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=args.download, transform=test_tf
    )

    writer = None
    if args.tb:
        log_dir = os.path.join(
            proj_dir,
            "runs",
            f"{args.exp}_frac{int(args.fraction*100)}_seed{args.seed}",
        )
        writer = SummaryWriter(log_dir=log_dir)

        if args.save_aug_demo:
            figs_dir = os.path.join(proj_dir, "figures")
            demo_path, demo_label = save_aug_demo_image(
                data_root=data_root,
                out_dir=figs_dir,
                tb_writer=writer,
                seed=args.seed,
                example_index=args.aug_demo_index,
            )
            writer.add_text(
                "augmentation_demo/info",
                f"Saved demo grid to: {demo_path} (example index={args.aug_demo_index}, label={demo_label})",
                global_step=0,
            )

    if args.fraction < 1.0:
        subset_idx = stratified_subsample_indices(full_train, args.fraction, args.seed)
        train_set = Subset(full_train, subset_idx)
    else:
        train_set = full_train

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = resnet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[EpochStats] = []
    
    print(f"Start training: {args.exp} (Frac={args.fraction}) on {device}")
    
    for ep in range(1, args.epochs + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        
        gap = train_acc - test_acc
        history.append(EpochStats(ep, train_loss, train_acc, test_loss, test_acc, gap))

        scheduler.step()

        print(
            f"[{args.exp}] Ep {ep:02d}/{args.epochs} | "
            f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
            f"Acc: {train_acc*100:.2f}%/{test_acc*100:.2f}% | "
            f"Gap: {gap*100:.2f}"
        )

        if writer is not None:
            writer.add_scalar("acc/train", train_acc, ep)
            writer.add_scalar("acc/test", test_acc, ep)
            writer.add_scalar("acc/gap", gap, ep)
            writer.add_scalar("loss/train", train_loss, ep)
            writer.add_scalar("loss/test", test_loss, ep)

    tb_log_dir = writer.log_dir if writer is not None else None
    if writer is not None:
        writer.close()

    return history, tb_log_dir, model


def save_csv(history: list[EpochStats], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Updated header with loss
        w.writerow(["Epoch", "Train_Loss", "Train_Acc", "Test_Loss", "Test_Acc", "Gap"])
        for s in history:
            w.writerow([s.epoch, s.train_loss, s.train_acc, s.test_loss, s.test_acc, s.gap])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="baseline",
        choices=["baseline", "geometric", "photometric", "composite"],
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Low-resource fraction of CIFAR-10 train set (e.g., 0.1 or 0.2). Use 1.0 for full.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--save_aug_demo", action="store_true")
    parser.add_argument("--aug_demo_index", type=int, default=0)

    args = parser.parse_args()

    t0 = time.time()
    history, tb_log_dir, model = run_single_experiment(args)
    dt = time.time() - t0

    proj_dir = os.path.dirname(__file__)
    
    # 1. Save CSV
    res_dir = os.path.join(proj_dir, "results")
    tag = f"{args.exp}_frac{int(args.fraction*100)}_seed{args.seed}"
    csv_path = os.path.join(res_dir, f"{tag}.csv")
    save_csv(history, csv_path)

    # 2. Save Checkpoint (Model Weights)
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save(model.state_dict(), ckpt_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Model: {ckpt_path}")
    
    if tb_log_dir is not None:
        print(f"TensorBoard: {tb_log_dir}")
    
    print(f"Done in {dt/60:.1f} min")


if __name__ == "__main__":
    main()