import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18


def get_transforms(exp_name: str):
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


@torch.no_grad()
def extract_penultimate(model, x):
    """
    ResNet-18: hook avgpool output -> [B, 512, 1, 1], flatten -> [B, 512]
    """
    model.eval()
    feats = {}

    def hook_fn(module, inp, out):
        feats["v"] = out.flatten(1)

    h = model.avgpool.register_forward_hook(hook_fn)
    _ = model(x)
    h.remove()
    return feats["v"]


@torch.no_grad()
def aug_aug_consistency(model, base_dataset_pil, eval_tf, indices, K=8, batch_size=64, device="cpu"):
    """
    Aug-AuG cosine consistency under a FIXED evaluation perturbation distribution (eval_tf):
      For each image, sample two independent perturbed views T1(x), T2(x) ~ eval_tf
      Compute cosine(f(T1(x)), f(T2(x))) and average over K pairs.

    Returns mean/std over all (image, pair) samples.
    """
    model.eval()
    sims_all = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs_pil = [base_dataset_pil[i][0] for i in batch_idx]

        for _ in range(K):
            x1 = torch.stack([eval_tf(img) for img in imgs_pil]).to(device)
            x2 = torch.stack([eval_tf(img) for img in imgs_pil]).to(device)

            f1 = extract_penultimate(model, x1)
            f2 = extract_penultimate(model, x2)

            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)

            sim = (f1 * f2).sum(dim=1)  # [B]
            sims_all.append(sim.detach().cpu().numpy())

    sims_all = np.concatenate(sims_all, axis=0)  # len = N*K
    return float(sims_all.mean()), float(sims_all.std())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_dir = os.path.dirname(__file__)
    data_root = os.path.join(proj_dir, "data")
    ckpt_dir = os.path.join(proj_dir, "checkpoints")

    # Fixed evaluation set (same for all models)
    N = 512
    rng = np.random.default_rng(0)
    train_pil = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=None
    )
    indices = rng.choice(len(train_pil), size=N, replace=False).tolist()

    # IMPORTANT: fixed evaluation perturbation for fair comparison across methods
    # You can switch this between "geometric" and "composite" to probe robustness.
    EVAL_PERTURB = "composite"
    eval_tf, _ = get_transforms(EVAL_PERTURB)

    exps = ["baseline", "geometric", "photometric", "composite"]
    fracs = [0.1, 0.2]
    seeds = [0, 1, 2]
    K = 8

    print(f"Device={device}, N={N}, K={K}")
    print(f"==== Aug-AuG cosine consistency under FIXED eval perturbation: {EVAL_PERTURB} ====")

    for frac in fracs:
        print(f"\n--- fraction = {int(frac*100)}% ---")
        for exp in exps:
            seed_means = []
            for seed in seeds:
                tag = f"{exp}_frac{int(frac*100)}_seed{seed}"
                ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")
                if not os.path.exists(ckpt_path):
                    print(f"[WARN] missing ckpt: {ckpt_path}")
                    continue

                model = resnet18(num_classes=10).to(device)
                state_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state_dict)

                mean_sim, std_sim = aug_aug_consistency(
                    model=model,
                    base_dataset_pil=train_pil,
                    eval_tf=eval_tf,       # fixed perturbation distribution
                    indices=indices,
                    K=K,
                    batch_size=64,
                    device=device,
                )
                seed_means.append(mean_sim)

            if len(seed_means) == 0:
                print(f"{exp:12s}: (no checkpoints found)")
                continue

            ms = np.array(seed_means, dtype=np.float64)
            print(f"{exp:12s}: cosine = {ms.mean():.4f} ± {ms.std(ddof=0):.4f}  (over seeds)")


if __name__ == "__main__":
    main()
