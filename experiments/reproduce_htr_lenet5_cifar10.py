import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


METHODS = ["base", "weight_decay", "spectral_norm", "weighted_alpha", "stable_rank", "powerlaw_prior", "frechet_prior"]
PAPER_BETAS = {
    "base": 0.0,
    "weight_decay": 5e-4,
    "spectral_norm": 1e-4,
    "weighted_alpha": 4e-3,
    "stable_rank": 1e-1,
    "powerlaw_prior": 7e-4,
    "frechet_prior": 5e-5,
}
PAPER_TABLE = {
    "base": 72.42,
    "weight_decay": 72.62,
    "spectral_norm": 71.98,
    "weighted_alpha": 72.61,
    "stable_rank": 73.63,
    "powerlaw_prior": 72.61,
    "frechet_prior": 72.58,
}


class LeNet5Cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def regularized_weights(model):
    return [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def matrix_view(weight):
    return weight.reshape(weight.size(0), -1)


def singular_values(weight):
    return torch.linalg.svdvals(matrix_view(weight))


def hill_alpha(eigs, eps=1e-12):
    eigs = eigs.sort().values.clamp_min(eps)
    n = eigs.numel()
    if n < 2:
        return eigs.new_tensor(1.0)
    k = max(1, n // 2)
    threshold = eigs[n - k - 1].clamp_min(eps) if n - k - 1 >= 0 else eigs[0].clamp_min(eps)
    top = eigs[-k:].clamp_min(eps)
    denom = torch.log(top / threshold).sum().clamp_min(eps)
    return 1.0 + k / denom


def reduce_terms(terms, reduction):
    stacked = torch.stack(terms)
    return stacked.mean() if reduction == "mean" else stacked.sum()


def spectral_norm_loss(model, reduction="sum"):
    terms = []
    for weight in regularized_weights(model):
        s = singular_values(weight)
        terms.append(0.5 * s.max().pow(2))
    return reduce_terms(terms, reduction)


def weighted_alpha_loss(model, reduction="sum"):
    terms = []
    for weight in regularized_weights(model):
        eigs = singular_values(weight).pow(2)
        alpha = hill_alpha(eigs)
        terms.append(alpha * torch.log(eigs.max().clamp_min(1e-12)))
    return reduce_terms(terms, reduction)


def stable_rank_loss(model, reduction="sum"):
    terms = []
    for weight in regularized_weights(model):
        s = singular_values(weight)
        terms.append(s.pow(2).sum() / s.max().pow(2).clamp_min(1e-12))
    return reduce_terms(terms, reduction)


def powerlaw_prior_loss(model, reduction="sum"):
    terms = []
    for weight in regularized_weights(model):
        eigs = singular_values(weight).pow(2).sort().values.clamp_min(1e-12)
        n = eigs.numel()
        k = max(1, n // 2)
        alpha = hill_alpha(eigs)
        terms.append(alpha * torch.log(eigs[-k:]).sum())
    return reduce_terms(terms, reduction)


def frechet_prior_loss(model, reduction="sum"):
    terms = []
    for weight in regularized_weights(model):
        eigs = singular_values(weight).pow(2).clamp_min(1e-12)
        alpha = hill_alpha(eigs)
        terms.append(eigs.max().pow(-alpha))
    return reduce_terms(terms, reduction)


def regularization_loss(model, method, beta, reduction="sum"):
    if method == "spectral_norm":
        return beta * spectral_norm_loss(model, reduction)
    if method == "weighted_alpha":
        return beta * weighted_alpha_loss(model, reduction)
    if method == "stable_rank":
        return beta * stable_rank_loss(model, reduction)
    if method == "powerlaw_prior":
        return beta * powerlaw_prior_loss(model, reduction)
    if method == "frechet_prior":
        return beta * frechet_prior_loss(model, reduction)
    return 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / total


def train_one(method, seed, args, train_loader, test_loader, device):
    set_seed(seed)
    model = LeNet5Cifar10().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=PAPER_BETAS["weight_decay"] if method == "weight_decay" else 0.0,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    beta = PAPER_BETAS[method]

    best_acc = 0.0
    best_epoch = 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            reg = regularization_loss(model, method, beta, args.reg_reduction)
            if isinstance(reg, torch.Tensor):
                loss = loss + reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        if epoch == 1 or epoch % args.print_freq == 0 or epoch == args.epochs:
            print(f"{method} seed={seed} epoch={epoch} acc={acc:.2f} best={best_acc:.2f} loss={running_loss/len(train_loader.dataset):.4f}")
        history.append({"epoch": epoch, "acc": acc, "loss": running_loss / len(train_loader.dataset)})
    return {
        "method": method,
        "seed": seed,
        "beta": beta,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "paper_acc": PAPER_TABLE[method],
        "gap_to_paper": best_acc - PAPER_TABLE[method],
        "seconds": time.time() - start,
        "history": history,
    }


def summarize(rows):
    summary = []
    for method in METHODS:
        values = np.array([row["best_acc"] for row in rows if row["method"] == method], dtype=float)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        se = std / math.sqrt(values.size) if values.size > 1 else 0.0
        mean = float(values.mean())
        summary.append({
            "method": method,
            "beta": PAPER_BETAS[method],
            "paper_acc": PAPER_TABLE[method],
            "mean": mean,
            "std": std,
            "se": se,
            "gap_to_paper": mean - PAPER_TABLE[method],
            "n": int(values.size),
        })
    return summary


def write_outputs(rows, args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [{k: v for k, v in row.items() if k != "history"} for row in rows]
    if raw_rows:
        with (out_dir / "raw_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
    summary = summarize(rows)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "beta", "paper_acc", "mean", "std", "se", "gap_to_paper", "n"])
        writer.writeheader()
        writer.writerows(summary)
    lines = ["| Method | beta | Paper acc. | Reproduced Top-1 | Gap |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['beta']:.4g} | {row['paper_acc']:.2f} | {row['mean']:.2f} +/- {row['se']:.2f} | {row['gap_to_paper']:+.2f} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "summary": summary, "raw": rows}, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../data")
    parser.add_argument("--output", default="output_htr_lenet5_cifar10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--reg-reduction", choices=["sum", "mean"], default="sum")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if args.no_augment:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    data_root = Path(args.data_path) / "cifar10"
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            result = train_one(method, seed, args, train_loader, test_loader, device)
            rows.append(result)
            write_outputs(rows, args)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
