import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer.htr import stable_rank as htr_stable_rank
from optimizer.htr import weight_alpha as htr_weight_alpha



class SimpleCNN(nn.Module):
    def __init__(self, in_channels, image_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        pooled_size = image_size // 4
        self.fc1 = nn.Linear(64 * pooled_size * pooled_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def regularized_modules(model, include_edges=False):
    modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return modules if include_edges else modules[1:-1]


def matrix_view(weight):
    return weight.reshape(weight.size(0), -1)


def spectral_norm_power(weight, n_power=5, eps=1e-12):
    weight_mat = matrix_view(weight)
    with torch.no_grad():
        u = F.normalize(torch.randn(weight_mat.size(0), device=weight.device), dim=0, eps=eps)
        for _ in range(n_power):
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps)
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=eps)
    return torch.dot(u, torch.mv(weight_mat, v))


def spectral_values(weight):
    return torch.linalg.svdvals(matrix_view(weight))



def tb_layer_metric(weight, is_conv, conv_norm=0.5, xmin_pos=2):
    with torch.no_grad():
        if is_conv:
            matrix = torch.flatten(weight.detach(), start_dim=2) * (conv_norm ** 0.5)
            singular_values = torch.linalg.svdvals(matrix).flatten()
        else:
            singular_values = torch.linalg.svdvals(matrix_view(weight.detach())).flatten()
        eigs = singular_values.pow(2).sort().values.clamp_min(1e-12)
        if eigs.numel() < 2:
            return None, int(eigs.numel())
        i = int(eigs.numel() / xmin_pos)
        i = min(max(i, 0), eigs.numel() - 2)
        tail = eigs[i:]
        n = float(tail.numel())
        denom = torch.log(tail).sum() - n * torch.log(eigs[i])
        if denom.abs().item() < 1e-12:
            return None, int(eigs.numel())
        alpha = 1.0 + n / denom
        metric = alpha * torch.log10(eigs[-1])
        return float(metric.item()), int(eigs.numel())


def tb_tuned_modules(model, eigs_thresh):
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    selected = []
    for name, module in modules[1:-1]:
        metric, eigs_num = tb_layer_metric(module.weight, isinstance(module, nn.Conv2d))
        if metric is not None and eigs_num >= eigs_thresh and np.isfinite(metric):
            selected.append((name, module, metric))
    return selected


def tb_lr_map(metrics, base_lr, lr_min_ratio, lr_max_ratio):
    metrics = np.asarray(metrics, dtype=float)
    if len(metrics) == 0:
        return []
    if np.isclose(metrics.min(), metrics.max()):
        return [base_lr for _ in metrics]
    return np.interp(metrics, [metrics.min(), metrics.max()], [lr_min_ratio * base_lr, lr_max_ratio * base_lr]).tolist()


def build_tb_param_groups(model, base_lr, lr_min_ratio, lr_max_ratio, eigs_thresh):
    selected = tb_tuned_modules(model, eigs_thresh)
    metrics = [item[2] for item in selected]
    lrs = tb_lr_map(metrics, base_lr, lr_min_ratio, lr_max_ratio)
    tuned_ids = set()
    param_groups = []
    for (_, module, _), lr in zip(selected, lrs):
        params = list(module.parameters())
        tuned_ids.update(id(p) for p in params)
        param_groups.append({"params": params, "lr": lr})
    untuned = [p for p in model.parameters() if id(p) not in tuned_ids]
    if untuned:
        param_groups.append({"params": untuned, "lr": base_lr})
    return param_groups, len(selected)


def update_tb_lrs(model, optimizer, base_lr, lr_min_ratio, lr_max_ratio, eigs_thresh, tuned_group_count):
    selected = tb_tuned_modules(model, eigs_thresh)
    metrics = [item[2] for item in selected[:tuned_group_count]]
    lrs = tb_lr_map(metrics, base_lr, lr_min_ratio, lr_max_ratio)
    for idx, lr in enumerate(lrs):
        if idx < len(optimizer.param_groups):
            optimizer.param_groups[idx]["lr"] = lr
    for idx in range(len(lrs), len(optimizer.param_groups)):
        optimizer.param_groups[idx]["lr"] = base_lr


def snr_loss(model):
    terms = [spectral_norm_power(m.weight).pow(2) for m in regularized_modules(model)]
    return 0.5 * torch.stack(terms).sum()


def snvr_loss(model):
    sns = torch.stack([spectral_norm_power(m.weight) for m in regularized_modules(model)])
    return 0.5 * (sns - sns.mean()).pow(2).sum()


def stable_rank_loss(model):
    return htr_stable_rank(model)


def weight_alpha_loss(model):
    return htr_weight_alpha(model)


@torch.no_grad()
def apply_spectral_bound(model, coeff=1.0):
    for module in regularized_modules(model, include_edges=True):
        weight = module.weight
        weight_mat = matrix_view(weight)
        norm_1 = weight_mat.abs().sum(dim=0).max()
        norm_inf = weight_mat.abs().sum(dim=1).max()
        upper_bound = torch.sqrt((norm_1 * norm_inf).clamp_min(1e-12))
        if upper_bound > coeff:
            weight.mul_(coeff / upper_bound)


def build_model(method, device, in_channels, image_size, num_classes):
    model = SimpleCNN(in_channels=in_channels, image_size=image_size, num_classes=num_classes)
    if method == "wn":
        for module in regularized_modules(model, include_edges=True):
            weight_norm(module, name="weight", dim=0)
    elif method == "sn":
        for module in regularized_modules(model, include_edges=True):
            spectral_norm(module, name="weight", n_power_iterations=1)
    return model.to(device)


def regularization_loss(method, model, lambdas):
    if method == "srr":
        return lambdas["srr"] * stable_rank_loss(model)
    if method == "war":
        return lambdas["war"] * weight_alpha_loss(model)
    if method == "snr":
        return lambdas["snr"] * snr_loss(model)
    if method == "snvr":
        return lambdas["snr"] * snr_loss(model) + lambdas["snvr"] * snvr_loss(model)
    return 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, targets).item()
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += targets.numel()
    return 100.0 * correct / total, loss_sum / total


def train_one(method, seed, args, train_loader, test_loader, device):
    set_seed(seed)
    model = build_model(method, device, args.in_channels, args.image_size, args.num_classes)
    tb_group_count = 0
    if method == "tb":
        param_groups, tb_group_count = build_tb_param_groups(
            model, args.lr, args.tb_lr_min_ratio, args.tb_lr_max_ratio, args.tb_eigs_thresh
        )
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    lambdas = {"snr": args.lambda_snr, "snvr": args.lambda_snvr, "srr": args.lambda_srr, "war": args.lambda_war}

    best_acc = 0.0
    best_epoch = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            reg = regularization_loss(method, model, lambdas)
            if isinstance(reg, torch.Tensor):
                loss = loss + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if method == "sb":
                apply_spectral_bound(model, coeff=args.sb_coeff)

        if method == "tb":
            update_tb_lrs(
                model, optimizer, args.lr, args.tb_lr_min_ratio, args.tb_lr_max_ratio,
                args.tb_eigs_thresh, tb_group_count
            )

        acc, _ = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

    final_acc, final_loss = evaluate(model, test_loader, device)
    return {
        "method": method,
        "seed": seed,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "seconds": time.time() - start,
    }


def summarize(rows):
    summary = []
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    for method in methods:
        values = np.array([row["best_acc"] for row in rows if row["method"] == method], dtype=float)
        summary.append(
            {
                "method": method,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    return summary


def write_outputs(rows, args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "raw_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(summary)

    payload = {"args": vars(args), "summary": summary, "raw": rows}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    order = {method: idx for idx, method in enumerate(args.methods)}
    lines = [
        "| Method | Top-1 Accuracy (%) |",
        "|---|---:|",
    ]
    for item in sorted(summary, key=lambda x: order[x["method"]]):
        lines.append(f"| {item['method'].upper()} | {item['mean']:.2f} +/- {item['std']:.2f} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../data")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-snr", type=float, default=1e-4)
    parser.add_argument("--lambda-snvr", type=float, default=1e-3)
    parser.add_argument("--lambda-srr", type=float, default=1e-4)
    parser.add_argument("--lambda-war", type=float, default=1e-6)
    parser.add_argument("--sb-coeff", type=float, default=1.0)
    parser.add_argument("--tb-lr-min-ratio", type=float, default=0.5)
    parser.add_argument("--tb-lr-max-ratio", type=float, default=1.5)
    parser.add_argument("--tb-eigs-thresh", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=["wd", "wn", "sn", "sb", "tb", "srr", "war", "snr", "snvr"])
    return parser.parse_args()


def build_datasets(args):
    if args.dataset == "mnist":
        args.in_channels = 1
        args.image_size = 28
        args.num_classes = 10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        data_root = Path(args.data_path) / "mnist"
        train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    elif args.dataset == "cifar10":
        args.in_channels = 3
        args.image_size = 32
        args.num_classes = 10
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        data_root = Path(args.data_path) / "cifar10"
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_set, test_set


def main():
    args = parse_args()
    if args.output is None:
        args.output = f"output_comment_2_3/{args.dataset}_simple_cnn"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set = build_datasets(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            result = train_one(method, seed, args, train_loader, test_loader, device)
            rows.append(result)
            print(
                f"{method} seed={seed} best={result['best_acc']:.2f} "
                f"final={result['final_acc']:.2f} epoch={result['best_epoch']}"
            )
            write_outputs(rows, args)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
