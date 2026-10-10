import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mnist_simple_cnn_comment_2_3 import train_one


METHODS = ["wd", "wn", "sn", "sb", "tb", "srr", "war", "snr", "snvr"]


SEARCH_GRIDS = {
    "wd": [
        {"weight_decay": 0.0},
        {"weight_decay": 5e-4},
        {"weight_decay": 1e-3},
    ],
    "wn": [{}],
    "sn": [{}],
    "sb": [
        {"sb_coeff": 1.0},
        {"sb_coeff": 2.0},
        {"sb_coeff": 5.0},
    ],
    "tb": [
        {"tb_lr_min_ratio": 0.5, "tb_lr_max_ratio": 1.5},
        {"tb_lr_min_ratio": 0.25, "tb_lr_max_ratio": 1.75},
        {"tb_lr_min_ratio": 0.75, "tb_lr_max_ratio": 1.25},
    ],
    "srr": [
        {"lambda_srr": 1e-2},
        {"lambda_srr": 1e-1},
        {"lambda_srr": 1.0},
    ],
    "war": [
        {"lambda_war": 4e-4},
        {"lambda_war": 4e-3},
        {"lambda_war": 4e-2},
    ],
    "snr": [
        {"lambda_snr": 1e-5},
        {"lambda_snr": 1e-4},
        {"lambda_snr": 1e-3},
    ],
    "snvr": [
        {"lambda_snr": 1e-4, "lambda_snvr": 1e-4},
        {"lambda_snr": 1e-4, "lambda_snvr": 1e-3},
        {"lambda_snr": 1e-4, "lambda_snvr": 1e-2},
    ],
}


DEFAULTS = {
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lambda_snr": 1e-4,
    "lambda_snvr": 1e-3,
    "lambda_srr": 1e-1,
    "lambda_war": 4e-3,
    "sb_coeff": 1.0,
    "tb_lr_min_ratio": 0.5,
    "tb_lr_max_ratio": 1.5,
    "tb_eigs_thresh": 10,
}


def make_args(base, overrides=None, epochs=None):
    values = deepcopy(DEFAULTS)
    if overrides:
        values.update(overrides)
    values.update(
        {
            "dataset": base.dataset,
            "data_path": base.data_path,
            "output": str(base.output),
            "epochs": base.search_epochs if epochs is None else epochs,
            "batch_size": base.batch_size,
            "test_batch_size": base.test_batch_size,
            "num_workers": base.num_workers,
            "in_channels": 1 if base.dataset == "mnist" else 3,
            "image_size": 28 if base.dataset == "mnist" else 32,
            "num_classes": 10,
        }
    )
    return SimpleNamespace(**values)


def dataset_objects(dataset, data_path):
    root = Path(data_path)
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_aug = datasets.MNIST(root=root / "mnist", train=True, download=True, transform=transform)
        train_eval = datasets.MNIST(root=root / "mnist", train=True, download=True, transform=transform)
        test = datasets.MNIST(root=root / "mnist", train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_aug = datasets.CIFAR10(root=root / "cifar10", train=True, download=True, transform=train_transform)
        train_eval = datasets.CIFAR10(root=root / "cifar10", train=True, download=True, transform=eval_transform)
        test = datasets.CIFAR10(root=root / "cifar10", train=False, download=True, transform=eval_transform)
    else:
        raise ValueError(dataset)
    return train_aug, train_eval, test


def loaders_for_search(base):
    train_aug, train_eval, _ = dataset_objects(base.dataset, base.data_path)
    rng = np.random.default_rng(base.split_seed)
    indices = rng.permutation(len(train_aug))
    val_size = int(len(train_aug) * base.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_set = Subset(train_aug, train_indices)
    val_set = Subset(train_eval, val_indices)
    return make_loader(train_set, base.batch_size, True, base), make_loader(val_set, base.test_batch_size, False, base)


def loaders_for_final(base):
    train_aug, _, test = dataset_objects(base.dataset, base.data_path)
    return make_loader(train_aug, base.batch_size, True, base), make_loader(test, base.test_batch_size, False, base)


def make_loader(dataset, batch_size, shuffle, base):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=base.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def summarize(rows):
    summary = []
    for method in METHODS:
        vals = np.array([r["best_acc"] for r in rows if r["method"] == method], dtype=float)
        if vals.size:
            summary.append({"method": method, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "n": int(vals.size)})
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_config(config):
    if not config:
        return "fixed"
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in config.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--data-path", default="../data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--search-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    base = parser.parse_args()
    if base.output is None:
        base.output = f"output_comment_2_3/hparam_search/{base.dataset}"
    base.output = Path(base.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_train_loader, val_loader = loaders_for_search(base)

    search_rows = []
    selected = {}
    for method in base.methods:
        best_score = -1.0
        best_config = None
        for config in SEARCH_GRIDS[method]:
            scores = []
            for seed in base.search_seeds:
                args = make_args(base, config, epochs=base.search_epochs)
                result = train_one(method, seed, args, search_train_loader, val_loader, device)
                scores.append(result["best_acc"])
            row = {
                "dataset": base.dataset,
                "method": method,
                "config": format_config(config),
                "val_mean": float(np.mean(scores)),
                "val_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n": len(scores),
            }
            search_rows.append(row)
            print(f"search {base.dataset} {method} {row['config']} val={row['val_mean']:.2f}")
            if row["val_mean"] > best_score:
                best_score = row["val_mean"]
                best_config = config
        selected[method] = deepcopy(best_config)

    final_train_loader, test_loader = loaders_for_final(base)
    final_rows = []
    for method in base.methods:
        config = selected[method]
        for seed in base.final_seeds:
            args = make_args(base, config, epochs=base.final_epochs)
            result = train_one(method, seed, args, final_train_loader, test_loader, device)
            result["dataset"] = base.dataset
            result["config"] = format_config(config)
            final_rows.append(result)
            print(f"final {base.dataset} {method} seed={seed} test={result['best_acc']:.2f} config={result['config']}")

    summary = summarize(final_rows)
    selected_rows = [
        {"dataset": base.dataset, "method": method, "selected_config": format_config(config)}
        for method, config in selected.items()
    ]

    write_csv(base.output / "search_results.csv", search_rows, ["dataset", "method", "config", "val_mean", "val_std", "n"])
    write_csv(base.output / "selected_hparams.csv", selected_rows, ["dataset", "method", "selected_config"])
    write_csv(base.output / "final_raw_results.csv", final_rows, ["method", "seed", "best_acc", "best_epoch", "final_acc", "final_loss", "seconds", "dataset", "config"])
    write_csv(base.output / "final_summary.csv", summary, ["method", "mean", "std", "n"])

    lines = ["| Method | Selected hyperparameter | Test Top-1 (%) |", "|---|---|---:|"]
    by_method = {row["method"]: row for row in summary}
    for method in base.methods:
        row = by_method[method]
        lines.append(f"| {method.upper()} | {format_config(selected[method])} | {row['mean']:.2f} +/- {row['std']:.2f} |")
    (base.output / "final_table.md").write_text("\n".join(lines) + "\n")

    payload = {"search": search_rows, "selected": selected_rows, "final_summary": summary, "final_raw": final_rows}
    (base.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
