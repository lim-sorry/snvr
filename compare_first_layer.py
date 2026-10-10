"""Compare SNVR with the first layer included (output/<model>/<dataset>/snvr_first/<seed>)
against the main results of the paper (Table 1, SNVR excludes the first layer).

usage: python compare_first_layer.py
writes output/first_layer_ablation.md and output/first_layer_ablation.csv
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd

MODELS = {'resnet18': 'ResNet-18', 'resnet34': 'ResNet-34', 'conv2next-tiny': 'Conv2NeXt-T'}
DATASETS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'tiny-imagenet': 'TinyImageNet',
            'stl10': 'STL-10', 'food101': 'Food-101'}

# Table 1 of the paper: (mean, std) of Top-1 accuracy over 5 seeds
PAPER = {
    'resnet18': {
        'WD':   [(96.28, 0.15), (80.11, 0.21), (68.68, 0.23), (87.93, 0.43), (85.82, 0.09)],
        'SNR':  [(96.27, 0.15), (80.03, 0.20), (68.74, 0.23), (90.01, 0.23), (85.56, 0.09)],
        'SNVR': [(96.25, 0.02), (80.25, 0.26), (69.15, 0.33), (89.75, 0.53), (85.94, 0.07)],
    },
    'resnet34': {
        'WD':   [(96.59, 0.08), (81.71, 0.21), (71.94, 0.21), (86.35, 1.16), (87.55, 0.25)],
        'SNR':  [(96.62, 0.04), (81.80, 0.10), (72.58, 0.11), (90.39, 0.21), (87.90, 0.03)],
        'SNVR': [(96.65, 0.06), (82.09, 0.04), (72.62, 0.16), (90.70, 0.14), (88.18, 0.07)],
    },
    'conv2next-tiny': {
        'WD':   [(97.36, 0.12), (83.23, 0.26), (71.83, 0.12), (89.66, 0.52), None],
        'SNR':  [(97.44, 0.06), (83.21, 0.29), (72.06, 0.02), (90.29, 0.31), None],
        'SNVR': [(97.46, 0.14), (83.19, 0.43), (72.07, 0.17), (90.40, 0.30), None],
    },
}

# raw runs of the paper's SNVR (first layer excluded) that are still kept locally
PAPER_RUNS = {
    ('resnet18', 'cifar100'): 'kt/resnet18/cifar100/snvr',
    ('resnet18', 'tiny-imagenet'): 'kt/resnet18/tiny-imagenet/snvr',
    ('resnet34', 'cifar100'): 'kt/resnet34/cifar100/snvr',
    ('resnet34', 'food101'): 'kt/resnet34/food101/snvr',
    ('conv2next-tiny', 'cifar100'): 'kt/conv2next-tiny/cifar100_1/snvr_1.0',
}


def load_acc(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['accuracy1']


def main(root='output', method='snvr_first'):
    rows = []
    for model in MODELS:
        for i, dataset in enumerate(DATASETS):
            snvr = PAPER[model]['SNVR'][i]
            if snvr is None:
                continue
            runs = {int(os.path.basename(os.path.dirname(p))): load_acc(p)
                    for p in glob.glob(os.path.join(root, model, dataset, method, '*', 'results.pkl'))}
            paper_runs = {}
            if (model, dataset) in PAPER_RUNS:
                paper_runs = {int(os.path.basename(os.path.dirname(p))): load_acc(p)
                              for p in glob.glob(os.path.join(PAPER_RUNS[(model, dataset)], '*', 'results.pkl'))}
            first = np.mean(list(runs.values())) if runs else np.nan
            first_std = np.std(list(runs.values())) if runs else np.nan
            same_seed = [s for s in runs if s in paper_runs]
            rows.append({
                'model': model, 'dataset': dataset,
                'WD': PAPER[model]['WD'][i][0], 'SNR': PAPER[model]['SNR'][i][0],
                'SNVR_mean': snvr[0], 'SNVR_std': snvr[1],
                'SNVR_first': first, 'SNVR_first_std': first_std, 'seeds': ','.join(map(str, sorted(runs))),
                'delta': first - snvr[0],
                'delta_std': (first - snvr[0]) / snvr[1] if snvr[1] > 0 else np.nan,
                'SNVR_same_seed': np.mean([paper_runs[s] for s in same_seed]) if same_seed else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'first_layer_ablation.csv'), index=False)

    lines = ['# SNVR: first layer excluded (paper) vs included', '',
             'WD / SNR / SNVR: Table 1 of the paper (5 seeds, first layer excluded). '
             'SNVR (same seeds): mean of the paper runs with the same seeds, where they are kept locally. '
             'SNVR+first: first layer included. Δ = SNVR+first − SNVR mean (in units of SNVR std).', '',
             '| Model | Dataset | WD | SNR | SNVR | SNVR (same seeds) | SNVR+first | Δ |',
             '|---|---|---|---|---|---|---|---|']
    for r in df.itertuples():
        same = f'{r.SNVR_same_seed:.2f}' if not np.isnan(r.SNVR_same_seed) else '-'
        if np.isnan(r.SNVR_first):
            first, delta = '-', '-'
        else:
            first = f'{r.SNVR_first:.2f}±{r.SNVR_first_std:.2f} (n={len(r.seeds.split(","))})'
            delta = f'{r.delta:+.2f} ({r.delta_std:+.1f}σ)'
        lines.append(f'| {MODELS[r.model]} | {DATASETS[r.dataset]} | {r.WD:.2f} | {r.SNR:.2f} | '
                     f'{r.SNVR_mean:.2f}±{r.SNVR_std:.2f} | {same} | {first} | {delta} |')
    done = df.dropna(subset=['SNVR_first'])
    if len(done):
        lines += ['', f'Mean Δ over {len(done)} settings: {done.delta.mean():+.2f}%p; '
                      f'SNVR+first > SNVR mean in {(done.delta > 0).sum()}/{len(done)}; '
                      f'|Δ| ≤ 1σ in {(done.delta_std.abs() <= 1).sum()}/{len(done)}; '
                      f'SNVR+first > max(WD, SNR) in {(done.SNVR_first > done[["WD", "SNR"]].max(axis=1)).sum()}/{len(done)} '
                      f'(paper SNVR: {(done.SNVR_mean > done[["WD", "SNR"]].max(axis=1)).sum()}/{len(done)})']
    md = '\n'.join(lines) + '\n'
    with open(os.path.join(root, 'first_layer_ablation.md'), 'w') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
