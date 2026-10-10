"""Aggregate output/<model>/<dataset>/<method>/<seed>/results.pkl into a results table.

usage: python summarize_results.py --model vit-tiny
writes output/<model>/results.csv and output/<model>/results.md
"""
import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

DATASETS = ['cifar10', 'cifar100', 'tiny-imagenet', 'stl10', 'food101']
DATASET_NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'tiny-imagenet': 'TinyImageNet',
                 'stl10': 'STL-10', 'food101': 'Food-101'}
METHODS = ['baseline', 'wn', 'sn', 'sb', 'tb', 'war', 'srr', 'snr', 'snvr']
METHOD_NAMES = {'baseline': 'WD', 'wn': 'WN', 'sn': 'SN', 'sb': 'SB', 'tb': 'TB', 'war': 'WAR',
                'srr': 'SRR', 'snr': 'SNR', 'snvr': 'SNVR (Ours)'}


def last_epoch_acc(log_path):
    # "accuracy1: <last> / <best>%" is logged after every epoch
    acc = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'accuracy1: ([\d.]+) / ', line)
            if m:
                acc = float(m.group(1))
    return acc


def collect(root, model):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, model, '*', '*', '*', 'results.pkl'))):
        dataset, method, seed = path.split(os.sep)[-4:-1]
        with open(path, 'rb') as f:
            results = pickle.load(f)
        log_path = os.path.join(os.path.dirname(path), 'log_rank0.txt')
        rows.append({
            'model': model, 'dataset': dataset, 'method': method, 'seed': int(seed),
            'best_acc1': results['accuracy1'], 'best_acc5': results['accuracy5'], 'best_epoch': results['epoch'],
            'last_acc1': last_epoch_acc(log_path) if os.path.exists(log_path) else np.nan,
        })
    return pd.DataFrame(rows)


def fmt(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f'{values[0]:.2f}'
    return f'{values.mean():.2f}±{values.std():.2f}'


def to_markdown(df, metric):
    datasets = [d for d in DATASETS if d in set(df.dataset)]
    methods = [m for m in METHODS if m in set(df.method)] + sorted(set(df.method) - set(METHODS))
    means = df.groupby(['method', 'dataset'])[metric].mean()

    header = '| Method | ' + ' | '.join(DATASET_NAMES.get(d, d) for d in datasets) + ' |'
    lines = [header, '|' + '---|' * (len(datasets) + 1)]
    for m in methods:
        cells = []
        for d in datasets:
            sub = df[(df.method == m) & (df.dataset == d)][metric]
            if len(sub) == 0:
                cells.append('-')
                continue
            cell = fmt(sub)
            if means[(m, d)] == means.xs(d, level='dataset').max():
                cell = f'**{cell}**'
            cells.append(cell)
        lines.append(f'| {METHOD_NAMES.get(m, m)} | ' + ' | '.join(cells) + ' |')

    if 'baseline' in methods:
        lines.append('')
        lines.append('Δ vs WD (mean, %p):')
        lines.append('')
        lines.append('| Method | ' + ' | '.join(DATASET_NAMES.get(d, d) for d in datasets) + ' |')
        lines.append('|' + '---|' * (len(datasets) + 1))
        for m in methods:
            if m == 'baseline':
                continue
            cells = []
            for d in datasets:
                if (m, d) in means.index and ('baseline', d) in means.index:
                    cells.append(f'{means[(m, d)] - means[("baseline", d)]:+.2f}')
                else:
                    cells.append('-')
            lines.append(f'| {METHOD_NAMES.get(m, m)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit-tiny')
    parser.add_argument('--root', default='output')
    args = parser.parse_args()

    df = collect(args.root, args.model)
    if df.empty:
        raise SystemExit(f'no results found under {args.root}/{args.model}')
    df = df.sort_values(['dataset', 'method', 'seed'])
    df.to_csv(os.path.join(args.root, args.model, 'results.csv'), index=False)

    seeds = sorted(df.seed.unique())
    md = [f'# {args.model} results (seeds: {", ".join(map(str, seeds))})', '',
          '## Top-1 accuracy, best epoch (paper protocol)', '', to_markdown(df, 'best_acc1'), '',
          '## Top-1 accuracy, last epoch', '', to_markdown(df, 'last_acc1'), '']
    md = '\n'.join(md)
    with open(os.path.join(args.root, args.model, 'results.md'), 'w') as f:
        f.write(md)
    print(md)
