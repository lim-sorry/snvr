"""Prepare EuroSAT RGB (10 land-cover classes, 27,000 Sentinel-2 images of 64x64) as ImageFolder
train / test directories. EuroSAT has no official split: 8:2 per class (stratified) with a fixed
seed, so every run and machine uses the same split; the file list is written to
../data/eurosat/split.txt.

    python eurosat.py   # downloads with torchvision into ../data/eurosat/raw if missing
"""
import os
import shutil

import numpy as np

BASE = '../data/eurosat'
SRC = os.path.join(BASE, 'raw', 'eurosat', '2750')
TEST_FRACTION = 0.2
SEED = 0


def main():
    if not os.path.isdir(SRC):
        from torchvision.datasets import EuroSAT
        EuroSAT(root=os.path.join(BASE, 'raw'), download=True)
    classes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    assert len(classes) == 10, classes
    rng = np.random.default_rng(SEED)
    lines = []
    for c in classes:
        files = sorted(f for f in os.listdir(os.path.join(SRC, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        perm = rng.permutation(len(files))
        n_test = int(round(TEST_FRACTION * len(files)))
        test = set(perm[:n_test].tolist())
        for i, f in enumerate(files):
            split = 'test' if i in test else 'train'
            dst = os.path.join(BASE, split, c)
            os.makedirs(dst, exist_ok=True)
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copy(os.path.join(SRC, c, f), dst)
            lines.append(f'{split}\t{c}/{f}')
        print(f'{c}: {len(files) - n_test} train, {n_test} test')
    with open(os.path.join(BASE, 'split.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
