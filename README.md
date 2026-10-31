# Spectral Norm Variance Regularization

Code for "Spectral Norm Variance Regularization for Balanced Layer-Wise Dynamics in Training Deep Neural Networks".

## Environment

Python 3.11, PyTorch 2.9.1.

```bash
pip install -r requirements.txt
```

## Datasets

Datasets are read from `../data/<dataset>` (`--data-path ../data`).
CIFAR-10, CIFAR-100 and STL-10 are downloaded by torchvision at the first run.
TinyImageNet, Food-101 and EuroSAT are prepared as ImageFolder train / test directories:

```bash
mkdir -p ../data

# TinyImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ../data
mv ../data/tiny-imagenet-200 ../data/tiny-imagenet
python tiny-imagenet.py

# Food-101
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz -C ../data
mv ../data/food-101 ../data/food101
python food101.py

# EuroSAT (RGB; downloaded by torchvision, split 8:2 per class with a fixed seed)
python eurosat.py
```

## Training

One config per network architecture, dataset and method: `config/final/<model>/<dataset>/<method>.yaml` with

- models: `resnet18`, `resnet34`, `conv2next-tiny`, `vit-tiny`
- datasets: `cifar10`, `cifar100`, `tiny-imagenet`, `stl10`, `food101`, `eurosat`
- methods: `wd` (baseline), `wn`, `sn`, `sb`, `tb`, `war`, `srr`, `snr`, `snvr` (ours)

```bash
# SNVR, ResNet-34 on CIFAR-100, seed 0
python main.py --cfg config/final/resnet34/cifar100/snvr.yaml --data-path ../data --output output --seed 0

# the WD baseline
python main.py --cfg config/final/resnet34/cifar100/wd.yaml --data-path ../data --output output --seed 0

# all methods and seeds 0-2 of the main experiments (MODELS / DATASETS / METHODS / SEEDS select a subset)
MODELS="resnet34" DATASETS="cifar100" bash run_main.sh
```

A run writes its log and results to `output/<model>/<dataset>/<method>/<seed>`.
The results in the paper are the mean ± std. dev. over seeds 0, 1 and 2.
The selected hyperparameters of every method are listed in [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

## Usage

SNVR is implemented as an optimizer in `optimizer/snvr.py`, with `snvr=(λ_SNR, λ_SNVR)`.
The weight matrices are regularized as one group (the mean spectral norm is taken over it); the first and the output layer are excluded:

```python
from optimizer.build import set_spectral_norm, set_weight_decay
from optimizer.snvr import SGDSNVR, AdamWSNVR

model = ...
params = set_spectral_norm(set_weight_decay(model), skip=True, skip_first=True)

# SGD-based optimizer
optimizer = SGDSNVR(params, lr=0.1, momentum=0.9, weight_decay=5e-4, snvr=(0.01, 0.1))

# AdamW-based optimizer
optimizer = AdamWSNVR(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05, snvr=(10.0, 0.1))
```
