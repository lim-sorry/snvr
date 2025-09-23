# Spectral Norm Variance Regularization
A repository for paper of "Spectral Norm Variance Regularization for Balanced Layer-Wise Dynamics in Training Deep Neural Networks"

## Environment
```{bash}
conda env create -f environment.yml
conda activate snvr
```

## Dataset
We use CIFAR-10, CIFAR-100, and STL-10 datasets from torchvision datasets. 
To use Tiny-ImageNet and Food-101 datasets, follow shell scripts for dataset setup:
```{bash}
# make directory
mkdir -p ../data

# Tiny-ImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ../data
mv ../data/tiny-imagenet-200 ../data/tiny-imagenet
python tiny-imagenet.py
rm tiny-imagenet-200.zip

# Food-101
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz -C ../data
mv ../data/food-101 ../data/food101
python food101.py
rm food-101.tar.gz
```

## Experiment
```{bash}
# Baseline
# modify name in yaml file
python -m torch.distributed.launch --use_env main.py --cfg config/resnet34/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0

# SNVR
python -m torch.distributed.launch --use_env main.py --cfg config/resnet34/cifar100/snvr.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
```

## Usage
See `optimizer/snvr.py` for our implementation version of SNVR.
```{python}
from optimizer.snvr import SGDSNVR, AdamWSNVR

model = ...

# SGD-based optimizer
optimizer = SGDSNVR(model.parameters(), momentum=0.9, lr=0.1, weight_decay=5e-4, snvr=(0.01, 1.0))

# AdamW-based optimizer
optimizer = AdamWSNVR(model.parameters(), momentum=(0.9, 0.999), lr=0.001, weight_decay=0.05, snvr=(0.01, 1.0), decouple=True)
```
