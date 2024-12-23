from __future__ import print_function
import json
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from os.path import join
from sgdsnr import SGDSNR
import config as cf
from utils import CustomDataset, set_seed, train, test, getNetwork
from sklearn.model_selection import train_test_split

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tiny-imagenet-200', help='cifar10 | cifar100 | svhn | tiny-imagenet-200')
parser.add_argument('--usage', default=0.5, type=float, help='Usage ratio of the training set')
parser.add_argument('--model', default='resnet', help='resnet')
parser.add_argument('--depth', default=18, type=int, help='depth of the network')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--snr', default=0.0, type=float, help='spectrum regularization')
parser.add_argument('--snvr', default=0.0, type=float, help='spectrum variance regularization')
parser.add_argument('--seed', default=42, type=int, help='Random seed')

parser.add_argument('--ckpt_path', default='ckpt/test', help='Checkpoint path')
parser.add_argument('--data_path', default='data', help='Data path')
parser.add_argument('--remark', default='', help='Remark for the experiment')

# default arguments
parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--optim_type', default='SGDSNR', help='SGD | SGDSNR | Adam')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight Decay')
parser.add_argument('--lr_sche', default='cosine', help='cosine')

args = parser.parse_args()
print(args)

# check if already trained in training_stats.pth
if Path(join(args.ckpt_path, "training_stats.pth")).exists():
    # check if epoch is already finished
    training_stats = torch.load(join(args.ckpt_path, "training_stats.pth"))
    if training_stats['epoch'] == args.num_epochs:
        print(f"seed {args.seed} already trained")
        sys.exit(0)
else:
    os.makedirs(args.ckpt_path, exist_ok=True)

# save args to file
with open(join(args.ckpt_path, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

# set seed
set_seed(args.seed)

# Data Preparation
print('\n[Phase 1] : Data Preparation')
print(f"| Preparing {args.dataset} dataset")
dataset_path = join(args.data_path, args.dataset)
if args.dataset == 'cifar10':
    num_classes = 10
    full_trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR10(root=dataset_path, train=False, download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    full_trainset = datasets.CIFAR100(root=dataset_path, train=True, download=True)
    testset = datasets.CIFAR100(root=dataset_path, train=False, download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    full_trainset = datasets.SVHN(root=dataset_path, split='train', download=True)
    testset = datasets.SVHN(root=dataset_path, split='test', download=True)
elif args.dataset == 'tiny-imagenet-200':
    num_classes = 200
    full_trainset = datasets.ImageFolder(root=join(dataset_path, 'train'))
    testset = datasets.ImageFolder(root=join(dataset_path, 'val'))
else:
    raise ValueError('Dataset not found')

# split stratified trainset based on usage ratio
print(f"| Splitting trainset with usage ratio {args.usage}")
if args.usage < 1.0:
    train_idx, _ = train_test_split(list(range(len(full_trainset))), train_size=args.usage, stratify=full_trainset.targets)
else:
    train_idx = list(range(len(full_trainset)))

# split trainset into trainset and validset with 80:20 ratio
train_idx, valid_idx = train_test_split(train_idx, train_size=0.8, stratify=[full_trainset.targets[i] for i in train_idx])

# check first idx of trainset and validset
print(f"| trainset idx: {train_idx[:5]}")
print(f"| validset idx: {valid_idx[:5]}")

# calculate mean and std of trainset for normalization
train_images = []
if args.dataset == 'tiny-imagenet-200':
    for idx in train_idx:
        train_images.append(np.array(full_trainset[idx][0]))
else:
    for idx in train_idx:
        train_images.append(full_trainset.data[idx])
train_images = (np.stack(train_images) / 255.0).reshape(-1,3)
train_mean = np.mean(train_images, axis=0)
train_std = np.std(train_images, axis=0)
print('| Train mean : ', train_mean, ', std : ', train_std)

# set transform
transform_train = transforms.Compose([
    transforms.RandomCrop(cf.crop_size[args.dataset], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

# set dataset
trainset = CustomDataset(full_trainset, train_idx, transform=transform_train)
validset = CustomDataset(full_trainset, valid_idx, transform=transform_test)
testset.transform = transform_test

# set dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.train_batchsize[args.dataset], shuffle=True, num_workers=4)
validloader = torch.utils.data.DataLoader(validset, batch_size=cf.eval_batchsize[args.dataset], shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=cf.eval_batchsize[args.dataset], shuffle=False, num_workers=4)

# extract first train and valid data
first_train_data = next(iter(trainloader))
first_valid_data = next(iter(validloader))
print("| First train label :", first_train_data[1][0].item(), ', valid label : ', first_valid_data[1][0].item())

# Model
print('\n[Phase 2] : Model setup')
print(f"| Building {args.model} {args.depth}")
model = getNetwork(args, num_classes)
model = model.cuda()
print(model)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
if args.optim_type == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim_type == 'SGDSNR':
    optimizer = SGDSNR(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, snr=args.snr, snvr=args.snvr)
else:
    raise ValueError('Optimizer not found')

# Learning Rate Scheduler
if args.lr_sche == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

# Training
print('\n[Phase 3] : Training')
print("| Training Epochs = ", args.num_epochs)
print("| Initial Learning Rate = ", args.lr)
print("| Optimizer = ", args.optim_type)
print("| Scheduler = ", args.lr_sche)

# save training stats
training_stats = dict(
    epoch=0,
    train_acc=[],
    train_loss=[],
    valid_acc=[],
    valid_loss=[],
    test_acc=[],
    test_loss=[],
    train_time=[],
    train_idx=train_idx,
    valid_idx=valid_idx
)
best_valid_acc = 0.0

# epoch loop
for epoch in range(args.num_epochs):
    train_acc, train_loss, train_time = train(model, trainloader, criterion, optimizer)
    valid_acc, valid_loss = test(model, validloader, criterion)
    test_acc, test_loss = test(model, testloader, criterion)


    if valid_acc > best_valid_acc:
        is_best = True
        best_valid_acc = valid_acc
        best_epoch = dict(
            epoch=epoch+1,
            model=model.state_dict(),
            train_acc=train_acc,
            train_loss=train_loss,
            valid_acc=valid_acc,
            valid_loss=valid_loss,
            test_acc=test_acc,
            test_loss=test_loss,
        )
    else:
        is_best = False

    # save training stats
    training_stats['epoch'] = epoch + 1
    training_stats['train_acc'].append(train_acc)
    training_stats['train_loss'].append(train_loss)
    training_stats['valid_acc'].append(valid_acc)
    training_stats['valid_loss'].append(valid_loss)
    training_stats['test_acc'].append(test_acc)
    training_stats['test_loss'].append(test_loss)
    training_stats['train_time'].append(train_time)

    # print stats
    print(f"| Epoch {epoch+1}/{args.num_epochs} | Train Acc: {train_acc:.2f} | Train Loss: {train_loss:.4f} | Valid Acc: {valid_acc:.2f} | Valid Loss: {valid_loss:.4f} | Test Acc: {test_acc:.2f} | Test Loss: {test_loss:.4f} {' best' if is_best else ''}\n")

    # update learning rate
    if args.lr_sche == 'cosine':
        scheduler.step()

# save training stats
torch.save(best_epoch, join(args.ckpt_path, "best_epoch.pth"))
torch.save(training_stats, join(args.ckpt_path, "training_stats.pth"))