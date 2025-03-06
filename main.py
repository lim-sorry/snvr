import sys
import time
import torch
import torch.nn as nn

from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset
from torchvision.models import resnet34, convnext_tiny
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import random

import argparse

from models import get_model
from snvr import AdamSNR, AdamWSNR
from utils import set_seed

class CustomDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--dataset', default='cifar100', help='cifar100 | imagenet-1k')
    parser.add_argument('--usage', default='1.0', type=str, help='training set usage ratio')
    parser.add_argument('--model', default='resnet_34', help='resnet34 | convnext-tiny | swin-v2-tiny')
    parser.add_argument('--lr', default='1e-3', type=str, help='learning rate')
    parser.add_argument('--snr', default='0.0', type=str, help='spectral norm regularization')
    parser.add_argument('--snvr', default='0.0', type=str, help='spectral norm variance regularization')
    parser.add_argument('--weight_decay', default='0.0', type=str, help='weight decay')
    
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--warm_up', default=10, type=int, help='warm up epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--optim', default='adam', help='adam | adamw')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--ckpt_path', default='ckpt', help='Checkpoint path')
    parser.add_argument('--data_path', default='..data', help='Data path')

    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    overwrite = False
    model_save = True
    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"| Device: {device}")

    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    ckpt_path = f"{args.ckpt_path}/{args.dataset}_{args.usage}_{args.model}_{args.lr}_{args.snr}_{args.snvr}_{args.weight_decay}.pth"
    print(f"| Checkpoint path: {ckpt_path}")
    args.usage = float(args.usage)
    args.lr = float(args.lr)
    args.snr = float(args.snr)
    args.snvr = float(args.snvr)
    args.weight_decay = float(args.weight_decay)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        if seed in ckpt.keys() and not overwrite:
            print(f"already trained\n\n")
            sys.exit(0)
    else:
        ckpt = dict()

    if args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        full_trainset = CIFAR100(root=f"{args.data_path}/cifar100", train=True, download=True)
        testset = CIFAR100(root=f"{args.data_path}/cifar100", train=False, download=True)
        num_classes = 100
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435,0.2616)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435,0.2616)),
        ])
        full_trainset = CIFAR10(root=f"{args.data_path}/cifar10", train=True, download=True)
        testset = CIFAR10(root=f"{args.data_path}/cifar10", train=False, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        full_trainset = ImageFolder(root=f"{args.data_path}/imagenet/train")
        testset = ImageFolder(root=f"{args.data_path}/imagenet/val")
        num_classes = 1000
    print(f"| Preparing {args.dataset} dataset")
            
    if args.usage < 1.0:
        train_idx, _ = train_test_split(list(range(len(full_trainset))), train_size=args.usage, stratify=full_trainset.targets, random_state=seed)
    else:
        train_idx = list(range(len(full_trainset)))
    print(f"| Usage ratio: {args.usage}")

    train_idx, valid_idx = train_test_split(train_idx, train_size=0.9, stratify=[full_trainset.targets[i] for i in train_idx], random_state=seed)
    print(f"| trainset idx: {train_idx[0]} {train_idx[1]}")

    # set dataset
    trainset = CustomDataset(full_trainset, train_idx, transform=transform)
    validset = CustomDataset(full_trainset, valid_idx, transform=test_transform)
    testset.transform = test_transform

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(validset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)

    model = get_model(model_name=args.model, dataset=args.dataset, num_classes=num_classes).to(device)
    print(f"| Model: {args.model}")
    # print(model)

    if args.optim == 'adam':
        from torch.optim import Adam
        optimizer = AdamSNR(model.parameters(), lr=args.lr, snr=args.snr, snvr=args.snvr, weight_decay=args.weight_decay)
        # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = AdamWSNR(model.parameters(), lr=args.lr, snr=args.snr, snvr=args.snvr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warm_up)
    criterion = nn.CrossEntropyLoss()

    history = dict(
        args=vars(args),
        train_loss=[],
        valid_loss=[],
        valid_top1_acc=[],
        valid_top5_acc=[],
        test_loss=[],
        test_top1_acc=[],
        test_top5_acc=[],
        best_epoch={}
    )
    best_top1_acc = 0

    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(trainloader)

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_total = 0
            valid_top1_acc = 0
            valid_top5_acc = 0
            
            for data, target in validloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
                _, predicted = output.max(1)
                valid_total += target.size(0)
                valid_top1_acc += predicted.eq(target).sum().item()
                _, top5_predicted = output.topk(5, 1, True, True)
                valid_top5_acc += top5_predicted.eq(target.view(-1, 1).expand_as(top5_predicted)).sum().item()

            valid_loss /= len(validloader)
            valid_top1_acc = valid_top1_acc / valid_total
            valid_top5_acc = valid_top5_acc / valid_total

            test_loss = 0
            test_total = 0
            test_top1_acc = 0
            test_top5_acc = 0

            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_top1_acc += predicted.eq(target).sum().item()
                _, top5_predicted = output.topk(5, 1, True, True)
                test_top5_acc += top5_predicted.eq(target.view(-1, 1).expand_as(top5_predicted)).sum().item()

            test_loss /= len(testloader)
            test_top1_acc = test_top1_acc / test_total
            test_top5_acc = test_top5_acc / test_total
            
        if valid_top1_acc > best_top1_acc:
            is_best = True
            best_top1_acc = valid_top1_acc
            history['best_epoch'] = dict(
                epoch=epoch,
                valid_loss=valid_loss,
                valid_top1_acc=valid_top1_acc,
                valid_top5_acc=valid_top5_acc,
                test_loss=test_loss,
                test_top1_acc=test_top1_acc,
                test_top5_acc=test_top5_acc
            )
            best_model = model.state_dict()
        else:
            is_best = False 

        if epoch < args.warm_up:
            lr_scale = min(1.0, float(epoch + 1) / args.warm_up)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale
        else:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_top1_acc'].append(valid_top1_acc)
        history['valid_top5_acc'].append(valid_top5_acc)
        history['test_loss'].append(test_loss)
        history['test_top1_acc'].append(test_top1_acc)
        history['test_top5_acc'].append(test_top5_acc)

        # hour, minute left
        time_elapsed = time.time() - time_start
        time_left = time_elapsed * (args.epochs - epoch - 1) / (epoch + 1)
        time_string = f'{int(time_elapsed // 3600):2d}:{int((time_elapsed % 3600) // 60):02d}/{int(time_left // 3600):2d}:{int((time_left % 3600) // 60):02d}'
       
        print(f' Epoch {epoch:3d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Acc@1: {valid_top1_acc:.4f}, Acc@5: {valid_top5_acc:.4f}, Test Loss: {test_loss:.4f}, Acc@1: {test_top1_acc:.4f}, Acc@5: {test_top5_acc:.4f}, Time: {time_string} {"*" if is_best else ""}')

    ckpt[seed] = history
    torch.save(ckpt, ckpt_path)
    if model_save:
        torch.save(best_model, ckpt_path.replace('.pth', '_model.pth'))
