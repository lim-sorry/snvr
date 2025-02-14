import sys
import torch
import torch.nn as nn

from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset
from torchvision.models import resnet34, convnext_tiny
from torchvision.datasets import CIFAR100, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import random

import argparse

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

def calc_sn_uv(weight, n_power=5, eps=1e-12):
    weight_mat = weight

    height = weight_mat.size(0)
    weight_mat = weight_mat.view(height, -1)

    u = torch.randn(1, weight_mat.shape[0]).cuda()  # move u to GPU
    for _ in range(n_power):
        v = torch.mm(u, weight_mat)  # [1, n]
        v = v / (torch.norm(v) + eps)
        u = torch.mm(v, weight_mat.T)  # [1, m]
        u = u / (torch.norm(u) + eps)

    sn = (torch.mul(torch.ones(1, weight_mat.shape[0]).cuda(), torch.sum(torch.mm(torch.mm(u, weight_mat), v.T)))).T
    uv = torch.mm(u.T, v)

    return sn, uv

class AdamWSNR(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.01, snr=0.0, snvr=0.0):
        super(AdamWSNR, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.snr = snr
        self.snvr = snvr
        
    def step(self):
        if self.snr:
            sn_list = []
            for group in self.param_groups:
                for param in group['params']:
                    param.snr = False
                    if param.grad is None:
                        continue
                    if param.dim() < 2:
                        continue
                    if param.size(0) == 1 or param.view(param.size(0), -1).size(1) == 1:
                        continue
                    param.snr = True
                    param.sn, param.uv = calc_sn_uv(param.data)
                    sn_list.append(param.sn)
            if self.snvr:
                n_layers = len(sn_list)
                sn_mean = sum(sn[0] for sn in sn_list) / n_layers
                len_coeff = (n_layers - 1) / n_layers


        for group in self.param_groups:
            params: list[torch.Tensor] = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            has_complex = self._init_group(
                group,
                params,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            for i, param in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                # update step
                step_t += 1

                # Perform step weight decay
                param.data.mul_(1 - lr * weight_decay)

                if param.snr:
                    sn, uv = param.sn, param.uv
                    snr = torch.reshape(torch.mul(sn, uv), param.shape)
                    # add snr * self.snr to grad
                    grad.data.add_(torch.mul(snr, self.snr))

                    if self.snvr != 0:
                        snvr = torch.reshape(torch.mul(torch.mul(sn - sn_mean, len_coeff), uv), param.shape)
                        # add snvr * self.snvr to grad
                        grad.data.add_(torch.mul(snvr, self.snvr))

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2.sqrt() if isinstance(bias_correction2, torch.Tensor) else np.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.data.addcdiv_(exp_avg, denom, value=-step_size)

def set_seed(seed):
    print(f"=====> Set the random seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--dataset', default='imagenet', help='cifar100 | imagenet-1k')
    parser.add_argument('--usage', default='1.0', type=str, help='training set usage ratio')
    parser.add_argument('--model', default='convnext-tiny', help='resnet34 | convnext-tiny | swin-v2-tiny')
    parser.add_argument('--lr', default='1e-3', type=str, help='learning rate')
    parser.add_argument('--snr', default='0.0', type=str, help='spectral norm regularization')
    parser.add_argument('--snvr', default='0.0', type=str, help='spectral norm variance regularization')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--warm_up', default=10, type=int, help='warm up epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--ckpt_path', default='test_ckpt', help='Checkpoint path')
    parser.add_argument('--data_path', default='..data', help='Data path')
    args = parser.parse_args()
    print(vars(args))

    seed = args.seed
    set_seed(seed)

    ckpt_path = f"{args.ckpt_path}/{args.dataset}_{args.usage}_{args.model}_{args.lr}_{args.snr}_{args.snvr}.pth"
    print(f"| Checkpoint path: {ckpt_path}")
    args.usage = float(args.usage)
    args.lr = float(args.lr)
    args.snr = float(args.snr)
    args.snvr = float(args.snvr)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        if seed in ckpt.keys():
            print(f"already trained\n\n")
            sys.exit(0)
    else:
        ckpt = dict()

    if args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        full_trainset = CIFAR100(root=f"{args.data_path}/cifar100", train=True, download=True)
        testset = CIFAR100(root=f"{args.data_path}/cifar100", train=False, download=True)
        num_classes = 100
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
        train_idx, _ = train_test_split(list(range(len(full_trainset))), train_size=args.usage, stratify=full_trainset.targets)
    else:
        train_idx = list(range(len(full_trainset)))
    print(f"| Usage ratio: {args.usage}")

    train_idx, valid_idx = train_test_split(train_idx, train_size=0.8, stratify=[full_trainset.targets[i] for i in train_idx], random_state=seed)
    print(f"| trainset idx: {train_idx[0]}")

    # set dataset
    trainset = CustomDataset(full_trainset, train_idx, transform=transform)
    validset = CustomDataset(full_trainset, valid_idx, transform=test_transform)
    testset.transform = test_transform

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(validset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)

    if args.model == 'resnet34':
        model = resnet34(num_classes=num_classes).cuda()
        if args.datset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).cuda()
            model.maxpool = nn.Identity().cuda()
    elif args.model == 'convnext-tiny':
        model = convnext_tiny(num_classes=num_classes).cuda()
    elif args.model == 'convnext-tiny-0.5':
        from torchvision.models.convnext import CNBlockConfig, _convnext
        block_setting = [
            CNBlockConfig(48, 96, 3),
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 9),
            CNBlockConfig(384, None, 3),
        ]
        model = _convnext(block_setting, 0.1, None, True, num_classes=num_classes).cuda()
    elif args.model == 'swin-v2-tiny-0.5':
        from torchvision.models.swin_transformer import _swin_transformer, PatchMergingV2, SwinTransformerBlockV2
        model =  _swin_transformer(
            patch_size=[4, 4],
            embed_dim=48,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 8],
            stochastic_depth_prob=0.2,
            weights=None,
            progress=True,
            block=SwinTransformerBlockV2,
            downsample_layer=PatchMergingV2,
            num_classes=num_classes,
        ).cuda()
    print(f"| Model: {args.model}")
    print(model)

    optimizer = AdamWSNR(model.parameters(), lr=args.lr, snr=args.snr, snvr=args.snvr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warm_up)
    criterion = nn.CrossEntropyLoss()

    history = dict(
        args=vars(args),
        train_loss=[],
        valid_loss=[],
        valid_acc=[],
        test_loss=[],
        test_acc=[],
        best_epoch={}
    )
    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(trainloader, ncols=80):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(trainloader)

        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in validloader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
                _, predicted = output.max(1)
                valid_total += target.size(0)
                valid_correct += predicted.eq(target).sum().item()

            valid_loss /= len(validloader)
            valid_acc = valid_correct / valid_total

            for data, target in testloader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

            test_loss /= len(testloader)
            test_acc = test_correct / test_total
            
        if valid_acc > best_acc:
            is_best = True
            best_acc = valid_acc
            history['best_epoch'] = dict(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_acc=valid_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        else:
            is_best = False 
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}{" *" if is_best else ""}')        
        
        if epoch < args.warm_up:
            lr_scale = min(1.0, float(epoch + 1) / args.warm_up)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale
        else:
            scheduler.step()

    ckpt[seed] = history
    torch.save(ckpt, ckpt_path)
