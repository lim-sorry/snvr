from __future__ import print_function
import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from os.path import join
from sgdsnr import SGDSNR
import config as cf
from utils import CustomDataset, train, test, getNetwork, save_args_to_file
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr',             type=float,      default=0.01,                         help='learning_rate')
parser.add_argument('--net-type',       type=str,        default='wide-resnet',                help='model')
parser.add_argument('--depth',          type=int,        default=28,                           help='depth of model')
parser.add_argument('--num-epochs',     type=int,        default=200,                          help='number of epochs')
parser.add_argument('--widen-factor',   type=float,      default=1,                           help='width of model')
parser.add_argument('--dataset',        type=str,        default='cifar100',                    help='dataset = [cifar10/cifar100]')
parser.add_argument('--usage',          type=float,      default=1.0,                          help='dataset usage')
parser.add_argument('--lr-sche',        type=str,        default='cosine',                       choices=['cosine'])
parser.add_argument('--weight-decay',   type=float,      default=1e-4) # 5e-4
parser.add_argument('--ckpt-path',      type=str,        default='',                            help='path to checkpoints')

parser.add_argument('--batch-size',   type=int,          default=128) # 5e-4
parser.add_argument('--datadir',        type=str,        default='',                            help='directory of dataset')
parser.add_argument('--optim-type',     type=str,        default='SGD',                        help='type of optimizer')
parser.add_argument('--seed',           type=int,        default=42) 

# spectral regularization related
parser.add_argument('--sg',                 type=float, default=0.01, help='spectrum regularization')
parser.add_argument('--svg',                type=float, default=0.0,  help='spectrum variance regularization')

args = parser.parse_args()

print(args)
# Save the arguments to a file
if os.path.exists(join(args.ckpt_path, "training_stats.npy")):
    exit()
save_args_to_file(args, join(args.ckpt_path, 'args.json'))
Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
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
    torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    

# seed fix
set_seed(args.seed)

# Data Loader
print('\n[Phase 1] : Data Preparation')
print(f"prepare preprocessing, {args.dataset}")

data_path = join(args.datadir, args.dataset)
if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset_full = torchvision.datasets.CIFAR10(root=data_path, train=True, 
                                            download=True, 
                                            transform=None)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, 
                                            download=False, 
                                            transform=None)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset_full = torchvision.datasets.CIFAR100(root=data_path, train=True, 
                                                download=True, 
                                                transform=None)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, 
                                                download=False, 
                                                transform=None)
    num_classes = 100

elif(args.dataset == 'svhn'):
    print("| Preparing SVHN dataset...")
    sys.stdout.write("| ")
    trainset_full = torchvision.datasets.SVHN(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=None)
    testset = torchvision.datasets.SVHN(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=None)
    num_classes = 10
    
elif(args.dataset == 'tiny-imagenet-200'): 
    print("| Preparing tiny-imagenet-200 dataset...")
    trainset_full = datasets.ImageFolder(os.path.join(data_path, 'train'), None)
    testset = datasets.ImageFolder(os.path.join(data_path, 'val'), None)
    num_classes = 200
else:
    raise NotImplementedError

# 전체 데이터셋 인덱스를 usage 비율에 따라 stratify하게 분할
if args.usage != 1.0:
    usage_idx, _ = train_test_split(
        range(len(trainset_full)),
        test_size=1.0 - args.usage,
        stratify=trainset_full.targets
    )
else:
    usage_idx = range(len(trainset_full))

# usage_idx로부터 다시 8:2 비율(train:valid)로 stratify 분할
train_idx, valid_idx = train_test_split(
    usage_idx,
    test_size=0.2,
    stratify=[trainset_full.targets[i] for i in usage_idx]
)

print('| Usage ratio:', args.usage)
print('| Train idx :', train_idx[0], ', valid idx :', valid_idx[0])

# 트레이닝 세트의 평균과 표준편차 계산
trainset_data = []
if args.dataset == 'tiny-imagenet-200':
    for idx in train_idx:
        trainset_data.append(np.array(trainset_full[idx][0]))  # 이미지를 numpy array로 변환
else:
    for idx in train_idx:
        trainset_data.append(trainset_full.data[idx])
trainset_data = (np.stack(trainset_data) / 255.0).reshape(-1,3)
train_mean = np.mean(trainset_data, axis=0)
train_std = np.std(trainset_data, axis=0)
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

# transform 적용
trainset = CustomDataset(trainset_full, train_idx, transform_train)
validset = CustomDataset(trainset_full, valid_idx, transform_test)
testset.transform = transform_test

trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            num_workers=6)
validloader = torch.utils.data.DataLoader(validset,
                                            batch_size=cf.eval_batchsize[args.dataset],
                                            shuffle=False,
                                            num_workers=4)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=cf.eval_batchsize[args.dataset], 
                                            shuffle=False, 
                                            num_workers=4)

# 첫 번째 데이터 추출
first_train_data = next(iter(trainloader))
first_valid_data = next(iter(validloader))
print("| First train label :", first_train_data[1][0].item(), ', valid label : ', first_valid_data[1][0].item())

# Model
print('\n[Phase 2] : Model setup')
print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args, num_classes)

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    
criterion = nn.CrossEntropyLoss()
# print(net)

if args.optim_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), 
                    lr=args.lr,  
                    momentum=0.9, 
                    weight_decay=args.weight_decay)
elif args.optim_type == 'SGDSNR':
    optimizer = SGDSNR(net.parameters(), 
                    lr=args.lr,
                    momentum=0.9, 
                    weight_decay=args.weight_decay, 
                    spectrum_regularization=args.sg,
                    spectrum_variance_regularization=args.svg)
elif args.optim_type == 'Adam':
    optimizer = optim.Adam(net.parameters(), 
                        lr=args.lr,  
                        weight_decay=args.weight_decay) 
else:
    raise NotImplementedError
    
# set a basic global learning rate scheduler
if args.lr_sche == 'cosine':
    lr_schedule = cf.cosine_decay
else:
    raise NotImplementedError

test_acc, test_loss = test(epoch=0, net=net, testloader=testloader, criterion=criterion)
print(f"| Init model\tTest Loss: {test_loss:.4f} Acc@1: {test_acc:5.2f}%")

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type) + '\n')

best_acc = 0
start_epoch = cf.start_epoch
elapsed_time = 0
training_stats = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'test_loss': [],
    'test_acc': [],
    'current_lr':[],
    'schedule_next_lr':[],
    'epoch_time':[],
    'elapsed_time':[],
    'train_idx':train_idx,
    'valid_idx':valid_idx,
}

untuned_lr = args.lr
for epoch in range(start_epoch, start_epoch+args.num_epochs):
    epoch_start_time = time.time()

    # this is current LR
    current_lr = untuned_lr
    train_acc, train_loss = \
                train(epoch, net, args.num_epochs, trainloader, criterion, optimizer)
    valid_acc, valid_loss = \
                test(epoch, net, validloader, criterion)
    test_acc, test_loss = \
                test(epoch, net, testloader, criterion)
    print(f"| Epoch #{epoch}\tTrain Loss: {train_loss:.4f} Acc@1: {train_acc:5.2f}% | Valid Loss: {valid_loss:.4f} Acc@1: {valid_acc:5.2f}% | Test Loss: {test_loss:.4f} Acc@1: {test_acc:5.2f}%")
    
    # save best
    if valid_acc > best_acc:
        print('| Saving Best ...')
        state = {
            'net': net.state_dict(),
            'train_loss':train_loss,
            'train_acc':train_acc,
            'valid_loss':valid_loss,
            'valid_acc':valid_acc,
            'test_loss':test_loss,
            'test_acc':test_acc,
            'epoch':epoch
        }
        best_acc = valid_acc
        torch.save(state, join(args.ckpt_path, f'epoch_best.ckpt'))
    
    untuned_lr = \
        lr_schedule(args.lr, epoch, args.num_epochs)
    
    # scheduling by default
    for param_group in optimizer.param_groups:
        param_group['lr'] = untuned_lr
            
    epoch_time = time.time() - epoch_start_time
    elapsed_time += epoch_time
    
    training_stats['train_loss'].append(train_loss)
    training_stats['train_acc'].append(train_acc)
    training_stats['valid_loss'].append(valid_loss)
    training_stats['valid_acc'].append(valid_acc)
    training_stats['test_loss'].append(test_loss)
    training_stats['test_acc'].append(test_acc)
    training_stats['current_lr'].append(current_lr)
    training_stats['schedule_next_lr'].append(untuned_lr)
    training_stats['epoch_time'].append(epoch_time)
    training_stats['elapsed_time'].append(elapsed_time)
    print()
    
np.save(join(args.ckpt_path, "training_stats.npy"), training_stats)
        