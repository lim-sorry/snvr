import os
import random
import torch
from networks import *
from operator import itemgetter
import numpy as np
import math
import tqdm
import time
import pandas as pd
import json
import logging
from torch.utils.data import Dataset

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

def train(model, trainloader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0 
    start_time = time.time()
    pbar = tqdm.tqdm(trainloader, desc="| Training", ncols=150)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    
    pbar.close()
    train_time = time.time() - start_time
    train_loss /= total
    train_acc = 100. * correct / total
    train_acc = train_acc.item()

    return train_acc, train_loss, train_time


# Testing
def test(model, testloader, criterion):
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        acc = acc.item()
        test_loss = test_loss/total
        
    return acc, test_loss


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
    

from torchvision.models import resnet18, resnet34, vgg16_bn, vgg19_bn

def getNetwork(args, num_classes):
    if args.model == 'resnet':
        if args.depth == 18:
            model = resnet18(num_classes=num_classes)
        if args.depth == 34:
            model = resnet34(num_classes=num_classes)

        # Modify the first convolutional layer to accept smaller images
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()  # Remove the maxpool layer
        
        raise ValueError('Initialization Not implemented yet')

    elif 'vgg' in args.model:
        if args.depth == 16:
            model = vgg16_bn(num_classes=num_classes)
        if args.depth == 19:
            model = vgg19_bn(num_classes=num_classes)
        
        # Modify the avgpool layer to accept smaller images
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if 'clf' in args.model:            
            linear_dim = args.remark['linear_dim']
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(512, linear_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(linear_dim, linear_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(linear_dim, num_classes)
            )
        else:
            model.classifier = torch.nn.Linear(512, num_classes)
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

    if args.model == 'convnext':
        from torchvision.models import convnext_tiny
        model = convnext_tiny(num_classes=num_classes)

    return model
