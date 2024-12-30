import os
import random
import torch
import torch.nn as nn
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
        if args.depth == '18':
            model = resnet18(num_classes=num_classes)
        if args.depth == '34':
            model = resnet34(num_classes=num_classes)

        # Modify the first convolutional layer to accept smaller images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove the maxpool layer

    elif 'vgg' in args.model:
        if args.depth == '16':
            model = vgg16_bn(num_classes=num_classes)
        if args.depth == '19':
            model = vgg19_bn(num_classes=num_classes)
        
        # Modify the avgpool layer to accept smaller images
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if 'clf' in args.model:            
            linear_dim = args.remark['linear_dim']
            model.classifier = nn.Sequential(
                nn.Linear(512, linear_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(linear_dim, linear_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(linear_dim, num_classes)
            )
        else:
            model.classifier = nn.Linear(512, num_classes)
        
        # if 'skip' in args.model, add skip connections from 
        # the output of the first input conv layer (with bn, relu) 
        # and the output of the last conv layer 
        # right before the classifier
        if 'skip' in args.model:
            class VGGWithSkipConnections(nn.Module):
                def __init__(self, model):
                    super(VGGWithSkipConnections, self).__init__()
                    self.input = model.features[:3]
                    self.features = model.features[3:]
                    self.features[-2] = nn.Identity()
                    self.skip = nn.Sequential(
                        nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(512)
                    )
                    self.avgpool = model.avgpool
                    self.relu = nn.ReLU(inplace=True)
                    self.classifier = model.classifier

                def forward(self, x):
                    x = self.input(x)
                    skip = self.skip(x)
                    skip = self.avgpool(skip)

                    x = self.features(x)
                    x = self.avgpool(x)
                    x = self.relu(x + skip)

                    x = torch.flatten(x, 1)
                    x = self.classifier(x)
                    return x

            model = VGGWithSkipConnections(model)

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
        from torchvision.models.convnext import CNBlockConfig, _convnext

        if args.depth == 'tiny':
            scale_factor = 0.5

            # block_setting 수정
            block_setting = [
                CNBlockConfig(int(96 * scale_factor), int(192 * scale_factor), 3),
                CNBlockConfig(int(192 * scale_factor), int(384 * scale_factor), 3),
                CNBlockConfig(int(384 * scale_factor), int(768 * scale_factor), 9),
                CNBlockConfig(int(768 * scale_factor), None, 3),
            ]

            stochastic_depth_prob = 0.1
            weights = None  # Pretrained weights를 사용하지 않음

            # 모델 생성
            model = _convnext(block_setting, stochastic_depth_prob, weights, progress=True, num_classes=num_classes)

            # features[0][0]의 Conv2d layer 수정
            if isinstance(model.features[0][0], nn.Conv2d):
                in_channels = 3
                out_channels = int(96 * scale_factor)
                model.features[0][0] = nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False
                )

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnext', help='Model name')
    parser.add_argument('--depth', type=str, default='tiny', help='Model depth')
    parser.add_argument('--remark', type=dict, default={}, help='Remark')
    args = parser.parse_args()
    
    model = getNetwork(args, 100)
    print(model)