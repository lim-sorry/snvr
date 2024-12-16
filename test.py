import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset_full = torchvision.datasets.CIFAR100(root='data/cifar100', train=True, 
                                            download=True, 
                                            transform=transform_train)
num_classes = 100

# 각 클래스별로 데이터를 동일한 비율로 추출
targets = trainset_full.targets
class_indices = {class_idx: [] for class_idx in range(num_classes)}
for idx, label in enumerate(targets):
    class_indices[label].append(idx)

# 각 클래스별 인덱스 리스트에서 절반을 랜덤으로 선택
reduced_indices = []
for indices in class_indices.values():
    reduced_indices.extend(torch.utils.data.random_split(indices, [len(indices) // 2, len(indices) - len(indices) // 2])[0])

# 선택된 인덱스로 새로운 서브셋 생성
trainset_reduced = Subset(trainset_full, reduced_indices)

trainloader = DataLoader(trainset_reduced,
                            batch_size=128,
                            shuffle=True,
                            num_workers=6)
