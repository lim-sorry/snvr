import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from networks import *

net = ResNet(18, 100, 1)
print(len(list(net.parameters())))
