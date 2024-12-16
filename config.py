############### Pytorch CIFAR configuration file ###############
import math
import numpy as np
start_epoch = 1

crop_size = {
    'cifar10': 32,
    'cifar100': 32,
    'svhn': 32,
    'tiny-imagenet-200': 64,
    }
eval_batchsize = {
    'cifar10': 500,
    'cifar100': 500,
    'svhn': 500,
    'tiny-imagenet-200': 200
}

def cosine_decay(init, epoch, total_epoch):
    epoch = min(epoch, total_epoch)
    cosine_decay = 0.5 * (1 + math.cos(np.pi * epoch / total_epoch))
    
    return init * cosine_decay




def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
