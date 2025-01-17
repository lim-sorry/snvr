# Description: Configuration file for the project

resize = {
    'cifar10': 32,
    'cifar100': 32,
    'svhn': 32,
    'tiny-imagenet-200': 64,
    'imagenet100': 256,
    }

padding = {
    'cifar10': 4,
    'cifar100': 4,
    'svhn': 4,
    'tiny-imagenet-200': 8,
    'imagenet100': None,
    }

crop_size = {
    'cifar10': 32,
    'cifar100': 32,
    'svhn': 32,
    'tiny-imagenet-200': 64,
    'imagenet100': 224,
    }

train_batchsize = {
    'cifar10': 128,
    'cifar100': 128,
    'svhn': 128,
    'tiny-imagenet-200': 128
}

eval_batchsize = {
    'cifar10': 500,
    'cifar100': 500,
    'svhn': 500,
    'tiny-imagenet-200': 500
}