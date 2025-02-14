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
    'tiny-imagenet-200': 128,
    'imagenet100': 128,
}

eval_batchsize = {
    'cifar10': 500,
    'cifar100': 500,
    'svhn': 500,
    'tiny-imagenet-200': 500,
    'imagenet100': 500,
}

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'svhn': (0.4377, 0.4438, 0.4728),
    'tiny-imagenet-200': (0.4802, 0.4481, 0.3975),
    'imagenet100': (0.485, 0.456, 0.406),
}

std = {
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'svhn': (0.1201, 0.1231, 0.1052),
    'tiny-imagenet-200': (0.2302, 0.2265, 0.2262),
    'imagenet100': (0.229, 0.224, 0.225),
}
