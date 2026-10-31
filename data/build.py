import os
import torch
import numpy as np
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.models.layers import to_2tuple

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler, EpochShuffleSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
    timm_transforms._pil_interpolation_to_str = {
        InterpolationMode.BILINEAR: 'BILINEAR',
        InterpolationMode.BICUBIC: 'BICUBIC',
        InterpolationMode.LANCZOS: 'LANCZOS',
        InterpolationMode.HAMMING: 'HAMMING',
    }
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f'training dataset: {len(dataset_train)} images')
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f'validation dataset: {len(dataset_val)} images')

    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        sampler_train = SubsetRandomSampler(np.arange(len(dataset_train)))
    else:
        sampler_train = EpochShuffleSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    dataset_path = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET)
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'cifar10':
        dataset = datasets.CIFAR10(root=dataset_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(root=dataset_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'tiny-imagenet':
        dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path, 'train_preprocess' if is_train else 'valid_preprocess'),
            transform=transform
        )
        nb_classes = 200
    elif config.DATA.DATASET == 'stl10':
        dataset = datasets.STL10(
            root=dataset_path, split='train' if is_train else 'test', download=True, transform=transform
        )
        nb_classes = 10
    elif config.DATA.DATASET == 'food101':
        dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path, 'train' if is_train else 'test'),
            transform=transform
        )
        nb_classes = 101
    elif config.DATA.DATASET == 'eurosat':
        # EuroSAT RGB (10 classes, 64x64), split 8:2 into train / test by eurosat.py
        dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path, 'train' if is_train else 'test'),
            transform=transform
        )
        nb_classes = 10
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    if is_train and config.DATA.SUBSET_RATIO < 1.0:
        dataset = stratified_subset(dataset, config.DATA.SUBSET_RATIO, config.SEED, config.OUTPUT)

    return dataset, nb_classes


def stratified_subset(dataset, ratio, seed, output_dir):
    """Keep round(ratio * n_c) samples of every class c, drawn with the given seed. The chosen
    files are written to <output_dir>/train_subset.txt."""
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(targets):
        idx_c = np.flatnonzero(targets == c)
        idx.extend(rng.choice(idx_c, size=int(round(ratio * len(idx_c))), replace=False).tolist())
    idx.sort()
    if output_dir and hasattr(dataset, 'samples'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'train_subset.txt'), 'w') as f:
            f.writelines(os.path.relpath(dataset.samples[i][0], dataset.root) + '\n' for i in idx)
    return torch.utils.data.Subset(dataset, idx)


def build_transform(is_train, config):
    if is_train:
        if config.AUG.RANDOM_AUGMENT == True:
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=config.DATA.IMG_SIZE // 8)
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(config.DATA.IMG_SIZE, padding=config.DATA.IMG_SIZE // 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        if config.DATA.DATASET in ('food101',):
            # images of varying size: resize the short side to 1.125x, then crop (128 -> 144, 96 -> 108)
            transform.transforms[0] = transforms.Resize(int(config.DATA.IMG_SIZE * 1.125), interpolation=_pil_interp(config.DATA.INTERPOLATION))
            transform.transforms.insert(1, transforms.RandomCrop(config.DATA.IMG_SIZE))
        return transform

    t = []
    if config.DATA.DATASET in ('food101',):
        t.append(transforms.Resize(int(config.DATA.IMG_SIZE * 1.125), interpolation=_pil_interp(config.DATA.INTERPOLATION)))
        t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
