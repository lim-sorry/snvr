from functools import partial
from torch import optim as optim

try:
    from .snvr import AdamWSNVR, SGDSNVR
except:
    from snvr import AdamWSNVR, SGDSNVR

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'sgdsnvr':
        parameters = set_spectral_norm(parameters, separate=config.TRAIN.OPTIMIZER.SEPARATE, skip=config.TRAIN.OPTIMIZER.SKIP)
        optimizer = SGDSNVR(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                            lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                            snvr=config.TRAIN.OPTIMIZER.SNVR, decouple=config.TRAIN.OPTIMIZER.DECOUPLE)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamwsnvr':
        parameters = set_spectral_norm(parameters, separate=config.TRAIN.OPTIMIZER.SEPARATE, skip=config.TRAIN.OPTIMIZER.SKIP)
        optimizer = AdamWSNVR(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                             lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                             snvr=config.TRAIN.OPTIMIZER.SNVR, decouple=config.TRAIN.OPTIMIZER.DECOUPLE)
    elif 'lwsam' in opt_lower:
        parameters = group_parameters_by_layer_type(model)
        if 'sgd' in opt_lower:
            base_optimizer = optim.SGD
            optimizer = LWSAM(parameters, base_optimizer=base_optimizer, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                             lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        elif 'adamw' in opt_lower:
            base_optimizer = optim.AdamW
            optimizer = LWSAM(parameters, base_optimizer=base_optimizer, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                             lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())

def set_spectral_norm(parameters, separate=True, skip=True):
    parameters_sn = []
    for group in parameters:
        if 'weight_decay' in group:
            group['snvr'] = (0.0, 0.0)
            parameters_sn.append(group)
            continue

        if skip:
            assert len(group['params']) >= 2, "Parameter group must have at least two parameters to skip"
            group_skip = {'params':[group['params'][0],group['params'][-1]]}
            group_skip['snvr'] = (0.0, 0.0)
            parameters_sn.append(group_skip)
            group['params'] = group['params'][1:-1]

        if separate:
            group_conv = {'params':[]}
            group_linear = {'params':[]}
            for param in group['params']:
                if param.dim() == 4:
                    group_conv['params'].append(param)
                elif param.dim() == 2:
                    group_linear['params'].append(param)
            parameters_sn.append(group_conv) if group_conv['params'] else None
            parameters_sn.append(group_linear) if group_linear['params'] else None
        else:
            parameters_sn.append(group)

    for group in parameters_sn:
        group['num'] = len(group['params'])
    
    return parameters_sn
        
if __name__ == "__main__":  
    from torchvision.models import resnet18
    import torch
    import torch.nn as nn   

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 128),
        nn.Linear(128, 10),
        nn.Linear(10, 2),

    )
    
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)
    parameters = set_spectral_norm(parameters, separate=True, skip=True)

    optimizer = AdamWSNVR(parameters, lr=1.0, weight_decay=0.5, snvr=(1.0, 2.0))
    print(optimizer)
    for group in optimizer.param_groups:
        print("Number of parameter groups:", len(group['params']))
    
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    loss = y.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


