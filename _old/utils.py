import torch
import random
import numpy as np
import os

def calc_sn_uv(weight, n_power=5, eps=1e-12):
    device = weight.device
    weight_mat = weight

    height = weight_mat.size(0)
    weight_mat = weight_mat.view(height, -1)

    u = torch.randn(1, weight_mat.shape[0]).to(device)  # move u to GPU
    for _ in range(n_power):
        v = torch.mm(u, weight_mat)  # [1, n]
        v = v / (torch.norm(v) + eps)
        u = torch.mm(v, weight_mat.T)  # [1, m]
        u = u / (torch.norm(u) + eps)
        
    sn = (torch.mul(torch.ones(1, weight_mat.shape[0]).to(device), torch.sum(torch.mm(torch.mm(u, weight_mat), v.T)))).T
    uv = torch.mm(u.T, v)

    return sn, uv

def set_seed(seed):
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
    # torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)