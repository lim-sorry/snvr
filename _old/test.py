import torch
import torch.nn as nn

from utils import calc_sn_uv

def calc_sn_uv_2(weight, n_power=1, eps=1e-12):
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
    sn = torch.mm(torch.mm(u, weight_mat), v.T).expand([height, 1])
    uv = torch.mm(u.T, v)

    return sn, uv

m = nn.Conv2d(3, 64, 3, 1, 1)
sn = torch.nn.utils.parametrizations.spectral_norm(m)
print(torch.linalg.matrix_norm(sn.weight, 2).shape)

sn, uv = calc_sn_uv(sn.weight)
print(sn[0])