import torch
import torch.nn as nn

@torch.no_grad()
def spectral_bound_weight(W: torch.Tensor, coeff: float = 1.0, eps: float = 1e-12):
    """
    W : (out_features, in_features, ...) 형태의 weight 텐서
        Linear -> (out, in)
        Conv2d -> (out_channels, in_channels, kH, kW)
    coeff : 원하는 upper bound (1-Lipschitz면 1.0)
    """

    W_mat = W.view(W.size(0), -1)  # [out_dim, in_dim_flat]

    # ||W||_1 = max_j sum_i |a_ij|
    col_sums = W_mat.abs().sum(dim=0)      # [in_dim_flat]
    norm_1 = col_sums.max()

    # ||W||_∞ = max_i sum_j |a_ij|
    row_sums = W_mat.abs().sum(dim=1)      # [out_dim]
    norm_inf = row_sums.max()

    # spectral upper bound
    spec_upper = torch.sqrt(norm_1 * norm_inf + eps)

    if spec_upper > coeff:
        scale = coeff / spec_upper
        W.mul_(scale)
        return scale.item()
    else:
        return 1.0


@torch.no_grad()
def apply_spectral_bound_model(model: nn.Module, coeff: float = 1.0):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(m, "weight") and m.weight is not None:
                spectral_bound_weight(m.weight, coeff=coeff)
