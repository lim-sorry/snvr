import torch
import torch.nn as nn


def _regularized_weights(model):
    return [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _matrix_view(weight):
    return weight.reshape(weight.size(0), -1)


def _singular_values(weight):
    return torch.linalg.svdvals(_matrix_view(weight))


def _hill_alpha(eigs, eps=1e-12):
    eigs = eigs.sort().values.clamp_min(eps)
    n = eigs.numel()
    if n < 2:
        return eigs.new_tensor(1.0)

    k = max(1, n // 2)
    threshold_idx = n - k - 1
    threshold = (
        eigs[threshold_idx].clamp_min(eps)
        if threshold_idx >= 0
        else eigs[0].clamp_min(eps)
    )
    top_k = eigs[-k:].clamp_min(eps)
    denom = torch.log(top_k / threshold).sum().clamp_min(eps)
    return 1.0 + k / denom


def _reduce_terms(terms, reduction):
    if not terms:
        return None
    stacked = torch.stack(terms)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def weight_alpha(model, reduction="sum"):
    """Weighted Alpha Regularization from Xiao et al. (2023).

    The penalty for each layer is alpha * log(lambda_max), where alpha is the
    Hill estimator of the tail index and lambda_max is the largest eigenvalue of
    W^T W, equivalently the squared largest singular value of W.
    """
    terms = []
    for weight in _regularized_weights(model):
        eigs = _singular_values(weight).pow(2)
        alpha = _hill_alpha(eigs)
        terms.append(alpha * torch.log(eigs.max().clamp_min(1e-12)))
    regularization = _reduce_terms(terms, reduction)
    if regularization is not None:
        return regularization
    return next(model.parameters()).new_tensor(0.0)


def stable_rank(model, reduction="sum"):
    """Stable Rank Regularization from Xiao et al. (2023)."""
    terms = []
    for weight in _regularized_weights(model):
        singular_values = _singular_values(weight)
        stable = singular_values.pow(2).sum() / singular_values.max().pow(2).clamp_min(1e-12)
        terms.append(stable)
    regularization = _reduce_terms(terms, reduction)
    if regularization is not None:
        return regularization
    return next(model.parameters()).new_tensor(0.0)


if __name__ == "__main__":
    x = torch.randn(4, 4, 3, 3)
    y = torch.randn(4, 128)

    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(16 * 3 * 3, 128),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)

    y_hat = model(x)
    loss = (y_hat - y).mean()
    print(loss)
    loss += 0.001 * stable_rank(model)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
