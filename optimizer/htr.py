"""Heavy-Tailed Regularization (Xiao et al. 2023): Weighted Alpha Regularization (WAR) and Stable
Rank Regularization (SRR) on the hidden conv / linear layers (first and last excluded, like every
other method here).

The paper computes the full spectrum (SVD) of every weight matrix at every step. Here the largest
singular value is approximated by power iteration with warm start, as in SNR / SNVR (N_POWER steps
per call, 15 at the first), which gives the exact gradient direction u v^T of sigma_max:
  SRR  p(W) = |W|_F^2 / sigma_max^2                      (eq. 7; nothing else is needed)
  WAR  p(W) = alpha * log lambda_max = alpha * 2 log sigma_max   (eq. 4)
       alpha = Hill estimator (k = n/2) of the squared singular values, which needs the spectrum:
       it is computed with an SVD under no_grad once per epoch (refresh_alpha, called at the start
       of every epoch) and treated as a constant in the gradient, like the alpha of TempBalance.
       This differs from the paper, where the gradient also flows through alpha.
"""
import torch
import torch.nn as nn


def hidden_weights(model):
    """Weights of the conv / linear layers except the first and the last (the same range as WN /
    SN / SB / TB / SNR here)."""
    mods = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return [m.weight for m in mods[1:-1]]


def hill_alpha(eigs, eps=1e-12):
    """Hill estimator of the tail index of eigs with k = n / 2 (eq. 2 of the paper)."""
    eigs = eigs.sort().values.clamp_min(eps)
    n = eigs.numel()
    if n < 2:
        return eigs.new_tensor(1.0)
    k = max(1, n // 2)
    threshold_idx = n - k - 1
    threshold = eigs[threshold_idx].clamp_min(eps) if threshold_idx >= 0 else eigs[0].clamp_min(eps)
    top_k = eigs[-k:].clamp_min(eps)
    denom = torch.log(top_k / threshold).sum().clamp_min(eps)
    return 1.0 + k / denom


class HeavyTailedRegularizer:
    def __init__(self, model, n_power=5, n_power_init=15):
        self.weights = hidden_weights(model)
        self.n_power, self.n_power_init = n_power, n_power_init
        self.u = {}      # id(weight) -> left singular vector estimate (warm start)
        self.alpha = {}  # id(weight) -> detached Hill alpha (WAR)

    def sigma_max(self, w):
        """Largest singular value of w (as an (out, in*k*k) matrix) by power iteration; the vectors
        are detached, so d sigma / dW = u v^T (the gradient of the true sigma_max)."""
        W = w.reshape(w.size(0), -1)
        with torch.no_grad():
            Wd = W.detach().float()
            u = self.u.get(id(w))
            n = self.n_power
            if u is None:
                u = torch.nn.functional.normalize(torch.randn(Wd.size(0), device=Wd.device, dtype=Wd.dtype), dim=0)
                n = self.n_power_init
            for _ in range(n):
                v = torch.nn.functional.normalize(Wd.t() @ u, dim=0)
                u = torch.nn.functional.normalize(Wd @ v, dim=0)
            self.u[id(w)] = u
            # the true sigma_max is at least |W|_F / sqrt(min(m, n)) (the RMS singular value); an
            # estimate below it (or non-finite) means the iteration broke (observed once as a
            # collapsing loss scale and an exploding stable rank): clamp the value and restart the
            # vector from random next time
            floor = (Wd.pow(2).sum() / min(Wd.shape)).sqrt().clamp_min(1e-12)
        sig = u @ (W.float() @ v)
        if not torch.isfinite(sig) or sig < floor:
            self.u.pop(id(w), None)
            return floor
        return sig

    def stable_rank(self):
        """SRR penalty: sum over the hidden layers of |W|_F^2 / sigma_max^2."""
        return sum(w.float().pow(2).sum() / self.sigma_max(w).pow(2).clamp_min(1e-12) for w in self.weights)

    @torch.no_grad()
    def refresh_alpha(self):
        """Hill alpha of every hidden layer from its full spectrum (once per epoch)."""
        for w in self.weights:
            eigs = torch.linalg.svdvals(w.detach().reshape(w.size(0), -1).float()).pow(2)
            self.alpha[id(w)] = hill_alpha(eigs)

    def weighted_alpha(self):
        """WAR penalty: sum over the hidden layers of alpha * log lambda_max, lambda_max = sigma_max^2."""
        if not self.alpha:
            self.refresh_alpha()
        return sum(self.alpha[id(w)] * 2.0 * torch.log(self.sigma_max(w).clamp_min(1e-12)) for w in self.weights)


# ---- reference implementations with a full SVD (the paper's form), for checks only
def _svd_stable_rank(weights):
    out = 0.0
    for w in weights:
        s = torch.linalg.svdvals(w.reshape(w.size(0), -1))
        out = out + s.pow(2).sum() / s.max().pow(2)
    return out


def _svd_weighted_alpha(weights, detach_alpha=True):
    out = 0.0
    for w in weights:
        eigs = torch.linalg.svdvals(w.reshape(w.size(0), -1)).pow(2)
        a = hill_alpha(eigs)
        if detach_alpha:
            a = a.detach()
        out = out + a * torch.log(eigs.max())
    return out
