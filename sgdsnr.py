import torch
from torch.optim.optimizer import Optimizer, required

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_weight(weight, do_power_iteration=True, n_power_iterations=5):
    weight_mat = weight

    height = weight_mat.size(0)
    weight_mat = weight_mat.view(height, -1)

    u = torch.randn(1, weight_mat.shape[0]).to(device)  # move u to GPU
    for i in range(n_power_iterations):
        v = torch.mm(u, weight_mat)  # [1, n]
        v = v / torch.norm(v)
        u = torch.mm(v, weight_mat.T)  # [1, m]
        u = u / torch.norm(u)

    sn = (torch.mul(torch.ones(1, weight_mat.shape[0]).to(device), torch.sum(torch.mm(torch.mm(u, weight_mat), v.T)))).T
    uv = torch.mm(u.T, v)

    return sn, uv

class SGDSNR(Optimizer):
    r"""Implements stochastic gradient descent with spectrum regularization.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        snr (float, optional): spectrum regularization factor (default: 0)
        snvr (float, optional): spectrum variance regularization factor (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """
    def __init__(self, params, lr=required, 
                    momentum=0, dampening=0,
                    weight_decay=0, nesterov=False, 
                    snr=0, 
                    snvr=0,
                    differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        spectrum_regularization=snr,
                        spectrum_variance_regularization=snvr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDSNR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDSNR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('spectrum_regularization', 0)
            group.setdefault('spectrum_variance_regularization', 0)
            group.setdefault('differentiable', False)
            group.setdefault('dampening', 0)            

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            spectrum_regularization = group['spectrum_regularization']
            spectrum_variance_regularization = group['spectrum_variance_regularization']

            if spectrum_regularization != 0:
                sn_dict, uv_dict = {}, {}
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    if p.dim() > 1:
                        sn_dict[i], uv_dict[i] = compute_weight(p.to(device))
                    else:
                        pass
                if spectrum_variance_regularization != 0:
                    n_layers = len(sn_dict.keys())
                    sn_mean = sum(sn[0] for sn in sn_dict.values()) / n_layers
                    len_coeff = (n_layers - 1) / n_layers            

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if spectrum_regularization != 0:
                    #print("spectrum_regularization", spectrum_regularization, p.dim())
                    if p.dim() > 1:
                        sn, uv = sn_dict[i], uv_dict[i]
                        d_p = d_p.add(torch.reshape(torch.mul(sn, uv), p.shape),
                                        alpha=spectrum_regularization)
                    else:
                        pass
                else:
                    pass

                if spectrum_variance_regularization != 0:
                    # print("spectrum_var_regularization", spectrum_var_regularization, p.dim(), p.shape)
                    if p.dim() > 1:
                        sn, uv = sn_dict[i], uv_dict[i]
                        d_p = d_p.add(torch.reshape(torch.mul(torch.mul(sn - sn_mean, len_coeff), uv), p.shape),
                                        alpha=spectrum_variance_regularization)
                    else:
                        pass
                else:
                    pass

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
