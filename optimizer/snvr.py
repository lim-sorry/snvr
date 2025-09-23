import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Optimizer

@torch.no_grad()
def calc_sn_uv(weight, n_power=5, eps=1e-12):
    weight_data = weight.data
    device = weight_data.device

    if weight_data.dim() < 2:
        raise ValueError("Weight tensor must be at least 2D for spectral norm calculation.")

    weight_mat = weight_data.view(weight_data.size(0), -1) if weight_data.dim() > 2 else weight_data
    height, width = weight_mat.shape

    u = F.normalize(torch.randn(1, height, device=device), eps=eps)

    for _ in range(n_power):
        v = F.normalize(torch.matmul(u, weight_mat), eps=eps, dim=1)
        u = F.normalize(torch.matmul(v, weight_mat.t()), eps=eps, dim=1)

    sn = torch.matmul(torch.matmul(u, weight_mat), v.t()).squeeze()
    uv = torch.matmul(u.t(), v)
    return sn, uv


class SGDSNVR(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, dampening=0.0, weight_decay=0.0005,
                 nesterov=False, snvr=(0.0, 0.0), decouple=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        snvr=snvr, decouple=decouple)
        super(SGDSNVR, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            snvr = group['snvr']
            decouple = group['decouple']
            lr = group['lr']

            params_with_grad = []
            grads = []
            momentum_buffers = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if momentum > 0 and 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if momentum > 0:
                        momentum_buffers.append(state['momentum_buffer'])
                    else:
                        momentum_buffers.append(None)

            # Pre compute spectral norms and uv
            is_snvr = snvr[0] > 0.0 or snvr[1] > 0.0
            if is_snvr:
                sn_mean = 0.0
                n_params = len(params_with_grad)
                for p in params_with_grad:
                    sn, uv = calc_sn_uv(p.data)
                    setattr(p, 'sn', sn)
                    setattr(p, 'uv', uv)
                    sn_mean += sn.item()
                sn_mean /= n_params
            else:
                for p in params_with_grad:
                    if hasattr(p, 'sn'):
                        del p.sn
                    if hasattr(p, 'uv'):
                        del p.uv

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                buf = momentum_buffers[i]

                if weight_decay != 0:
                    if decouple:
                        param.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)

                # Add SNVR gradients
                if is_snvr:
                    sn = param.sn
                    uv = param.uv
                    snr_grad_base = torch.reshape(sn * uv, param.shape)
                    snvr_grad_base = torch.reshape((sn.item() - sn_mean) * uv, param.shape)

                    if decouple:
                        param.add_(snr_grad_base, alpha=(-lr * snvr[0]))
                        param.add_(snvr_grad_base, alpha=(-lr * snvr[1]))
                    else:
                        grad.add_(snr_grad_base, alpha=snvr[0])
                        grad.add_(snvr_grad_base, alpha=snvr[1])

                if momentum > 0:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf
                else:
                    update = grad

                param.add_(update, alpha=-lr)

        return loss

class AdamWSNVR(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 snvr=(0.0, 0.0), decouple=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        snvr=snvr, decouple=decouple)
        super(AdamWSNVR, self).__init__(params, defaults)
            
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            snvr = group['snvr']
            decouple = group['decouple']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            # Pre compute spectral norms and uv
            is_snvr = snvr[0] > 0.0 or snvr[1] > 0.0
            if is_snvr:
                sn_mean = 0.0
                n_params = len(params_with_grad)
                for p in params_with_grad:
                    sn, uv = calc_sn_uv(p.data)
                    setattr(p, 'sn', sn)
                    setattr(p, 'uv', uv)
                    sn_mean += sn.item()
                sn_mean /= n_params
            else:
                for p in params_with_grad:
                    if hasattr(p, 'sn'):
                        del p.sn
                    if hasattr(p, 'uv'):
                        del p.uv

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                if amsgrad:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)

                # Add SNVR gradients
                if is_snvr:
                    sn = param.sn
                    uv = param.uv

                    snr_grad_base = torch.reshape(torch.mul(sn, uv), param.shape)
                    snvr_grad_base = torch.reshape((sn.item() - sn_mean) * uv, param.shape)

                    if decouple:
                        param.add_(snr_grad_base, alpha=(-lr * snvr[0]))
                        param.add_(snvr_grad_base, alpha=(-lr * snvr[1]))
                    else:
                        grad.add_(snr_grad_base, alpha=snvr[0])
                        grad.add_(snvr_grad_base, alpha=snvr[1])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2 ** step_t.item())).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2 ** step_t.item())).add_(eps)

                step_size = lr / (1.0 - beta1 ** step_t.item())
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    


        


    
        

