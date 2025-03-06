import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
import numpy as np

from utils import calc_sn_uv

class AdamWSNR(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.01, snr=0.0, snvr=0.0):
        super(AdamWSNR, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.snr = snr
        self.snvr = snvr
        
    def step(self):
        if self.snr:
            sn_list = []
            for group in self.param_groups:
                for param in group['params']:
                    param.snr = False
                    if param.grad is None:
                        continue
                    if param.dim() < 2:
                        continue
                    if param.size(0) == 1 or param.view(param.size(0), -1).size(1) == 1:
                        continue
                    param.snr = True
                    param.sn, param.uv = calc_sn_uv(param.data)
                    sn_list.append(param.sn)
            if self.snvr:
                n_layers = len(sn_list)
                sn_mean = sum(sn[0] for sn in sn_list) / n_layers
                len_coeff = (n_layers - 1) / n_layers


        for group in self.param_groups:
            params: list[torch.Tensor] = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            has_complex = self._init_group(
                group,
                params,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            for i, param in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                # update step
                step_t += 1

                # Perform step weight decay
                param.data.mul_(1 - lr * weight_decay)

                if self.snr and param.snr:
                    sn, uv = param.sn, param.uv
                    snr = torch.reshape(torch.mul(sn, uv), param.shape)
                    # add snr * self.snr to grad
                    grad.data.add_(torch.mul(snr, self.snr))

                    if self.snvr != 0:
                        snvr = torch.reshape(torch.mul(torch.mul(sn - sn_mean, len_coeff), uv), param.shape)
                        # add snvr * self.snvr to grad
                        grad.data.add_(torch.mul(snvr, self.snvr))

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2.sqrt() if isinstance(bias_correction2, torch.Tensor) else np.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.data.addcdiv_(exp_avg, denom, value=-step_size)

class AdamSNR(Adam):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.0, snr=0.0, snvr=0.0):
        super(AdamSNR, self).__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.snr = snr
        self.snvr = snvr
        
    def step(self):
        if self.snr:
            sn_list = []
            for group in self.param_groups:
                for param in group['params']:
                    param.snr = False
                    if param.grad is None:
                        continue
                    if param.dim() < 2:
                        continue
                    if param.size(0) == 1 or param.view(param.size(0), -1).size(1) == 1:
                        continue
                    param.snr = True
                    param.sn, param.uv = calc_sn_uv(param.data)
                    sn_list.append(param.sn)
            if self.snvr:
                n_layers = len(sn_list)
                sn_mean = sum(sn[0] for sn in sn_list) / n_layers
                len_coeff = (n_layers - 1) / n_layers


        for group in self.param_groups:
            params: list[torch.Tensor] = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            has_complex = self._init_group(
                group,
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)


            for i, param in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                # update step
                step_t += 1

                # Perform step weight decay
                if weight_decay:
                    grad = grad.add(param.data, alpha=weight_decay)

                if self.snr and param.snr:
                    sn, uv = param.sn, param.uv
                    snr = torch.reshape(torch.mul(sn, uv), param.shape)
                    # add snr * self.snr to grad
                    grad.data.add_(torch.mul(snr, self.snr))

                    if self.snvr != 0:
                        snvr = torch.reshape(torch.mul(torch.mul(sn - sn_mean, len_coeff), uv), param.shape)
                        # add snvr * self.snvr to grad
                        grad.data.add_(torch.mul(snvr, self.snvr))

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2.sqrt() if isinstance(bias_correction2, torch.Tensor) else np.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.data.addcdiv_(exp_avg, denom, value=-step_size)

if __name__ == '__main__':
    from torch.nn.utils.parametrizations import spectral_norm, _SpectralNorm
    m = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m = spectral_norm(m)
    print(m._u)
    print(m._v)