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


class SGDSNR(SGD):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, snr=0.0, snvr=0.0, separate=False, drop=True, **kwargs):
        super(SGDSNR, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        self.snr_lambda = snr
        self.snvr_lambda = snvr
        self.separate_snvr_by_type = separate
        self.drop_sn_boundary = drop 

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        sn_list_all = []
        sn_list_linear = []
        sn_list_conv = []
        params_for_sn = []

        # SNR/SNVR 계산을 위한 파라미터 수집
        if self.snr_lambda > 0.0 or self.snvr_lambda > 0.0:
            all_candidates = []

            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None or param.dim() not in {2, 4}:
                        continue
                    all_candidates.append(param)

            # drop 기능 적용
            if self.drop_sn_boundary and len(all_candidates) >= 3:
                # print("Drop SNR boundary")
                # print(all_candidates[0].shape, all_candidates[-1].shape)
                target_params = all_candidates[1:-1]
            else:
                target_params = all_candidates

            for param in target_params:
                for attr in ['snr_applied', 'sn_val', 'uv_val', 'layer_type']:
                    if hasattr(param, attr):
                        delattr(param, attr)

                sn, uv = calc_sn_uv(param.data)
                layer_type = 'linear' if param.dim() == 2 else 'conv'

                param.snr_applied = True
                param.sn_val = sn
                param.uv_val = uv
                param.layer_type = layer_type
                params_for_sn.append(param)

                sn_item = sn.item()
                sn_list_all.append(sn_item)
                if self.separate_snvr_by_type:
                    (sn_list_linear if layer_type == 'linear' else sn_list_conv).append(sn_item)

            sn_mean_linear, sn_mean_conv, sn_mean_all = 0.0, 0.0, 0.0
            snvr_factor_linear, snvr_factor_conv, snvr_factor_all = 0.0, 0.0, 0.0

            if self.separate_snvr_by_type:
                n_linear = len(sn_list_linear)
                if self.snvr_lambda > 0.0 and n_linear > 1:
                    sn_mean_linear = sum(sn_list_linear) / n_linear
                    snvr_factor_linear = self.snvr_lambda * (n_linear - 1) / n_linear

                n_conv = len(sn_list_conv)
                if self.snvr_lambda > 0.0 and n_conv > 1:
                    sn_mean_conv = sum(sn_list_conv) / n_conv
                    snvr_factor_conv = self.snvr_lambda * (n_conv - 1) / n_conv
            else:
                n_layers = len(sn_list_all)
                if self.snvr_lambda > 0.0 and n_layers > 1:
                    sn_mean_all = sum(sn_list_all) / n_layers
                    snvr_factor_all = self.snvr_lambda * (n_layers - 1) / n_layers


        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if maximize:
                    d_p = -d_p

                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if hasattr(param, 'snr_applied') and param.snr_applied:
                    sn = param.sn_val
                    uv = param.uv_val
                    layer_type = param.layer_type

                    if self.snr_lambda > 0.0:
                        snr_grad_base = torch.reshape(torch.mul(sn, uv), param.shape)
                        d_p.add_(snr_grad_base, alpha=self.snr_lambda)

                    current_sn_mean = 0.0
                    current_snvr_factor = 0.0
                    apply_snvr = False

                    if self.separate_snvr_by_type:
                        if layer_type == 'linear' and snvr_factor_linear > 0.0:
                            current_sn_mean = sn_mean_linear
                            current_snvr_factor = snvr_factor_linear
                            apply_snvr = True
                        elif layer_type == 'conv' and snvr_factor_conv > 0.0:
                            current_sn_mean = sn_mean_conv
                            current_snvr_factor = snvr_factor_conv
                            apply_snvr = True
                    else:
                        if snvr_factor_all > 0.0:
                            current_sn_mean = sn_mean_all
                            current_snvr_factor = snvr_factor_all
                            apply_snvr = True

                    if apply_snvr:
                        snvr_term_before_reshape = torch.mul(sn.item() - current_sn_mean, uv)
                        snvr_grad_base = torch.reshape(snvr_term_before_reshape, param.shape)
                        d_p.add_(snvr_grad_base, alpha=current_snvr_factor)

                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-lr)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                 state = self.state[p]
                 state['momentum_buffer'] = momentum_buffer

            for p in group['params']:
                 if hasattr(p, 'snr_applied'): del p.snr_applied
                 if hasattr(p, 'sn_val'): del p.sn_val
                 if hasattr(p, 'uv_val'): del p.uv_val
                 if hasattr(p, 'layer_type'): del p.layer_type

        return loss

class AdamWSNVR(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 snvr=(1e-3, 1.0), decouple=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        snvr=snvr, decouple=decouple)
        super(AdamWSNVR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWSNVR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('snvr', (1e-3, 1.0))
            group.setdefault('decouple', True)
            
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

                if is_snvr:
                    sn = param.sn
                    uv = param.uv

                    snr_grad_base = torch.reshape(torch.mul(sn, uv), param.shape)
                    snvr_grad_base = torch.reshape(torch.mul(sn.item() - sn_mean, uv) * (n_params - 1) / n_params, param.shape)

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
    


        


    
        

