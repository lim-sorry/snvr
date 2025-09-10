import torch
import torch.nn as nn

from torchvision.models import resnet18

class LWSAM(torch.optim.Optimizer):
    """
    Layer-wise SAM (LWSAM) optimizer with an integrated step function.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        base_optimizer (torch.optim.Optimizer): base optimizer.
        rho (float, optional): neighborhood size. (default: 0.05)
        k (int, optional): the number of layers to perturb. (default: 4)
        adaptive (bool, optional): whether to use adaptive SAM. (default: False)
        **kwargs: keyword arguments passed to the base optimizer.
    """
    def __init__(self, params, base_optimizer, rho=0.05, k=4, adaptive=False, **kwargs):
        if not rho >= 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if not k >= 1:
            raise ValueError(f"Invalid k, should be at least 1: {k}")

        defaults = dict(rho=rho, k=k, adaptive=adaptive, **kwargs)
        super(LWSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        for group in self.param_groups:
            group["layer_grad_norm"] = 0.0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norms = [group["layer_grad_norm"] for group in self.param_groups]
        
        num_groups = len(grad_norms)
        k_val = min(self.defaults['k'], num_groups)

        if k_val > 0:
            _, topk_indices = torch.topk(torch.tensor(grad_norms), k_val)
            topk_indices = topk_indices.tolist()
        else:
            topk_indices = []

        for i, group in enumerate(self.param_groups):
            if i in topk_indices:
                scale = group["rho"] / (self._grad_norm(group) + 1e-12)
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for i, group in enumerate(self.param_groups):
            group["layer_grad_norm"] = self._grad_norm(group, for_storage=True)
            for p in group["params"]:
                if p.grad is None: continue
                if p in self.state and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
                    self.state[p].pop("old_p")
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step that includes the SAM logic.

        Args:
            closure (callable): A closure that re-evaluates the model,
                                calculates the loss, and calls loss.backward().
                                This closure is used for the second SAM step.
        """
        self.first_step(zero_grad=True)
        
        with torch.enable_grad():
            closure()
            
        self.second_step()
        self.base_optimizer.step()


    def _grad_norm(self, group, for_storage=False):
        shared_device = group["params"][0].device
        
        params_with_grad = [p for p in group['params'] if p.grad is not None]
        if not params_with_grad:
            return 0.0 if for_storage else torch.tensor(0.0, device=shared_device)
            
        grad_concat = torch.cat([p.grad.view(-1) for p in params_with_grad])
        norm = torch.norm(grad_concat, p=2)
        
        return norm.item() if for_storage else norm.to(shared_device)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def group_parameters_by_layer_type(model: nn.Module, trigger_layers=(nn.Conv1d, nn.Conv2d, nn.Linear)):
    param_groups = []
    current_group = []

    for module in model.modules():
        if list(module.parameters(recurse=False)):
            if isinstance(module, trigger_layers):
                if current_group:
                    param_groups.append({'params': current_group})
                current_group = list(module.parameters(recurse=False))
            else:
                current_group.extend(list(module.parameters(recurse=False)))

    if current_group:
        param_groups.append({'params': current_group})
        
    print(f"Model has been divided into {len(param_groups)} parameter groups.")
    return param_groups

if __name__ == "__main__":

    inputs = torch.randn(10, 3, 32, 32)
    labels = torch.randn(10, 1)

    model = resnet18(num_classes=1)
    parameters = group_parameters_by_layer_type(model)
    optimizer = LWSAM(parameters, torch.optim.SGD, rho=0.05, k=4, lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, labels)
    loss.backward()

    def closure():
        predictions_perturbed = model(inputs)
        loss_perturbed = criterion(predictions_perturbed, labels)
        loss_perturbed.backward()
        return loss_perturbed

    optimizer.step(closure)
