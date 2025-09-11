import torch
import torch.nn as nn

def weight_alpha(model):
    alphas = []
    for m in model.modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        weight = m.weight
        # if isinstance(m, nn.Conv2d):
        #     weight = weight.flatten(start_dim=2)
        #     weight = weight.transpose(1,2).transpose(0,1)
        if isinstance(m, nn.Conv2d):
            weight = weight.flatten(start_dim=1)
        _U, S, _V = torch.pca_lowrank(weight, q=min(weight.shape[-2:]), center=False, niter=2)
        eigs = torch.sort(S.pow(2).flatten())[0]
        max_eig = eigs[-1]
        n = len(eigs)
        k = n // 2

        top_k_eigs = eigs[-k:]
        x_nk = eigs[n - k - 1]

        log_sum = torch.sum(torch.log(top_k_eigs)) - k * torch.log(x_nk)
        alpha = 1.0 + k / (log_sum + 1e-6)
        alphas.append(alpha * max_eig)

    return sum(alphas)

def stable_rank(model):
    ranks = []
    for m in model.modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        weight = m.weight
        # if isinstance(m, nn.Conv2d):
        #     weight = weight.flatten(start_dim=2)
        #     weight = weight.transpose(1,2).transpose(0,1)
        if isinstance(m, nn.Conv2d):
            weight = weight.flatten(start_dim=1)
        _U, S, _V = torch.pca_lowrank(weight, q=min(weight.shape[-2:]), center=False, niter=2)
        rank = (S**2).sum() / (S.max()**2 + 1e-6)
        ranks.append(rank)
    return sum(ranks)

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