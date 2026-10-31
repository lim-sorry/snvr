# Hyperparameters

Selected hyperparameters of every method, network architecture and dataset; the configs in `config/final/<model>/<dataset>/<method>.yaml` contain exactly these values. Methods: WD (weight decay only, the baseline), WN (weight normalization), SN (spectral normalization), SB (spectral bound), TB (TempBalance), WAR / SRR (weighted alpha / stable rank regularization), SNR (spectral norm regularization) and SNVR (ours). Every method uses the learning rate and weight decay selected for WD, except SB, which has its own learning rate. SNVR uses the λ_SNR selected for SNR.

Each result in the paper is the mean ± std. dev. over seeds 0, 1 and 2 (`--seed`).

## ResNet-18

| Method | Hyperparameter | CIFAR-10 | CIFAR-100 | TinyImageNet | STL-10 | Food-101 | EuroSAT |
|---|---|---|---|---|---|---|---|
| All (shared) | learning rate η | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
|  | weight decay | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 5e-4 |
| WN | – | – | – | – | – | – | – |
| SN | – | – | – | – | – | – | – |
| SB | bound coefficient c | 0.5 | 1 | 2 | 0.5 | 1 | 2 |
|  | learning rate η_SB | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| TB | LR scaling range (s1, s2) | (0.5, 1.5) | (0.5, 1.5) | (0.25, 1.75) | (0.75, 1.25) | (0.25, 1.75) | (0.25, 1.75) |
| WAR | λ_WAR | 1e-5 | 1e-5 | 1e-3 | 1e-4 | 1e-3 | 1e-6 |
| SRR | λ_SRR | 1e-5 | 1e-5 | 1e-4 | 0.01 | 1e-4 | 1e-3 |
| SNR | λ_SNR | 0.01 | 0.01 | 1e-3 | 0.01 | 1e-3 | 0.1 |
| SNVR | λ_SNVR | 0.01 | 0.01 | 1 | 0.01 | 1e-3 | 1e-3 |

## ResNet-34

| Method | Hyperparameter | CIFAR-10 | CIFAR-100 | TinyImageNet | STL-10 | Food-101 | EuroSAT |
|---|---|---|---|---|---|---|---|
| All (shared) | learning rate η | 0.1 | 0.1 | 0.1 | 0.01 | 0.1 | 0.01 |
|  | weight decay | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 5e-4 |
| WN | – | – | – | – | – | – | – |
| SN | – | – | – | – | – | – | – |
| SB | bound coefficient c | 0.5 | 1 | 1 | 0.1 | 0.5 | 1 |
|  | learning rate η_SB | 1e-3 | 1e-3 | 1e-3 | 1e-4 | 1e-3 | 1e-4 |
| TB | LR scaling range (s1, s2) | (0.75, 1.25) | (0.5, 1.5) | (0.75, 1.25) | (0.75, 1.25) | (0.5, 1.5) | (0.75, 1.25) |
| WAR | λ_WAR | 1e-5 | 1e-5 | 1e-4 | 1e-5 | 1e-3 | 1e-7 |
| SRR | λ_SRR | 1e-7 | 1e-4 | 1e-7 | 1e-3 | 1e-3 | 1e-3 |
| SNR | λ_SNR | 1e-3 | 0.01 | 1e-3 | 1 | 0.01 | 1 |
| SNVR | λ_SNVR | 0.1 | 0.1 | 10 | 1 | 1e-3 | 1 |

## Conv2NeXt-T

| Method | Hyperparameter | CIFAR-10 | CIFAR-100 | TinyImageNet | STL-10 | Food-101 | EuroSAT |
|---|---|---|---|---|---|---|---|
| All (shared) | learning rate η | 0.01 | 0.01 | 1e-3 | 1e-3 | 0.01 | 0.01 |
|  | weight decay | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| WN | – | – | – | – | – | – | – |
| SN | – | – | – | – | – | – | – |
| SB | bound coefficient c | 5 | 5 | 5 | 0.5 | 5 | 5 |
|  | learning rate η_SB | 1e-3 | 1e-3 | 1e-4 | 1e-4 | 1e-3 | 1e-3 |
| TB | LR scaling range (s1, s2) | (0.75, 1.25) | (0.5, 1.5) | (0.75, 1.25) | (0.75, 1.25) | (0.75, 1.25) | (0.5, 1.5) |
| WAR | λ_WAR | 1e-7 | 1e-7 | 1e-6 | 1e-7 | 1e-4 | 1e-5 |
| SRR | λ_SRR | 1e-7 | 1e-6 | 1e-4 | 1e-5 | 1e-3 | 1e-4 |
| SNR | λ_SNR | 0.01 | 0.1 | 1e-3 | 1e-3 | 1 | 1 |
| SNVR | λ_SNVR | 10 | 1e-4 | 10 | 1 | 10 | 10 |

## ViT-Tiny

| Method | Hyperparameter | CIFAR-10 | CIFAR-100 | TinyImageNet | STL-10 | Food-101 | EuroSAT |
|---|---|---|---|---|---|---|---|
| All (shared) | learning rate η | 1e-3 | 1e-3 | 1e-3 | 1e-4 | 1e-4 | 1e-3 |
|  | weight decay | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| WN | – | – | – | – | – | – | – |
| SN | – | – | – | – | – | – | – |
| SB | bound coefficient c | 5 | 5 | 5 | 5 | 5 | 5 |
|  | learning rate η_SB | 1e-4 | 1e-4 | 1e-4 | 1e-5 | 1e-5 | 1e-4 |
| TB | LR scaling range (s1, s2) | (0.75, 1.25) | (0.75, 1.25) | (0.75, 1.25) | (0.75, 1.25) | (0.75, 1.25) | (0.5, 1.5) |
| WAR | λ_WAR | 1e-5 | 1e-5 | 1e-6 | 1e-5 | 1e-6 | 1e-4 |
| SRR | λ_SRR | 1e-5 | 1e-6 | 1e-4 | 1e-4 | 1e-4 | 1e-5 |
| SNR | λ_SNR | 10 | 10 | 10 | 10 | 10 | 10 |
| SNVR | λ_SNVR | 0.01 | 100 | 0.1 | 10 | 1 | 0.1 |
