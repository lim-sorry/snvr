# Spectral Norm Variance Regularization

Jiseong Lim [[Github](https://github.com/lim-sorry)]

## Introduction
In deep learning, we aim to study generalizability based on the variance in sensitivity across layers of a neural network. 
**Spectral Norm Regularization**, which imposes a penalty on the spectral norms of the weight matrices, is known to improve generalization performance by reducing sensitivity to input perturbations.
However, consistently imposing a uniform penalty across all layers may lead to an imbalance, excessively decreasing the sensitivity of certain layers. 
<br/>
<br/>
To address this, we propose **Spectral Norm Variance Regularization**, which penalizes the variance in spectral norms across layers in order to alleviate sensitivity imbalances. 
The goal of this study is to experimentally demonstrate that adding our proposed method to existing approaches helps reduce layer-wise sensitivity imbalances and results in better generalization performance.

## Methodology
We first define a simple feedforward neural network setting. Consider a feedforward network that maps input vectors through a sequence of layers as follows:

$$
x^{\ell} = f_{\ell}(W^{\ell} x^{\ell-1} + b^{\ell}), \quad \ell = 1, \dots, L
$$

Here, $x^{\ell} \in \mathbb{R}^{n_\ell}$ represents the output of the $\ell$-th layer, and $f_{\ell}: \mathbb{R}^{n_\ell} \to \mathbb{R}^{n_\ell}$ is the layer’s activation function. Each layer is defined by the weight matrix $W^{\ell} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ and the bias vector $b^{\ell} \in \mathbb{R}^{n_\ell}$. Collecting all parameters into $\Theta = \\{W^{\ell}, b^{\ell}\\}\_{\ell=1}^L$, the network function $f_\theta : \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$ is given by $f_\theta(x^{0}) = x^{L}$.
<br/>
<br/>

**Spectral Norm Regularization (SNR)** penalizes the spectral norm of each layer’s weight matrix $W^{\ell}$, thereby helping reduce the overall sensitivity of the network to input perturbations.

Given training data $\\{x_i, y_i\\}_{i=1}^K$, here is the SNR-adapted objective function:

$$
\min_\theta \frac{1}{K}\sum_{i=1}^K L_{ce}(f_\theta(x_i), y_i) + \frac{\lambda_{SNR}}{2} \sum_{\ell=1}^L (\sigma(W^{\ell}))^2
$$

where $L_{ce}$ is the cross-entropy loss commonly chosen for classification tasks, and $\sigma(W^{\ell})$ is the spectral norm of the weight matrix at layer $\ell$. The hyperparameter $\lambda_{SNR}$ controls the strength of SNR.
<br/>
<br/>

To further improve generalization and address imbalances in layer-wise sensitivities, we introduce **Spectral Norm Variance Regularization (SNVR)**. While SNR regulates individual layers, SNVR additionally penalizes the variance of the spectral norms across layers. This encourages a more balanced distribution of sensitivities, preventing any single layer from becoming disproportionately suppressed.

Combining these components, we define the following objective function:

$$
\min_\theta \frac{1}{K}\sum_{i=1}^K L_{ce}(f_\theta(x_i), y_i) + \frac{\lambda_{SNR}}{2} \sum_{\ell=1}^L (\sigma(W^{\ell}))^2 + \frac{\lambda_{SNVR}}{2} \sum_{\ell=1}^L (\sigma(W^{\ell}) - \mu_\sigma)^2
$$

where $\mu_\sigma$ is the average spectral norm across all layers. The hyperparameter $\lambda_{SNVR}$ controls the strength of SNVR.

By incorporating SNVR on top of SNR, we aim to achieve more balanced layer-wise sensitivities and ultimately enhance the network’s generalization performance.

## Components
- **sgdsnr.py** - script for SNVR-adapted optimizer based on SGD.
- **main.py** - script for main experiments.
- **utils.py** - script for utilities, containing train, test, load and saving
- **config.py** - script for configurations.

## Usage
```python
import torch
from sgdsnr import SGDSNR
model = ...
# initialize SGD-based SGDSNR optimizer
optimizer = SGDSNR(
    model.parameters(),
    ...
    spectrum_regularization=0.1,
    spectrum_variance_regularization=0.1,
    )
# training loop
for epoch in range(1, ...):
    train()
    test()
    ...
```

## Experiments 
```bash
# Set up absolute paths for code repository, and checkpoints.
export src_path=...
export ckpt_src_path=...
export data_path=...

# run test.sh
bash bash_scripts/test.sh
```

## Dependency
- torch==2.2.0
- torchvision==0.17.0
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

## References

- Zhou, Y., Pang, T., Liu, K., Martin, C. H., Mahoney, M. W., & Yang, Y. (2023).  
  *Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training*.  
  [arXiv:2312.00359](https://arxiv.org/abs/2312.00359)

- Yoshida, Y., & Miyato, T. (2017).  
  *Spectral Norm Regularization for Improving the Generalizability of Deep Learning*.  
  [arXiv:1705.10941](https://arxiv.org/abs/1705.10941)
