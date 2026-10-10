# Comment 2.3 Hyperparameter Search Addendum

## Search Protocol

A lightweight hyperparameter search was conducted for the simple CNN experiments on MNIST and CIFAR-10. For each dataset, 10% of the training set was used as a validation split. Candidate hyperparameters were evaluated for 5 epochs with seed 0, and the best validation setting for each method was then evaluated on the test set for 10 epochs over five random seeds.

The simple CNN architecture was kept fixed across methods:

- MNIST: `Conv(1,32,5) -> ReLU -> MaxPool -> Conv(32,64,5) -> ReLU -> MaxPool -> FC(3136,128) -> ReLU -> FC(128,10)`
- CIFAR-10: same architecture with 3 input channels and `FC(4096,128)` after flattening

Batch normalization and dropout were intentionally excluded so that the effect of each regularization method could be isolated.

## Hyperparameter Search Space

| Method | Search space | Notes |
|---|---|---|
| WD | `weight_decay in {0, 5e-4, 1e-3}` | Standard L2/weight decay baseline |
| WN | fixed | No regularization-strength hyperparameter in the compared implementation |
| SN | fixed | Spectral normalization with one power iteration |
| SB | fixed | Spectral bound coefficient fixed to 1.0 |
| SRR | `lambda_srr in {1e-5, 1e-4, 1e-3}` | Stable-rank regularization strength |
| WAR | `lambda_war in {1e-7, 1e-6, 1e-5}` | Weight-alpha regularization strength |
| SNR | `lambda_snr in {1e-5, 1e-4, 1e-3}` | Spectral-norm regularization strength |
| SNVR | `lambda_snr=1e-4`, `lambda_snvr in {1e-4, 1e-3, 1e-2}` | SNVR strength searched with SNR fixed |

## Selected Hyperparameters

| Method | MNIST selected setting | CIFAR-10 selected setting |
|---|---|---|
| WD | `weight_decay=0` | `weight_decay=0` |
| WN | fixed | fixed |
| SN | fixed | fixed |
| SB | fixed | fixed |
| SRR | `lambda_srr=1e-3` | `lambda_srr=1e-4` |
| WAR | `lambda_war=1e-5` | `lambda_war=1e-6` |
| SNR | `lambda_snr=1e-5` | `lambda_snr=1e-5` |
| SNVR | `lambda_snr=1e-4`, `lambda_snvr=1e-2` | `lambda_snr=1e-4`, `lambda_snvr=1e-4` |

## Original Dataset/Task Support in Referenced Methods

| Method | Original paper/task context | MNIST in original paper? | CIFAR-10 in original paper? | Note for this thesis response |
|---|---|---:|---:|---|
| WD | General baseline regularization | N/A | N/A | Used as a standard baseline rather than a dataset-specific method |
| WN | Supervised classification and generative modeling | Yes, but mainly generative modeling on MNIST | Yes, supervised CIFAR-10 and generative CIFAR-10 | Supports the claim that WN was validated in simpler image settings |
| SN | GAN discriminator normalization | No confirmed MNIST experiment in the SN-GAN paper | Yes, CIFAR-10 GAN experiments | Original purpose is GAN stability, not supervised classification |
| SB | Strict 1-Lipschitz constraint for GANs | Not treated as supervised MNIST classification support | Not treated as supervised CIFAR-10 classification support | Original purpose is GAN Lipschitz control; classification use is outside its main target |
| SRR/WAR | Heavy-tailed regularization for classifiers | KMNIST, not original MNIST | Yes, CIFAR-10 with FC3, LeNet5, and ResNet18 | Closest baseline support for shallow models and CIFAR-10 |
| SNR | Spectral norm regularization for generalization | No confirmed MNIST experiment | Yes, CIFAR-10 with VGGNet | Classification-oriented, but evaluated on older/smaller model settings |
| SNVR | Proposed method | Additional experiment in this response | Additional experiment in this response | Included for comparison under the same simple-CNN setting |

## Tuned Test Results

Top-1 accuracy is reported as mean +/- standard deviation over five random seeds.

| Method | MNIST Top-1 (%) | CIFAR-10 Top-1 (%) |
|---|---:|---:|
| WD | 99.23 +/- 0.05 | 75.19 +/- 0.47 |
| WN | 99.22 +/- 0.03 | 74.81 +/- 0.42 |
| SN | 99.27 +/- 0.02 | 74.34 +/- 0.38 |
| SB | 95.16 +/- 0.44 | 42.72 +/- 1.05 |
| SRR | 99.18 +/- 0.04 | 75.20 +/- 0.89 |
| WAR | 99.20 +/- 0.05 | 75.01 +/- 1.01 |
| SNR | 99.22 +/- 0.05 | 75.24 +/- 0.46 |
| SNVR | 99.21 +/- 0.04 | 75.29 +/- 0.49 |

## Interpretation

The hyperparameter search does not change the main conclusion. On MNIST, WN, SN, SRR, WAR, SNR, and SNVR all reach a saturated accuracy range near WD, while SB remains lower. On CIFAR-10, WN, SRR, WAR, SNR, and SNVR remain close to WD after tuning, while SN is lower and SB is substantially worse.

This supports the response that the low performance of several baselines in the main ResNet/Conv2NeXt experiments is unlikely to be due to an implementation failure or a lack of tuning alone. Rather, the regularization effects observed in simple CNNs on MNIST/CIFAR-10 do not necessarily transfer to modern architectures and more complex multi-class classification datasets. In particular, SN and SB were proposed for GAN training stability and Lipschitz control, so their weaker performance in supervised classification is consistent with their original design goal.

## Suggested Korean Summary

간단한 hyperparameter search를 수행한 후에도 결론은 유지됩니다. MNIST에서는 WN, SN, SRR, WAR, SNR, SNVR이 모두 WD와 유사한 99.2% 수준의 성능을 보였고, CIFAR-10에서도 WN, SRR, WAR, SNR, SNVR은 WD와 유사한 범위에 머물렀습니다. 반면 SB는 MNIST와 CIFAR-10 모두에서 낮은 성능을 보였으며, 특히 CIFAR-10에서는 42.72%로 크게 하락했습니다. 이는 strict Lipschitz constraint가 GAN 학습 안정화에는 적합하지만 supervised classification에서는 표현력을 과도하게 제한할 수 있음을 보여줍니다.

따라서 main experiment에서 일부 baseline이 WD보다 낮은 성능을 보인 것은 단순한 hyperparameter tuning 부족이나 구현 오류라기보다는, 해당 방법들이 검증된 단순 CNN/MNIST-CIFAR10 수준의 환경과 ResNet/Conv2NeXt 및 CIFAR-100, TinyImageNet, STL-10, Food-101과 같은 복잡한 classification 환경 사이의 차이에서 비롯된 것으로 해석할 수 있습니다.
