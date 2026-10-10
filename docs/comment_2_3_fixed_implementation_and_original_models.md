# Comment 2.3 Fixed Implementation and Original Model Summary

## What Was Changed

The first simple-CNN implementation produced unusually low results for SB and potentially unfavorable behavior for WAR/SN/WN. After checking the original papers and reference implementation conventions, the following issues were corrected.

| Method | Previous issue | Fix |
|---|---|---|
| WN | Applied only to hidden layers, excluding first and last parameterized layers | Apply WN to all convolutional and linear layers, following the usual `weight_norm(module)` usage |
| SN | Applied only to hidden layers | Apply SN to all convolutional and linear layers, closer to discriminator-wide spectral normalization in SN-GAN |
| SB | Bound coefficient fixed to 1.0 and not tuned | Search `sb_coeff in {1, 2, 5}`; coefficient 1.0 is too restrictive for classification |
| WAR | Implemented as `alpha * lambda_max`, which over-penalizes large spectral norms | Changed to `alpha * log(lambda_max)`, matching the form described in the thesis/literature review as alpha times log squared spectral norm |

## Original Papers: Models and Datasets

| Method | Original paper | Original task | Datasets used | Models/architectures used |
|---|---|---|---|---|
| WN | Salimans & Kingma, 2016 | Supervised classification, generative modeling, RL | CIFAR-10 for supervised classification; MNIST/CIFAR-10 for generative modeling | CIFAR-10 ConvPool-CNN-C style network: 3 blocks of conv layers, max-pooling, dropout, global average pooling; CVAE and DRAW for generative experiments |
| SN | Miyato et al., 2018 | GAN training | CIFAR-10, STL-10, ImageNet/ILSVRC2012 | Standard CNN generator/discriminator and ResNet generator/discriminator for GANs; discriminator uses SN |
| SB | Zhang et al., 2020 | GAN training with strict 1-Lipschitz constraint | GAN benchmark setting; not a supervised classifier method | GAN discriminator/generator setting; method enforces spectral upper bound for Lipschitz control |
| TB | Zhou et al., 2023 | Layer-wise adaptive learning rate / temperature balancing | CIFAR-10, CIFAR-100, SVHN, TinyImageNet | ResNets, VGGs, and WideResNets with various depths and widths |
| SRR/WAR | Xiao et al., 2023 | Supervised classification regularization | KMNIST and CIFAR-10 | FC3 on KMNIST/CIFAR-10; LeNet5 on CIFAR-10; ResNet18 on CIFAR-10 with an added `512 x 128` linear layer |
| SNR | Yoshida & Miyato, 2017 | Supervised classification/generalization | CIFAR-10, CIFAR-100, STL-10 | VGGNet on CIFAR-10; NIN and DenseNet on CIFAR-100; DenseNet on STL-10 |

## Fixed Hyperparameter Search Space

| Method | Search space |
|---|---|
| WD | `weight_decay in {0, 5e-4, 1e-3}` |
| WN | fixed |
| SN | fixed |
| SB | `sb_coeff in {1, 2, 5}` |
| TB | `tb_lr_min_ratio, tb_lr_max_ratio in {(0.5,1.5), (0.25,1.75), (0.75,1.25)}` |
| SRR | `lambda_srr in {1e-5, 1e-4, 1e-3}` |
| WAR | `lambda_war in {1e-5, 5e-5, 1e-4}` |
| SNR | `lambda_snr in {1e-5, 1e-4, 1e-3}` |
| SNVR | `lambda_snr=1e-4`, `lambda_snvr in {1e-4, 1e-3, 1e-2}` |

## Selected Hyperparameters After Fix

| Method | MNIST | CIFAR-10 |
|---|---|---|
| WD | `weight_decay=0` | `weight_decay=5e-4` |
| WN | fixed | fixed |
| SN | fixed | fixed |
| SB | `sb_coeff=5` | `sb_coeff=5` |
| TB | `tb_lr_min_ratio=0.75`, `tb_lr_max_ratio=1.25` | `tb_lr_min_ratio=0.5`, `tb_lr_max_ratio=1.5` |
| SRR | `lambda_srr=1e-5` | `lambda_srr=1e-5` |
| WAR | `lambda_war=1e-5` | `lambda_war=5e-5` |
| SNR | `lambda_snr=1e-4` | `lambda_snr=1e-5` |
| SNVR | `lambda_snr=1e-4`, `lambda_snvr=1e-4` | `lambda_snr=1e-4`, `lambda_snvr=1e-4` |

## Fixed Results

Top-1 accuracy is reported as mean +/- standard deviation over five random seeds.

| Method | MNIST Top-1 (%) | CIFAR-10 Top-1 (%) |
|---|---:|---:|
| w/o WD | 99.23 +/- 0.05 | 75.27 +/- 0.82 |
| WD | 99.25 +/- 0.03 | 75.35 +/- 0.83 |
| WN | 99.20 +/- 0.03 | 74.49 +/- 0.62 |
| SN | 99.20 +/- 0.05 | 70.78 +/- 0.17 |
| SB | 99.16 +/- 0.04 | 68.32 +/- 0.88 |
| TB | 99.19 +/- 0.04 | 74.99 +/- 0.50 |
| SRR | 99.21 +/- 0.05 | 75.28 +/- 0.63 |
| WAR | 99.21 +/- 0.05 | 75.11 +/- 0.86 |
| SNR | 99.21 +/- 0.06 | 75.27 +/- 0.40 |
| SNVR | 99.22 +/- 0.05 | 75.19 +/- 0.64 |

## Analysis of Remaining Low Results

### WN

After applying WN to all convolutional and linear layers, MNIST remains comparable to WD, but CIFAR-10 is still lower. This is likely due to a mismatch with the original WN CIFAR-10 setup. The WN paper used a much wider ConvPool-CNN-C style architecture, ZCA whitening, Gaussian noise, dropout, global average pooling, and Adam training for 200 epochs. Our supplementary experiment intentionally uses a small LeNet-style CNN and SGD for a lightweight controlled comparison. Therefore, WN's weaker CIFAR-10 result should not be interpreted as WN being invalid, but as evidence that the original WN recipe does not transfer automatically to this simplified controlled setting.

### SN

SN becomes lower on CIFAR-10 after applying it to all layers. This is consistent with the method's original purpose: SN was proposed for stabilizing GAN discriminators by controlling the Lipschitz constant. In supervised classification, especially with a small CNN, constraining every layer's spectral norm can reduce representational flexibility. Thus, the remaining low CIFAR-10 result is primarily due to task mismatch: GAN discriminator regularization vs. classifier accuracy optimization.

### SB

The original `sb_coeff=1` setting was too restrictive. Tuning the coefficient to 5 restores MNIST performance from a very low regime to WD-level accuracy. However, CIFAR-10 remains low even with `sb_coeff=5`. This suggests that strict spectral bounding still imposes a strong capacity constraint on the classifier. Since SB is designed to strictly satisfy Lipschitz constraints for GAN training, its weaker supervised classification performance is expected when applied outside its original target setting.

### TB

TB is not a loss regularizer but a layer-wise adaptive learning-rate method. After adding a lightweight implementation based on the original TempBalance idea, TB performs similarly to WD on both MNIST and CIFAR-10. This indicates that TB should not be grouped with the low-performing spectral-constraint baselines in the simple-CNN setting. Its behavior is better interpreted separately as an optimizer/scheduler-style baseline.

### WAR

WAR was previously too low mainly because the penalty scale was implemented too aggressively. After changing the penalty to the alpha-log-spectral-norm form and searching a more appropriate coefficient range, WAR recovers to WD-level performance on both MNIST and CIFAR-10. Therefore, the earlier low WAR result should be attributed to implementation/scale mismatch rather than an inherent failure of WAR in simple CNN settings.

## Suggested Korean Explanation

원 논문 및 공개 구현 관례를 기준으로 WN/SN/SB/WAR 구현을 재점검한 결과, 일부 낮은 결과는 구현 및 hyperparameter 설정의 영향이 있었습니다. WN과 SN은 기존에는 첫 layer와 마지막 classifier layer를 제외한 hidden layer에만 적용했으나, 원 방법론의 일반적인 적용 방식에 맞춰 모든 convolution/linear layer에 적용하도록 수정했습니다. SB는 `coeff=1.0`이 supervised classification에는 지나치게 강한 제약이므로 `coeff in {1, 2, 5}`로 탐색했습니다. WAR는 기존 구현이 `alpha * lambda_max` 형태로 과도하게 큰 penalty를 주고 있었기 때문에, 원 논문에서 사용하는 alpha-log-spectral-norm 계열의 형태에 맞춰 `alpha * log(lambda_max)`로 수정했습니다.

수정 후 MNIST에서는 no-weight-decay baseline과 TB를 포함한 모든 방법이 WD와 유사한 99.2% 수준을 보였습니다. CIFAR-10에서는 TB, SRR, WAR, SNR, SNVR이 WD와 유사한 범위의 성능을 보였습니다. 특히 WAR는 수정 후 WD와 유사하게 회복되어, 이전의 낮은 결과는 penalty 구현 및 scale mismatch의 영향으로 판단됩니다. No-weight-decay baseline 역시 CIFAR-10에서 75.27 +/- 0.82%로 WD와 유사했으며, TB도 74.99 +/- 0.50%로 WD 75.35 +/- 0.83%와 유사하여 simple CNN 환경에서는 낮은 baseline으로 보이지 않습니다.

반면 WN, SN, SB는 CIFAR-10에서 여전히 WD보다 낮았습니다. WN의 원 논문 CIFAR-10 실험은 ConvPool-CNN-C 계열의 더 넓고 깊은 CNN, ZCA whitening, Gaussian noise, dropout, global average pooling, Adam, 200 epoch 학습을 사용했기 때문에, 본 추가 실험의 LeNet-style small CNN + SGD + 10 epoch 설정과 차이가 큽니다. SN과 SB는 본래 GAN discriminator의 Lipschitz control 및 학습 안정화를 위해 제안된 방법이므로, supervised classification에서 모든 layer의 spectral scale을 강하게 제한하면 표현력이 줄어들 수 있습니다. 따라서 수정 후에도 SN/SB가 낮은 것은 구현 오류라기보다는 원 방법론의 목적과 본 classification task 사이의 mismatch로 해석하는 것이 타당합니다.
