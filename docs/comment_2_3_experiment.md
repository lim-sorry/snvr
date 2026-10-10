# Comment 2.3 Additional Experiment

## Experiment Design

To address Comment 2.3, we add supplementary experiments under settings where several baseline regularizers were originally proposed or commonly validated: shallow CNNs on MNIST and CIFAR-10. The purpose is to separate two possibilities: whether the relatively low performance of WN, SN, SB, SRR, and WAR in the main experiments is due to implementation error, or whether these methods are less suitable as general-purpose regularizers for modern multi-class image classification settings.

The experiment uses a simple CNN with two convolutional layers and two fully connected layers. For MNIST, the model is `Conv(1,32,5) -> MaxPool -> Conv(32,64,5) -> MaxPool -> FC(3136,128) -> FC(128,10)`. For CIFAR-10, the same architecture is used with a 3-channel input and the corresponding 32x32 flatten size.

All methods are trained for 10 epochs with SGD, momentum 0.9, weight decay 5e-4, batch size 128, and five random seeds. The compared methods are WD, WN, SN, SB, SRR, WAR, SNR, and SNVR. The implementation is available at `experiments/mnist_simple_cnn_comment_2_3.py`.

## How To Reproduce

```bash
python experiments/mnist_simple_cnn_comment_2_3.py --dataset mnist --data-path ../data --output output_comment_2_3/mnist_simple_cnn
python experiments/mnist_simple_cnn_comment_2_3.py --dataset cifar10 --data-path ../data --output output_comment_2_3/cifar10_simple_cnn
```

The script writes raw results, summary statistics, and markdown tables to each output directory. The combined table is saved at `output_comment_2_3/combined_simple_cnn/table.md`.

## Results

Top-1 accuracy is reported as mean +/- standard deviation over five random seeds.

| Method | MNIST Top-1 (%) | CIFAR-10 Top-1 (%) |
|---|---:|---:|
| WD | 99.21 +/- 0.04 | 75.32 +/- 0.49 |
| WN | 99.20 +/- 0.05 | 75.04 +/- 0.57 |
| SN | 99.28 +/- 0.06 | 74.00 +/- 0.40 |
| SB | 95.58 +/- 0.72 | 42.64 +/- 0.94 |
| SRR | 99.19 +/- 0.02 | 75.23 +/- 0.83 |
| WAR | 99.21 +/- 0.05 | 75.36 +/- 0.59 |
| SNR | 99.22 +/- 0.04 | 75.30 +/- 0.92 |
| SNVR | 99.20 +/- 0.05 | 75.40 +/- 0.59 |

## Interpretation For Thesis

The supplementary experiments evaluate whether the baseline regularizers that underperform WD in the main experiments behave differently in simpler settings close to their original validation environments. On MNIST with a shallow CNN, WN, SN, SRR, WAR, SNR, and SNVR all reach a similar accuracy range to WD. On CIFAR-10 with the same shallow CNN family, WN, SRR, WAR, SNR, and SNVR again remain close to WD, while SN becomes slightly lower than WD. These results indicate that the low performance of these baselines in the main experiments is unlikely to be caused by implementation failure. Rather, their effectiveness depends strongly on the target setting, and the regularization effects observed in simple CNNs do not transfer consistently to modern architectures and more complex datasets.

SB remains substantially lower than WD on both MNIST and CIFAR-10, and the degradation is especially large on CIFAR-10. This is consistent with the fact that SB enforces a strict Lipschitz-oriented constraint, which can strongly restrict representational capacity in supervised classification. Therefore, SB is better interpreted as a GAN-stability-oriented method than as a general-purpose classifier regularizer. SN is milder than SB and performs well on MNIST, but it is still lower than WD on CIFAR-10 and does not consistently improve classification performance.

## Suggested Response 2.3

[Response 2.3] 지적해주신 WD 대비 일부 baseline의 낮은 성능에 대한 설명을 보강하기 위해, 해당 방법들이 제안되거나 주로 검증된 환경에 가까운 simple CNN 기반 추가 실험을 MNIST와 CIFAR-10에서 수행하였습니다. 실험은 2개의 convolution layer와 2개의 fully connected layer로 구성된 shallow CNN을 사용하였고, 동일한 SGD 학습 조건에서 5개 random seed 평균 Top-1 accuracy를 비교하였습니다.

MNIST에서는 WD 99.21 +/- 0.04%, WN 99.20 +/- 0.05%, SN 99.28 +/- 0.06%, SRR 99.19 +/- 0.02%, WAR 99.21 +/- 0.05%, SNR 99.22 +/- 0.04%, SNVR 99.20 +/- 0.05%로 대부분 WD와 유사한 성능을 보였습니다. CIFAR-10에서도 WD 75.32 +/- 0.49%, WN 75.04 +/- 0.57%, SRR 75.23 +/- 0.83%, WAR 75.36 +/- 0.59%, SNR 75.30 +/- 0.92%, SNVR 75.40 +/- 0.59%로 WD와 유사한 범위의 성능을 보였습니다. 따라서 WN, SRR, WAR 등이 main experiment에서 WD보다 낮은 성능을 보인 것은 구현 오류라기보다는, 비교적 단순한 CNN 및 MNIST/CIFAR-10 수준의 설정에서 관찰된 효과가 ResNet/Conv2NeXt와 같은 상용 수준 architecture 및 CIFAR-100, TinyImageNet, STL-10, Food-101과 같은 더 복잡한 multi-class classification task로 확장될 때 일관되게 유지되지 않기 때문으로 해석할 수 있습니다.

반면 SB는 MNIST에서 95.58 +/- 0.72%, CIFAR-10에서 42.64 +/- 0.94%로 낮은 성능을 보였습니다. 이는 strict Lipschitz constraint가 classification accuracy 개선보다는 GAN 학습 안정화 목적에 더 적합하며, 일반적인 supervised classification에서는 표현력을 과도하게 제한할 수 있음을 보여줍니다. SN 역시 MNIST에서는 WD와 유사했지만 CIFAR-10에서는 74.00 +/- 0.40%로 WD보다 낮아, GAN 학습에서 유래한 spectral constraint가 일반 classification task에서 항상 유리하지는 않음을 확인하였습니다. 해당 추가 실험과 해석을 최종 논문에 반영하겠습니다.
