#!/bin/bash
# Main experiments: every method on the given models / datasets, seeds 0-2 (one run after another).
#   bash run_main.sh                                   # everything (4 models x 6 datasets x 9 methods x 3 seeds)
#   MODELS="resnet34" DATASETS="cifar100" bash run_main.sh
#   MODELS="vit-tiny" DATASETS="cifar100 tiny-imagenet" METHODS="snr snvr" SEEDS="0" bash run_main.sh
MODELS=${MODELS:-"resnet18 resnet34 conv2next-tiny vit-tiny"}
DATASETS=${DATASETS:-"cifar10 cifar100 tiny-imagenet stl10 food101 eurosat"}
METHODS=${METHODS:-"wd wn sn sb tb war srr snr snvr"}
SEEDS=${SEEDS:-"0 1 2"}
DATA_PATH=${DATA_PATH:-../data}
OUTPUT=${OUTPUT:-output}

for m in $MODELS; do
  for d in $DATASETS; do
    for k in $METHODS; do
      for s in $SEEDS; do
        python main.py --cfg config/final/$m/$d/$k.yaml --data-path $DATA_PATH --output $OUTPUT --seed $s
      done
    done
  done
done
