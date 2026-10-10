#!/bin/bash
# ViT-Tiny: WD (baseline) vs SNR vs SNVR on CIFAR-10/100, STL-10, TinyImageNet.
# Runs are split into 5 lanes that share the GPU concurrently (SNR/SNVR are CPU-bound
# by the per-layer power iteration loop, so TinyImageNet gets its own lane for them).
# Finished runs are moved to output/vit-tiny/<dataset>/<method>/<seed>.
# usage: SEEDS="0 1 2" bash vit_tiny.sh

cd "$(dirname "$0")"

# NCCL init fails while NVML reports a driver/library mismatch; gloo is fine for 1 GPU.
export DIST_BACKEND=${DIST_BACKEND:-gloo}
# override on other machines, e.g. TORCHRUN=torchrun bash vit_tiny.sh
TORCHRUN=${TORCHRUN:-/home/dm22/miniconda3/envs/snvr/bin/torchrun}
SEEDS=${SEEDS:-0}
STAGING=output_staging
declare -A NAME=([baseline]=wd [snr]=snr [snvr]=snvr)

run_lane() {
  local method=$1 port=$2 datasets=$3
  for seed in $SEEDS; do
    for dataset in $datasets; do
      local dst=output/vit-tiny/$dataset/$method/$seed
      local src=$STAGING/vit-tiny/$dataset/${NAME[$method]}_$seed/$seed
      if [ -f $dst/results.pkl ]; then
        continue
      fi
      mkdir -p $STAGING/logs
      $TORCHRUN --nproc_per_node 1 --master_port $port main.py \
        --cfg config/vit-tiny/$dataset/$method.yaml --data-path ../data \
        --output $STAGING --tag $seed --seed $seed \
        > $STAGING/logs/${dataset}_${method}_${seed}.log 2>&1
      if [ -f $src/results.pkl ]; then
        mkdir -p $(dirname $dst)
        mv $src $dst
      else
        echo "[FAILED] $dataset/$method/$seed (see $STAGING/logs/${dataset}_${method}_${seed}.log)"
      fi
    done
  done
}

run_lane baseline 29621 "cifar10 cifar100 stl10 tiny-imagenet" &
run_lane snr      29622 "cifar10 cifar100 stl10" &
run_lane snvr     29623 "cifar10 cifar100 stl10" &
run_lane snr      29624 "tiny-imagenet" &
run_lane snvr     29625 "tiny-imagenet" &
wait
echo "[DONE] vit-tiny"
