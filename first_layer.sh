#!/bin/bash
# SNVR with the first layer included (SKIP_FIRST: FALSE) on the main-result settings.
# Finished runs are moved to output/<model>/<dataset>/snvr_first/<seed>.
# usage: SEEDS="0 1 2" bash first_layer.sh

cd "$(dirname "$0")"

# NCCL init fails while NVML reports a driver/library mismatch; gloo is fine for 1 GPU.
export DIST_BACKEND=${DIST_BACKEND:-gloo}
# override on other machines, e.g. TORCHRUN=torchrun bash first_layer.sh
TORCHRUN=${TORCHRUN:-/home/dm22/miniconda3/envs/snvr/bin/torchrun}
SEEDS=${SEEDS:-0}
STAGING=output_staging
METHOD=snvr_first

run_lane() {
  local port=$1; shift
  for seed in $SEEDS; do
    for job in "$@"; do
      local model=${job%/*} dataset=${job#*/}
      local dst=output/$model/$dataset/$METHOD/$seed
      local src=$STAGING/$model/$dataset/${METHOD}_$seed/$seed
      if [ -f $dst/results.pkl ]; then
        continue
      fi
      mkdir -p $STAGING/logs
      $TORCHRUN --nproc_per_node 1 --master_port $port main.py \
        --cfg config/$model/$dataset/$METHOD.yaml --data-path ../data \
        --output $STAGING --tag $seed --seed $seed \
        > $STAGING/logs/${model}_${dataset}_${METHOD}_${seed}.log 2>&1
      if [ -f $src/results.pkl ]; then
        mkdir -p $(dirname $dst)
        mv $src $dst
      else
        echo "[FAILED] $model/$dataset/$METHOD/$seed (see $STAGING/logs/${model}_${dataset}_${METHOD}_${seed}.log)"
      fi
    done
  done
}

# run only the given jobs as a single lane, e.g. bash first_layer.sh resnet18/cifar10 resnet18/stl10
if [ $# -gt 0 ]; then
  run_lane ${PORT:-29636} "$@"
  echo "[DONE] $METHOD $*"
  exit
fi

# CIFAR-10/100 and STL-10 only for now (TinyImageNet / Food-101 take too long);
# at most two Conv2NeXt-T runs at once to stay within GPU memory (STL-10 alone takes ~19GB,
# so two STL-10 runs at once go OOM: rerun them with `SEEDS="1 2" bash first_layer.sh conv2next-tiny/stl10`)
for seed in $SEEDS; do
  SEEDS=$seed run_lane $((29630 + seed)) conv2next-tiny/cifar10 conv2next-tiny/cifar100 conv2next-tiny/stl10 &
done
run_lane 29641 resnet18/cifar10 resnet18/cifar100 resnet18/stl10 &
run_lane 29642 resnet34/cifar10 resnet34/cifar100 resnet34/stl10 &
wait
echo "[DONE] $METHOD"
