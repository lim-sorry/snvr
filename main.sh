# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/snvr.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/snr.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/tb.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/wn.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0
# python -m torch.distributed.launch --use_env grad_norm_trace.py --cfg config/resnet34/cifar100/sn.yaml --local_rank 0 --data-path ../data --tag 0 --seed 0


for i in {0..4}
do
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snvr_unify.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/wn.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/sn.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/tb.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

done

for i in {0..0}
do
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/baseline_default.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/baseline_default.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/baseline_default.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/baseline_default.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/baseline_default.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    # python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/cifar10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/tiny-imagenet/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/stl10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/food101/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/snr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/snvr.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/snvr1.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/snvr1.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/snvr1.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/snvr1.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    # python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/snvr1.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
done
 