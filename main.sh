for i in {0..0}
do
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/cifar10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/tiny-imagenet/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/stl10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/food101/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/cifar100/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/tiny-imagenet/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/stl10/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin-tiny/food101/baseline.yaml --local_rank 0 --data-path ../data --tag $i --seed $i

done
 