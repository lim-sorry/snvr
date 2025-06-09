for i in {0..4}
do
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin_tiny_cifar100_baseline.yaml --local_rank 0 --data-path ~/js/repo/data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin_tiny_cifar100_snr_0.yaml --local_rank 0 --data-path ~/js/repo/data --tag $i --seed $i
    python -m torch.distributed.launch --use_env main.py --cfg config/eswin_tiny_cifar100_snvr_0.yaml --local_rank 0 --data-path ~/js/repo/data --tag $i --seed $i
done
