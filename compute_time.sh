
for dataset in cifar100 stl10
do
    for model in resnet34 conv2next-tiny
    do
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name wd --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name wn --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name sn --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name sb --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name tb --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name srr --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_compute_time --name war --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/snr.yaml --local_rank 0 --data-path ../data --output output_compute_time --name snr --tag "" --lr 0.001 --seed 0
        python -m torch.distributed.launch --use_env compute_time.py --cfg config/$model/$dataset/snvr.yaml --local_rank 0 --data-path ../data --output output_compute_time --name snvr --tag "" --lr 0.001 --seed 0
    done
done
