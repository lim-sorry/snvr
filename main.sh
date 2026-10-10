
for seed in 0 1 2
do
  for dataset in cifar10 cifar100 stl10
  do
    python -m torch.distributed.launch --use_env main.py --cfg config/resnet18/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_sb_weight --name sb --tag 1.0 --lr 0.001 --seed $seed
    python -m torch.distributed.launch --use_env main.py --cfg config/resnet34/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_sb_weight --name sb --tag 1.0 --lr 0.001 --seed $seed
    python -m torch.distributed.launch --use_env main.py --cfg config/conv2next-tiny/$dataset/baseline.yaml --local_rank 0 --data-path ../data --output output_sb_weight --name sb --tag 10.0 --lr 0.001 --seed $seed
  done
  python -m torch.distributed.launch --use_env main.py --cfg config/resnet18/food101/baseline.yaml --local_rank 0 --data-path ../data --output output_sb_weight --name sb --tag 1.0 --lr 0.001 --seed $seed
  python -m torch.distributed.launch --use_env main.py --cfg config/resnet34/food101/baseline.yaml --local_rank 0 --data-path ../data --output output_sb_weight --name sb --tag 1.0 --lr 0.001 --seed $seed
done

