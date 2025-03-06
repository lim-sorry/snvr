# export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1

lines=(2 3 4 5 6 7 8)

for line in "${lines[@]}"
    do
        cfg=$(sed -n "$line"p config.csv)

        # seed_lst = 42 51 66 71 94
        seed_lst=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $1); print $1}')

        dataset=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $2); print $2}')
        usage=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $3); print $3}')
        model=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $4); print $4}')
        lr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $5); print $5}')
        snr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $6); print $6}')
        snvr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $7); print $7}')
        weight_decay=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $8); print $8}')
        epochs=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $9); print $9}')
        warm_up=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $10); print $10}')
        batch_size=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $11); print $11}')
        optim=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $12); print $12}')

        for seed in $(echo "$seed_lst" | tr " " "\n")
            do
                python main.py \
                    --seed $seed \
                    --dataset $dataset \
                    --usage $usage \
                    --model $model \
                    --lr $lr \
                    --snr $snr \
                    --snvr $snvr \
                    --weight_decay $weight_decay \
                    --epochs $epochs \
                    --warm_up $warm_up \
                    --batch_size $batch_size \
                    --optim $optim \
                    --ckpt_path ckpt \
                    --data_path ../data
            done
    done
