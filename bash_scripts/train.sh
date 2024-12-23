export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

experiment_num=$1
cfg=$(sed -n "$experiment_num"p ${src_path}/bash_scripts/config/test.csv)

#dataset,usage,model,depth,lr,snr,snvr,seed,accuracy,remark
dataset=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $1); print $1}')
usage=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $2); print $2}')
model=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $3); print $3}')
depth=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $4); print $4}')
lr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $5); print $5}')
snr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $6); print $6}')
snvr=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $7); print $7}')
seed_lst=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $8); print $8}')
# accuracy=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $9); print $9}')
remark=$(echo $cfg | awk -F, '{gsub(/^ *| *$/, "", $10); print $10}')
remark=${remark:-"None"}

# create folder for saving checkpoints
base_path=${ckpt_src_path}/${dataset}_${usage}/${model}_${depth}/${lr}_${snr}_${snvr}
mkdir -p ${base_path}

# run experiments with different seeds
seeds=$(echo $seed_lst | tr ";" "\n")
for seed in $seeds
    do 
        echo "Running experiment with seed ${seed}"
        ckpt_path=${base_path}/${seed}
        mkdir -p ${ckpt_path}

        python main.py \
            --dataset ${dataset} \
            --usage ${usage} \
            --model ${model} \
            --depth ${depth} \
            --lr ${lr} \
            --snr ${snr} \
            --snvr ${snvr} \
            --seed ${seed} \
            --remark ${remark} \
            --ckpt_path ${ckpt_path} \
            --data_path ${data_path}
    done
