export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# if using windows os
# set CUBLAS_WORKSPACE_CONFIG=:4096:8 

SLURM_ARRAY_TASK_ID=$1
src_path=/home/js/repo/TempBalance
ckpt_src_path=/home/js/repo/TempBalance/ckpt
data_path=/home/js/repo/TempBalance/data

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/bash_scripts/config/test.tsv)

netType=$(echo $cfg | cut -f 1 -d ' ')
width=$(echo $cfg | cut -f 2 -d ' ')
depth=$(echo $cfg | cut -f 3 -d ' ')
dataset=$(echo $cfg | cut -f 4 -d ' ')
usage=$(echo $cfg | cut -f 5 -d ' ')
num_epochs=$(echo $cfg | cut -f 6 -d ' ')
lr_sche=$(echo $cfg | cut -f 7 -d ' ')
seed_lst=$(echo $cfg | cut -f 8 -d ' ')
lr=$(echo $cfg | cut -f 9 -d ' ')
weight_decay=$(echo $cfg | cut -f 10 -d ' ')
sg=$(echo $cfg | cut -f 11 -d ' ')
svg=$(echo $cfg | cut -f 12 -d ' ')
optim_type=$(echo $cfg | cut -f 13 -d ' ')


# saving folders of trained models 
base_path=${ckpt_src_path}/${dataset}/usage${usage}/${netType}_${depth}_${width}
base_path=${base_path}/${optim_type}_init${lr}_${lr_sche}
base_path=${base_path}/snr_sg${sg}_snvr_svg${svg}

# array=($(echo $seed_lst | tr ',' ' '))

# run five random seeds 
for seed in 43 37 13 51 71 #24 68 16 77 98
    do 
        echo run experiment with seed $seed
        ckpt_folder=epochs${num_epochs}_wd${weight_decay}_seed${seed}
        mkdir -p ${base_path}/${ckpt_folder}

        python main.py \
            --lr ${lr} \
            --net-type ${netType} \
            --depth ${depth} \
            --widen-factor ${width} \
            --num-epochs ${num_epochs} \
            --seed ${seed} \
            --dataset ${dataset} \
            --usage ${usage} \
            --optim-type ${optim_type} \
            --lr-sche ${lr_sche} \
            --weight-decay ${weight_decay} \
            --sg ${sg} \
            --svg ${svg} \
            --ckpt-path ${base_path}/${ckpt_folder} \
            --datadir ${data_path}
    done
