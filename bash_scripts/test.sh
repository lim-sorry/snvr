#!/bin/bash
export src_path=.
export ckpt_src_path=./ckpt
export data_path=./data

for line in 104
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done