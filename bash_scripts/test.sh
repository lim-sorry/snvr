#!/bin/bash

# SNR baseline
for line in 15 16
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done