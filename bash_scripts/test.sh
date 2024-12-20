#!/bin/bash
export src_path=/home/js/repo/spectral-norm-variance-regularization
export ckpt_src_path=/home/js/repo/spectral-norm-variance-regularization/ckpt
export data_path=/home/js/repo/spectral-norm-variance-regularization/data

for line in 91 92 93 94 96 97 98 99
    do 
        CUDA_VISIBLE_DEVICES=0 ./bash_scripts/train.sh ${line} ${src_path} ${ckpt_src_path} ${data_path}
    done