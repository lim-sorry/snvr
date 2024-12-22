#!/bin/bash
export src_path=.
export ckpt_src_path=./ckpt
export data_path=./data

## download the TIN dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
mkdir -p data
mv tiny-imagenet-200.zip data
cp tiny_imagenet_setup.sh data
cd data
unzip tiny-imagenet-200.zip
chmod +x tiny_imagenet_setup.sh
./tiny_imagenet_setup.sh