# conda setup
conda env create -f environment.yml
conda activate snvr

sudo apt update
sudo apt install nvidia-cudnn
sudo apt clean

# data setup
mkdir -p ../data

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ../data & rm tiny-imagenet-200.zip
mv ../data/tiny-imagenet-200 ../data/tiny-imagenet
python tiny-imagenet.py

wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz -C ../data & rm food-101.tar.gz
mv ../data/food-101 ../data/food101
python food-101.py