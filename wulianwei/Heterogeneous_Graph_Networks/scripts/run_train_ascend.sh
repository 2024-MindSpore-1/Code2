#!/bin/bash
if [ $# != 1 ]
then
    echo "Usage: bash run_train_ascend.sh [DATASET_PATH]"
    exit 1
fi
DATASET_PATH=$1

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train

if [ -d "ckpts" ];
then
    rm -rf ./ckpts
fi
mkdir ./ckpts

cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit
env > env.log
echo "start training"

python train.py --datapath=$DATASET_PATH --ckptpath=../ckpts &> log &

cd ..
