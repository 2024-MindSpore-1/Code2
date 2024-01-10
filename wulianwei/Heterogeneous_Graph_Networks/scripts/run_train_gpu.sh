#!/bin/bash
if [ $# -lt 2 ]
then
    echo "Usage: bash run_train_gpu.sh [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]"
exit 1
fi

export DEVICE_NUM=1
DATASET_PATH=$2

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

export CUDA_VISIBLE_DEVICES="$1"

python train.py --datapath=$DATASET_PATH --ckptpath=../ckpts \
                --device_target='GPU' --num_epoch=680 \
                --dist_reg=0 > log 2>&1 &

cd ..
