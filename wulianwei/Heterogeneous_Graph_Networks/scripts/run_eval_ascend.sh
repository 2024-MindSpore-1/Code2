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

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval

cp ../*.py ./eval
cp ../*.yaml ./eval
cp *.sh ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
env > env.log
echo "start evaluation"

python eval.py --datapath=$DATASET_PATH --ckptpath=../ckpts &> log &

cd ..
