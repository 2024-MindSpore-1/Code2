#!/bin/bash
ulimit -u unlimited

if [ $# -lt 2 ]
then
    echo "Usage: bash run_eval_gpu.sh [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]"
exit 1
fi

export DEVICE_NUM=1
DATASET_PATH=$2

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

export CUDA_VISIBLE_DEVICES="$1"

python eval.py --datapath=$DATASET_PATH --ckptpath=../ckpts \
               --device_target='GPU' --num_epoch=680 \
               --dist_reg=0 > log 2>&1 &

cd ..
