# Vision-based-fruit-recognition-via-multi-scale-attention-cnn

## What can I find here?

This repository contains all code and implementations used in:

```
Vision-Based Fruit Recognition via Multi-Scale Attention CNN

```

### Requirements:

* Mindspore 2.1.0
* Python 3.7+

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.7
(5) conda activate DL
(6) conda install mindspore=2.1.0 -c mindspore -c conda-forge
(7) Run the scripts!
```

### Datasets:
Data for
* Fruit-92, Fruit-360, FruitVeg-81


* For Fruit-92:
```
fru92
└───images
|    └───almond
|           │   f_07_01_0001.jpg
|    ...
```

You could set the data directory by `data_dir` in `main_run.py`

### Training:
Training is done by using `main_run.py`, like the following:
```
python /run/main_run.py
```


