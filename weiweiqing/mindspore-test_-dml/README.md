# Towards-food-image-retrieval-via-generalization-oriented-sampling-and-loss-function-design

## What can I find here?

This repository contains all code and implementations used in:

```
Towards Food Image Retrieval via Generalization-oriented Sampling and Loss Function Design

```

### Requirements:

* Mindspore 2.1.0
* Python 3.7+
* faiss-gpu

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.7
(5) conda activate DL
(6) conda install mindspore=2.1.0 -c mindspore -c conda-forge
(7) conda install faiss-gpu
(8) Run the scripts!
```

### Datasets:
Data for
* ETH Food-101, Vireo Food-172, ISIA Food-500


* For ETH Food-101:
```
food101
└───images
|    └───apple_pie
|           │   134.jpg
|    ...
```

You could set the data directory by `source_path` in `/run/run_dml/opt.py`

### Training:
Training is done by using `dml_run.py`, like the following:
```
python /run/run_dml/dml_run.py
```


