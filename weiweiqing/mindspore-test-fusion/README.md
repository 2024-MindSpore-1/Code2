# vision-based-food-nutrition-estimation-via-RGB-D-fusion-network

## What can I find here?

This repository contains all code and implementations used in:

```
Vision-based food nutrition estimation via RGB-D fusion network

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
Dataset metadata is stored in nutrition5k_dataset. The organizational form of the dataset is as follows:

```
|-nutrition5k_dataset
    |---imagery
        |---realsense_overhead
            |---Dish1
		|---depth_color.png
		|---rgb.png
            |---Dish2
                |---depth_color.png
		|---rgb.png
            ......
            |---DishM
                |---
```

Also,The labels for the training and testing sets are as follows:

```
Training set tags:
rgbd_train_processed.txt  
rgb_in_overhead_train_processed.txt
Testing set tags:
rgbd_test_processed.txt
rgb_in_overhead_test_processed.txt
```

Before you start training,you can store the tags in the imagery folder like this:

```
|-nutrition5k_dataset
    |---imagery
        |---realsense_overhead
            |---Dish1
            ......
            |---DishM
        |---rgbd_train_processed.txt  
	|---rgb_in_overhead_train_processed.txt
	|---rgbd_test_processed.txt
	|---rgb_in_overhead_test_processed.txt
```

You could set the data directory by `data_root` in `/run/run_fusion/args.py`, also set other options freely

### Training:
Training is done by using `/run/run_fustion/funsion_run.py`, like the following:
```
python /run/run_fustion/funsion_run.py
```


