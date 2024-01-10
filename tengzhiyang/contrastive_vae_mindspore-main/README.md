# Contrastive Latent Variable Models for Neural Text Generation
This repository contains the ongoing Mindspore implementations of our UAI 2022 paper  [Contrastive Latent Variable Models for Neural Text Generation](https://proceedings.mlr.press/v180/teng22a/teng22a.pdf). 

We propose the first neural text generalization models which apply contrastive learning over the latent variables in traditional VAEs. We integrate the constrastive learning over latent variables into an implicit VAE models (iVAEs), and improve the approximation of the KL divergence of iVAEs. 

In this repo, the code structures are almost the same as the [Implicit-LVM](https://github.com/fangleai/Implicit-LVM). For running each experiment, please refer to the repo of Implicit-LVM for the detailed instructions. For simplicity, we adapt the readme file of [Implicit-LVM](https://github.com/fangleai/Implicit-LVM) for our project. 

## Contents
1. [Proof-of-Concept on two Toy Datasets ](#Proof-of-Concept-on-two-Toy-Datasets )
2. [Language modeling on PTB](#Language-modeling-on-PTB)
    
## 1. Proof-of-Concept on two Toy Datasets 
a. The first toy dataset contains 4 data points **x**: 4 different one-hot four-dimensional vectors, and we learn the corresponding latent code **z** in 2D space for each **x**. Run the following in cmd lime:
```
cd toy_onehot/
python vae_onehot.py
python train_onehot.py
python train_onehot_clvae.py
```
The folder starts with ``results`` will show the intermediate pictures generated during learning. 

b. The second toy dataset contains K (k=8, 16) data points **y**: k different one-hot k-dimensional vectors, and we learn the corresponding latent code **z** in 2D space for each **x**. Run the following in cmd lime:
```
cd toy_onehot_K/
python vae_onehot.py
python train_onehot.py
python train_onehot_clvae.py
```
The folder starts with ``results`` will show the intermediate pictures generated during learning. 

## 2. Language modeling

### 2.1 Language modeling on PTB

After downloading, run
```
cd lang_model_ptb/
python preprocess_ptb.py --trainfile data/train.txt --valfile data/val.txt --testfile data/test.txt --outputfile data/ptb
```
This will create the `*.hdf5` files (data tensors) to be used by the model, as well as the `*.dict` file which contains the word-to-integer mapping for each word.

The command for training is for example
```
./train_mle_mi.sh
```

## Questions?
Please contact [Zhiyang](https://zeeeyang.github.io/) if you have any questions.

## Citation 

```
@InProceedings{pmlr-v180-teng22a,
  title = 	 {Contrastive latent variable models for neural text generation},
  author =       {Teng, Zhiyang and Chen, Chenhua and Zhang, Yan and Zhang, Yue},
  booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1928--1938},
  year = 	 {2022},
  editor = 	 {Cussens, James and Zhang, Kun},
  volume = 	 {180},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {01--05 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v180/teng22a/teng22a.pdf},
  url = 	 {https://proceedings.mlr.press/v180/teng22a.html},
  abstract = 	 {Deep latent variable models such as variational autoencoders and energy-based models are widely used for neural text generation. Most of them focus on matching the prior distribution with the posterior distribution of the latent variable for text reconstruction. In addition to instance-level reconstruction, this paper aims to integrate contrastive learning in the latent space, forcing the latent variables to learn high-level semantics by exploring inter-instance relationships. Experiments on various text generation benchmarks show the effectiveness of our proposed method. We also empirically show that our method can mitigate the posterior collapse issue for latent variable based text generation models. }
}
```
## Acknowledgement 

1. [Implicit-LVM](https://github.com/fangleai/Implicit-LVM)
```
@inproceedings{Fang_iLVM_2019_EMNLP,
  title={Implicit Deep Latent Variable Models for Text Generation},
  author={Le Fang, Chunyuan Li, Jianfeng Gao, Wei Dong, Changyou Chen},
  booktitle={EMNLP},
  year={2019}
}
```
2. [SimCSE](https://github.com/princeton-nlp/SimCSE)
```
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```
3. [ZLPR](https://spaces.ac.cn/archives/7359)
```
@misc{su2022zlpr,
      title={ZLPR: A Novel Loss for Multi-label Classification}, 
      author={Jianlin Su and Mingren Zhu and Ahmed Murtadha and Shengfeng Pan and Bo Wen and Yunfeng Liu},
      year={2022},
      eprint={2208.02955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
When citing this paper, please also consider citing the above papers.  Thanks. 

4. CCAI-Huawei Open Research Fund