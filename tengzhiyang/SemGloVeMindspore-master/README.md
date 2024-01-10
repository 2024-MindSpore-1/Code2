
## Citation
This is the [MindSpore](https://www.mindspore.cn/en) implementation of [SemGloVe: Semantic Co-occurrences for GloVe from BERT](https://arxiv.org/abs/2012.15197). If you find this repository helpful, please cite our paper.
```
@article{gan2020semglove,
  title={SemGloVe: Semantic Co-occurrences for GloVe from BERT},
  author={Gan, Leilei and Teng, Zhiyang and Zhang, Yue and Zhu, Linchao and Wu, Fei and Yang, Yi},
  journal={arXiv preprint arXiv:2012.15197},
  year={2020}
}
```

## Dowload Pre-trained SemGloVe.
 - BERT-based [SemGloVe](https://pan.baidu.com/s/1HzEihfQD8qhnwKSKe9Lz7g) code: sema
 - ALBERTa-based [SemGloVe](https://pan.baidu.com/s/1dK0_Vahji1G2iS4RvaMWVA) code: sema

## Training
### 1. Dump semantic word co-occurrences using BERT or RoBERTa
```shell
python src/bert_cooccur.py --corpus_name wiki --model_name bert_large --divide --mlm_glove
```
### 2. Convert semantic word co-occurrences to bin file.
```shell
python src/bert_cooccur.py --txt2bin wiki --vocab data/vocab/vocab.wiki.word.txt 
```
### 3. Train SemGloVe using dumped word co-occurrences.
For example:
```shell
bash sh/bert_glove.sh mlm.word.coo.bin mlm.glove.vectors.txt
```

## Intrinsic Evaluation
 For intrinsic tasks, we use tools from [word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks.git). Specifically, you can evaluate SemGloVe use the following code:
 ```shell
 git clone https://github.com/kudkudak/word-embeddings-benchmarks.git
 cd word-embeddings-benchmarks
 python scripts/evaluate_on_all.py -f vectors.txt -p glove
 ``` 
## Contact
If you have any issues or questions about this repo, feel free to contact leileigan@zju.edu.cn.

### License
All work contained in this package is licensed under the Apache License, Version 2.0. See the include LICENSE file.
