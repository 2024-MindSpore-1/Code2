# -*- coding: utf-8 -*-
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import codecs
import sys
from gensim.models import KeyedVectors, word2vec
from gensim.models.fasttext import FastText
import fasttext
import datetime
import argparse

EN_VOCAB_PATH = '/mnt/data2/ganleilei/data/wwm_bert/wwm_uncased_L-24_H-1024_A-16/vocab.txt'

def gen_bpe_vocab(path, outpath):
    print('in path:', path)
    print('out path:', outpath)
    tokenizer = BertTokenizer.from_pretrained(EN_VOCAB_PATH)
    fout = codecs.open(outpath, 'w+', 'utf-8')
    reval = {}
    for line in codecs.open(path, 'r', 'utf-8'):
        for token in line.strip().split():
            if token not in reval:
                reval[token] = tokenizer.tokenize(token)

    for k, v in reval.items():
        fout.write(k + "\t" + "\t".join(v) + '\n')

    fout.close()

def gen_bpe_corpus(path, outpath):
    print('in path:', path)
    print('out path:', outpath)
    tokenizer = BertTokenizer.from_pretrained(EN_VOCAB_PATH)
    fout = codecs.open(outpath, 'w+', 'utf-8')
    for linenum, line in enumerate(codecs.open(path, 'r', 'utf-8')):
        if linenum % 1000 == 0:
            print('processing %d lines.' % linenum)
        tokenized_text = tokenizer.tokenize(line.strip())
        fout.write(' '.join(tokenized_text) + '\n')

    fout.close()

def gen_vec_from_bpe(vocab_path, bpe_vectors_path, outpath):
    print('vocab path:', vocab_path)
    print('bpe vectors path:', bpe_vectors_path)
    tokenizer = BertTokenizer.from_pretrained(EN_VOCAB_PATH)
    bpe_vectors = KeyedVectors.load_word2vec_format(bpe_vectors_path, binary=False)
    fout = codecs.open(outpath, 'w+', 'utf-8')
    for line in codecs.open(vocab_path, 'r', 'utf-8'):
        parts = line.strip().split()
        if len(parts) != 2:
            print('error line: ', line)
            continue
        word = parts[0]
        subwords = tokenizer.tokenize(word)
        word_vec = np.zeros(50, dtype=np.float16)
        for bpe_token in subwords:
            word_vec += bpe_vectors[bpe_token]
        fout.write(word + ' ' + ' '.join([str(item) for item in word_vec.tolist() / len(subwords)]) + '\n')

    fout.close()


def concat_bpe_vectors(bpe_vectors_path, word_bpe_pair_path, output_path):
    print('bpe vectors:', bpe_vectors_path)
    print('word bpe pair path:', word_bpe_pair_path)
    print('output path:', output_path)
    word_bpe_pair = {}
    fout = codecs.open(output_path, 'w+', 'utf-8')
    for line in codecs.open(word_bpe_pair_path, 'r', 'utf-8'):
        parts = line.strip().split('\t')
        word_bpe_pair[parts[0]] = parts[1:]
    bpe_vectors = KeyedVectors.load_word2vec_format(bpe_vectors_path)
    for word, bpes in word_bpe_pair.items():
        first = bpe_vectors[bpes[0]]
        fout.write(word + ' ' + ' '.join('%.8f' % item for item in first) + '\n')
    fout.close()


def average_word_vectors(vec1_path, vec2_path, outputpath):
    print('vec1 path:', vec1_path)
    print('vec2 path:', vec2_path)
    print('output path:', outputpath)
    vec1 = KeyedVectors.load_word2vec_format(vec1_path, binary=False)
    vec2 = KeyedVectors.load_word2vec_format(vec2_path, binary=False)
    fout = codecs.open(outputpath, 'w+', 'utf-8')

    count = 0
    for word in vec1.wv.vocab:
        if word not in vec2.wv.vocab:
            ave_vectors = vec1[word]
        else:
            count += 1
            ave_vectors = vec2[word]
        fout.write(word + ' ' + ' '.join(["%.8f" % item for item in ave_vectors]) + '\n')

    print('loading %d overlap words...' % count)
    fout.close()


if __name__ == '__main__':
    '''
    path = sys.argv[1]
    outpath = sys.argv[2]
    gen_bpe_vocab(path, outpath)
    '''

    '''
    path = sys.argv[1]
    outpath = sys.argv[2]
    gen_bpe_corpus(path, outpath)
    '''

    '''
    vocab_path = sys.argv[1]
    bpe_vectors_path = sys.argv[2]
    outpath = sys.argv[3]
    gen_vec_from_bpe(vocab_path, bpe_vectors_path, outpath)
    '''

    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description="Using fastext training word embeddings from bpe embedding.")
    parser.add_argument('--corpus')
    parser.add_argument('--bpedic', default='')
    parser.add_argument('--prebpevec', default='')
    parser.add_argument('--modelpath')
    parser.add_argument('--prewordvec', default='')
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--dim', default=300, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--min', default=3, type=int)
    parser.add_argument('--max', default=6, type=int)
    parser.add_argument('--useNgrams', default='True')
    parser.add_argument('--updateBpes', default='true')

    args = parser.parse_args()

    corpus_file = args.corpus
    bpe_dic = args.bpedic
    pre_bpe_vec = args.prebpevec
    model_path = args.modelpath
    epoch = args.epoch
    dim = args.dim
    pre_word_vec = args.prewordvec
    lr = args.lr
    min = args.min
    max = args.max
    useNgrams = True if args.useNgrams.lower() == 'true' else False
    updateBpes = True if args.updateBpes.lower() == 'true' else False

    print('corpus file: ', corpus_file)
    print('bpe pair dic: ', bpe_dic)
    print('pre bpe vec path: ', pre_bpe_vec)
    print('pre word vec path: ', pre_word_vec)
    print('save model path: ', model_path)
    print('epoch: ', epoch)
    print('dim: ', dim)
    print('lr:', lr)
    print('min:', min)
    print('max:', max)
    print('use Ngrams:', useNgrams)
    print('update bpes:', updateBpes)

    sys.stdout.flush()

    model = fasttext.train_unsupervised(corpus_file, lr=lr, epoch=epoch, dim=dim, model='skipgram', bpeDic=bpe_dic,
                                        pretrainedBpeVectors=pre_bpe_vec, pretrainedVectors=pre_word_vec, minn=min, maxn=max,
                                        useNgrams=useNgrams, updateBpes=updateBpes)
    model.save_model(model_path)
    model = FastText.load_fasttext_format(model_path)
    model.wv.save_word2vec_format(model_path + '.vectors.txt', binary=False)

    '''    

    concat_bpe_vectors('/mnt/data2/ganleilei/data/wiki_en/finalvectors/wiki.bertglove.bpe.reciprocal.iter100.dim300.correct.vectors.txt',
                       '../data/vocab/wiki.word.bpe.pair.txt',
                       'wiki.word.concat.bertglove.bpe.recirpocal.vectors.txt')
    '''