# -*- coding: utf-8 -*-
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors
import seaborn as sns
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phraser, Phrases
import codecs
import sys
import nltk
import numpy as np
import matplotlib.cm as cm

def load_vocab(path):
    res = set()
    for line in codecs.open(path, 'r', 'utf-8'):
        res.add(line.strip())

    print('loading words vocab finished: ', len(res))
    return res


def tsne_plot(vocab, model):

    tsne_model = TSNE(n_components=2, random_state=np.random.RandomState(110), learning_rate=1)
    plt.figure(figsize=(10, 10))
    ax1 = plt.axes(frameon=False)
    ax1.get_xaxis().tick_bottom()
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)

    x = np.arange(len(vocab))
    ys = [i+x+(i*x)**2 for i in range(len(vocab))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    marks = ['o', 'v', '^', 's', 'p', 'h', 'H', 'D']

    labels, tokens, c, m = [], [], [], []
    for i, item in enumerate(vocab):
        for word in item:
            if word in model.wv.vocab:
                tokens.append(model.wv[word])
                labels.append(word)
                c.append(colors[i])
                m.append(marks[i])

    new_values = tsne_model.fit_transform(tokens)

    res_x, res_y, res_color, res_marker = [], [], [], []
    for index, label in enumerate(labels):
        x, y = new_values[index, :]
        res_x.append(x)
        res_y.append(y)
        res_color.append(m[index])
        res_marker.append(c[index])
        plt.scatter(x, y, marker=m[index], color=c[index], s=450)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        print((x, y, label))

    print(res_x)
    print(res_y)
    print(labels)
    plt.show()

def uni_test(model):
    words = load_vocab('data/vocab/vocab.gigaword.txt')
    most_similar_words = {}
    for word in words:
        most_similar_words[word] = model.most_similar(word, topn=5)

    res = sorted(most_similar_words.items(), key=lambda k: k[1][0][1], reverse=True)

    for item in list(res):
        print(item)

def remove_stop_words(vocab_path):
    STOP_WORDS = nltk.corpus.stopwords.words()
    vocab = load_vocab(vocab_path)
    count_vocab = dict(filter(lambda item: item[1] >= 5e3 and item[1] <= 5.3e3 and item[0] not in STOP_WORDS, vocab.items()))
    print('count vocab len:', len(count_vocab))
    return count_vocab

def tsne_plot(model_path):

    model1 = KeyedVectors.load_word2vec_format(model_path, binary=False)

    vocab1 = load_vocab('..\\data\\vocab\\time.txt')
    vocab2 = load_vocab('..\\data\\vocab\\city.txt')
    vocab3 = load_vocab('..\\data\\vocab\\medal.txt')
    vocab4 = load_vocab('..\\data\\vocab\\moon.txt')
    vocab5 = load_vocab('..\\data\\vocab\\animal.txt')
    vocab6 = load_vocab('..\\data\\vocab\\party.txt')
    vocab7 = load_vocab('..\\data\\vocab\\clothing.txt')
    vocab8 = load_vocab('..\\data\\vocab\\creator.txt')

    tsne_plot([vocab1, vocab2, vocab3, vocab4, vocab5, vocab6, vocab7, vocab8], model1)


def similar_word(model_path, target_word, context_word):
    vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return vectors.similarity(target_word, context_word)

def gender_debias(target_word, context_word):
    gender_word_pairs = {
        ('boy', 'girl'),
        ('woman', 'man'),
        ('she', 'he'),
        ('mother', 'father'),
        ('daughter', 'son'),
        ('gal', 'guy'),
        ('female', 'male'),
        ('her', 'his'),
        ('herself', 'himself'),
        ('Mary', 'John')
    }

def norm_to_unit(model_path):
    outpath = model_path + '.norm'
    vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
    f_w = codecs.open(outpath, mode='w+', encoding='utf-8')
    for item in vectors.vocab:
        len = np.linalg.norm(vectors.wv[item], ord=1)
        uni_vectors = vectors.wv[item] / len
        f_w.write(item + ' ' + ' '.join(str(item) for item in uni_vectors.tolist()) + '\n')

if __name__ == '__main__':

    word2vec = KeyedVectors.load_word2vec_format('D:\\data\\wiki\\finalvectors\\wiki.basew2v.skipgram.dim300.iter5.vectors.txt')
    model1 = KeyedVectors.load_word2vec_format('D:\\data\\wiki\\vectors\\wiki.word.glove.iter100.xmax10.dim300.vectors.txt', binary=False)
    # model_path = 'D:\\data\\wiki\\vectors\\wiki.word.san.divide.window5.count.iter100.dim300.xmax10.vectors.txt'
    # model_path = 'D:\\data\\wiki\\vectors\\wiki.word.san.reciprocal.window5.count.iter100.dim300.xmax10.vectors.txt'
    model2 = KeyedVectors.load_word2vec_format('D:\\data\\wiki\\vectors\\wiki.word.mlm.divide.count.iter100.dim300.xmax10.filter.purewords.vectors.txt', binary=False)
    # model_path = 'D:\\data\\wiki\\vectors\\wiki.word.mlm.reciprocal.count.iter100.dim300.xmax10.filter.pure.vectors.txt'
    # count_vocab = remove_stop_words('..\\data\\vocab\\vocab.wiki.word.txt')
    she_pairs = ['homemaker', 'nurse', 'receptionist', 'librarian', 'socialite', 'hairdresser', 'nanny', 'bookkeeper', 'stylist', 'housekeeper']
    he_pairs = ['maestro', 'skipper', 'protege', 'philosopher', 'captain', 'architect', 'financier', 'warrior', 'broadcaster', 'magician']

    print(word2vec.most_similar('he'))
    print(word2vec.most_similar('she'))

    print(model1.most_similar('he'))
    print(model1.most_similar('she'))

    print(model2.most_similar('he'))
    print(model2.most_similar('she'))

    for word in she_pairs:
        print('she: ' + word)
        print(model1.similarity(word, 'she'))
        print(model2.similarity(word, 'she'))

    for word in he_pairs:
        print('he: ' + word)
        print(word2vec.similarity(word, 'she'))
        print(model1.similarity(word, 'he'))
        print(model2.similarity(word, 'he'))
