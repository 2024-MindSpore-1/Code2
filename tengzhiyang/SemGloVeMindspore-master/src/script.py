# -*- coding: utf-8 -*-
import codecs
import sys
from transformers import BertTokenizer

def test(path):
    vocabs = set()
    for line in codecs.open(path, 'r', 'utf-8'):
        parts = line.strip().split('\t')
        vocabs.add(parts[0])

    print('vocabs len:', len(vocabs))


def analysis(path, target_word, context_word):
    word_count = 0
    c_word_count = {}
    c_word_ratio = {}

    for line in codecs.open(path, mode='r', encoding='utf-8'):
        parts = line.strip().split('\t')
        if parts[0] == target_word:
            word_count += float(parts[2])
            if parts[1] in context_word:
                if parts[1] in c_word_count:
                    c_word_count[parts[1]] += float(parts[2])
                else:
                    c_word_count[parts[1]] = float(parts[2])

    for k, v in c_word_count.items():
        c_word_ratio[k] = c_word_count[k] / word_count

    print('word count:', word_count)
    print('co-occurrence count:', c_word_count)
    print('ratio:', c_word_ratio)

def remove_noisy_word(mlm_word_coo_path, glove_word_coo_path, tokenizer):
    glove_word_coo = {}
    output = codecs.open(mlm_word_coo_path + '.reduce1', mode='w+', encoding='utf-8')
    print('mlm word coo path:', mlm_word_coo_path)
    print('glove word coo path:', glove_word_coo_path)
    for idx, line in enumerate(codecs.open(glove_word_coo_path, mode='r', encoding='utf-8')):
        if (idx + 1) % 1e8 == 0:
            print('processing %s lines.' % (idx + 1))
            sys.stdout.flush()

        parts = line.strip().split('\t')
        glove_word_coo[(parts[0], parts[1])] = float(parts[2])

    print('finish load standard glove word co-occurrence.')

    for idx, line in enumerate(codecs.open(mlm_word_coo_path, mode='r', encoding='utf-8')):
        if (idx + 1) % 1e8 == 0:
            print('processing %s lines.' % (idx + 1))
            sys.stdout.flush()
        
        parts = line.strip().split('\t')

        if (parts[0], parts[1]) in glove_word_coo and glove_word_coo[(parts[0], parts[1])] > 1:
            output.write(line)

        elif len(tokenizer._tokenize(parts[0])) == 1 and len(tokenizer._tokenize(parts[1])) == 1:
            output.write(line) # write pure words


if __name__=="__main__":
    mlm_word_coo_path = sys.argv[1]
    glove_word_coo_path = sys.argv[2]
    tokenizer_path = sys.argv[3]
    bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    remove_noisy_word(mlm_word_coo_path, glove_word_coo_path, bert_tokenizer)
