#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/06/01
# @Contact: 11921071@zju.edu.cn

import codecs, struct
import sys, os, time, random
from ctypes import *
import torch
import datetime, argparse
import numpy as np
from tqdm import tqdm
import mindspore
from cybertron import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from torch.multiprocessing import Pool
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='false'

BERT_MAX_LEN = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {'bert-base-uncased'    : (BertModel,    BertForMaskedLM,    BertConfig,       BertTokenizer,     'bert-base-uncased'),
          'bert-large-uncased'  : (BertModel,    BertForMaskedLM,    BertConfig,       BertTokenizer,     'bert-large-uncased')
          }

class CustomDataset(Dataset):
    def __init__(self, model_name, dataset, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.dataset = dataset
        self._allennlp_tokenizer = PretrainedTransformerIndexer(model_name=model_name)._allennlp_tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        line_text, line_idx = self.dataset[index]
        if 'roberta' in self.model_name:
            words = line_text.strip().split()
            words = [words[0]] + [' '+ word for word in words[1:]]
        else:
            words = line_text.strip().split()
        
        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize(words)
        wordpiece_ids = [t.text_id for t in wordpieces]
        if len(wordpiece_ids) > BERT_MAX_LEN:
            print(f'Sample {line_idx} exceeding pre-trained model maximum length!')   
            print(line_text)
            return self[random.randint(0, len(self)-1)]

        return {
            'line_idx': line_idx,
            'line_text': line_text,
            'lengths': len(words),
            'offsets': offsets,
            'wordpiece_ids': mindspore.Tensor(wordpiece_ids, mindspore.int64),
            'word_masks': mindspore.Tensor([True] * len(words), mindspore.bool_),
            'wordpiece_masks': mindspore.Tensor([True] * len(wordpiece_ids), mindspore.bool_)
        }

def collate_fn(batch_data):
    output = {}
    batch_size = len(batch_data)
    max_pieces = max(x['wordpiece_masks'].shape[0] for x in batch_data)
    max_words = max(x['word_masks'].shape[0] for x in batch_data)

    for field in ['wordpiece_ids', 'wordpiece_masks']:
        pad_output = mindspore.numpy.full([batch_size, max_pieces], 0, dtype=batch_data[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch_data[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data            
        output[field] = pad_output

    for field in ['word_masks']:
        pad_output = mindspore.numpy.full([batch_size, max_words], 0, dtype=batch_data[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch_data[sample_idx][field]
            pad_output[sample_idx][: data.shape[0]] = data            
        output[field] = pad_output
    
    for field in ['offsets']:
        pad_output = []
        for sample_idx in range(batch_size):
            data = batch_data[sample_idx][field]
            pad_output.append(data)            
        output[field] = pad_output
    
    for field in ['line_idx', 'line_text']:
        pad_output = []
        for sample_idx in range(batch_size):
            data = batch_data[sample_idx][field]
            pad_output.append(data)            
        output[field] = pad_output
    
    for field in ['lengths']:
        pad_output = mindspore.numpy.full([batch_size], 0, dtype=mindspore.int64)
        for sample_idx in range(batch_size):
            data = batch_data[sample_idx][field]
            pad_output[sample_idx] = data            
        output[field] = pad_output
    
    return output

def load_data(corpus_path):
    dataset = []
    for linenum, line in tqdm(enumerate(codecs.open(corpus_path, 'r', 'utf-8', errors='ignore'))):
        dataset.append((line, linenum))

    return dataset

class CR(Structure):
    _fields_ = [('word1', c_int), ('word2', c_int), ('val', c_double)]


def read_from_bin(path):
    print('read from bin: ', path)
    with open(path, 'rb') as fin:
        x = CR()
        while fin.readinto(x) == sizeof(x):
            yield (x.word1, x.word2, x.val)

def read_word_bpe_pair(word_bpe_pair_path):
    pairs = {}
    for line in codecs.open(filename=word_bpe_pair_path, mode='r', encoding='utf-8'):
        parts = line.strip().split('\t')
        if len(parts) <= 0:
            print('Read word bpe pair error line:', line)
            continue
        word = parts[0]
        pairs[word] = parts[1:]

    print('Reading word bpe pairs from file: %s and size: %d' % (word_bpe_pair_path, len(pairs)))
    return pairs

def build_vocab(vocab_path):
    vocab = {}
    for index, line in enumerate(codecs.open(filename=vocab_path, mode='r', encoding='utf-8')):
        parts = line.strip().rsplit(maxsplit=1)
        if len(parts) != 2:
            print('Error line:', line)
            continue
        vocab[parts[0]] = index + 1

    if '[UNK]' not in vocab:
        vocab['[UNK]'] =  len(vocab)
    print("Reading vocab from file: %s and size: %d" % (vocab_path, len(vocab)))
    return vocab


def write_to_bin(path, x):
    '''
    x = CR()
    x.word1 = ''
    x.word2 = ''
    x.val = 123
    '''
    with open(path, 'wb') as fout:
        fout.write(x)

def convert_bin_to_txt(vocab_path, path, outpath):

    iter = read_from_bin(path)
    vocab = build_vocab(vocab_path)
    vocab = dict(zip(vocab.values(), vocab.keys()))
    fout = codecs.open(outpath, 'w', 'utf-8')
    for line in iter:
        fout.write('%s\t%s\t%.8f\n' % (vocab[line[0]], vocab[line[1]], line[2]))
    print('finish converting bin to text.')
    fout.close()

def convert_txt_to_bin(vocab_path, coo_path, out_path):

    vocab = build_vocab(vocab_path)
    fout = codecs.open(out_path, 'wb')
    for line in tqdm(codecs.open(coo_path, mode='r', encoding='utf-8')):
        parts = line.strip().split('\t')
        x = CR()
        if parts[0] in vocab and parts[1] in vocab:
            x.word1 = vocab[parts[0]]
            x.word2 = vocab[parts[1]]
            x.val = float(parts[2])
            fout.write(x)

    fout.close()
    print('finish converting txt to bin...')

def read_word_pair(path):
    word_pairs = set()
    for idx, line in enumerate(codecs.open(path, mode='r', encoding='utf-8')):
        if (idx + 1) % 1e6 == 0:
            print('processing %d number lines...' % (idx + 1))
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print('Read error line for pair count:', line)
            continue
        word1 = parts[0]
        word2 = parts[1]
        word_pairs.add((word1, word2))

    print('Finish reading pair from: %s and size: %d' % (path, len(word_pairs)))
    return word_pairs


def read_pair_count(path):
    print('read bpe pair count...')
    pair_count = {}
    for idx, line in enumerate(codecs.open(path, mode='r', encoding='utf-8')):
        if (idx + 1) % 1e6 == 0:
            print('processing %d number lines...' % (idx + 1))
            sys.stdout.flush()
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print('Read error line for pair count:', line)
            continue
        word1 = parts[0]
        word2 = parts[1]
        count = float(parts[2])
        pair_count[(word1, word2)] = count

    print('Finish reading pair from: %s and size: %d' % (path, len(pair_count)))
    return pair_count

def read_coo_matrix(file, res_coo):
    for line in tqdm(codecs.open(file, 'r', 'utf-8')):
        parts = line.strip().split('\t')
        k = (parts[0], parts[1])
        if k not in res_coo:
            res_coo[k] = float(parts[-1])
        else:
            res_coo[k] = float(parts[-1]) + res_coo[k]
    
    return res_coo

def merge_coo_matrix(path):
    res_coo = {}
    for file in os.listdir(path):
        coo_path = os.path.join(path, file)
        print("Merge coo path:", coo_path)
        sys.stdout.flush()
        read_coo_matrix(coo_path, res_coo)
    
    save_path = os.path.join(path, 'word.san.coo')
    print("final cooccurrence save path:", save_path)
    print("final cooccurrence size:", len(res_coo))
    write_table_to_file(res_coo, save_path)

#################### masked language model based glove ##############################################
def dump_mlm_predictions(corpus, outpath, batch_size, model, tokenizer, window_size):

    if not os.path.exists(corpus):
        raise ValueError('corpus file does not exit: ', corpus)

    dataset = load_data(corpus)
    custom_dataset = CustomDataset(model_name, dataset, tokenizer)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print("Finish building custom dataset!")
    fout = codecs.open(outpath, 'w+', 'utf-8')
    start = time.time()
    line_buffer, line_wordpieces_buffer, pred_ids_buffer, pred_scores_buffer = [], [], [], []

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if (batch_idx + 1) % 1e3 == 0:
            write_buffer = []
            for i in range(len(line_buffer)):
                # print("###" + line_buffer[i])
                write_buffer.append('###' + line_buffer[i].strip())
                target_tokens = line_wordpieces_buffer[i] #[max_wordpieces]
                context_token_ids = pred_ids_buffer[i] #[max_wordpieces, top_k]
                context_token_scores = pred_scores_buffer[i] #[max_wordpieces, top_k]
                for j in range(0, len(target_tokens)):
                    target_token = target_tokens[j]
                    c_tokens = tokenizer.convert_ids_to_tokens(context_token_ids[j+1], skip_special_tokens=True)
                    c_tokens_len = len(c_tokens)
                    c_token_scores = context_token_scores[j+1][:c_tokens_len]
                    # print(f'target token: {target_token} and context_tokens: {c_tokens}')
                    condidate = target_token + ' ' + ' '.join([item[0] + ':' + str(item[1]) for item in list(zip(c_tokens, c_token_scores))])
                    write_buffer.append(condidate)

            print('%.2fs writing %d batch dump data.' % (time.time() - start, batch_idx + 1))
            [fout.write(item + '\n') for item in write_buffer]

            sys.stdout.flush()
            fout.flush()
            line_buffer, line_wordpieces_buffer, pred_ids_buffer, pred_scores_buffer = [], [], [], []


        input_ids, masks, line_texts = batch_data['wordpiece_ids'], batch_data['wordpiece_masks'], batch_data['line_text']
        ori_tokenized_text = [tokenizer.convert_ids_to_tokens(item, skip_special_tokens=True) for item in input_ids]
        lengths, offsets = batch_data["lengths"], batch_data['offsets'] #[batch_size, max_word, 2]
        batch_size = len(lengths)
        masked_lm_logits_scores = model(input_ids=input_ids, attention_mask=masks)[0]
        # masked_lm_logits_scores = model(input_ids=input_ids, attention_mask=masks).logits #[batch_size, max_word, vocab_size]
        topk = mindspore.ops.TopK(sorted=True)
        top_scores, top_score_ids = topk(masked_lm_logits_scores,window_size+1) #[batch_size, max_wordpieces, top_k]

        line_buffer.extend(line_texts)
        line_wordpieces_buffer.extend(ori_tokenized_text)
        pred_ids_buffer.extend(top_score_ids.asnumpy().tolist())
        pred_scores_buffer.extend(top_scores.asnumpy().tolist())
            

    print('writing final buffer data......')
    write_buffer = []
    for i in range(len(line_buffer)):
        # print("###" + line_buffer[i])
        write_buffer.append('###' + line_buffer[i].strip())
        target_tokens = line_wordpieces_buffer[i] #[max_wordpieces]
        context_token_ids = pred_ids_buffer[i] #[max_wordpieces, top_k]
        context_token_scores = pred_scores_buffer[i] #[max_wordpieces, top_k]
        for j in range(0, len(target_tokens)):
            target_token = target_tokens[j]
            c_tokens = tokenizer.convert_ids_to_tokens(context_token_ids[j+1], skip_special_tokens=True)
            c_tokens_len = len(c_tokens)
            c_token_scores = context_token_scores[j+1][:c_tokens_len]
            # print(f'target token: {target_token} and context_tokens: {c_tokens}')
            condidate = target_token + ' ' + ' '.join([item[0] + ':' + str(item[1]) for item in list(zip(c_tokens, c_token_scores))])
            write_buffer.append(condidate)

    print('%.2fs writing %d batch dump data.' % (time.time() - start, batch_idx + 1))
    [fout.write(item + '\n') for item in write_buffer]
    sys.stdout.flush()
    fout.close()


def get_mlm_bpe_cooccurr_from_dump_file(window_size, divide, reciprocal, dump_file, coo_path):

    if not os.path.exists(dump_file):
        raise ValueError('dump file does not exit: ', dump_file)

    start = time.time()
    bigram_table, line_num, pre_line = {}, 0, ''

    for line in codecs.open(dump_file, 'r', 'utf-8'):

        if line.startswith('###'):
            line_num += 1
            pre_line = line
            if line_num % 1e5 == 0:
                print('%.2fs processing %d line text.' % (time.time() - start, line_num))
                sys.stdout.flush()
        else:
            parts = line.strip().split()
            if len(parts) < 2:
                print('wrong line: ', len(parts))
                continue
            else:
                context_tokens, top_scores, target_token = [], [], parts[0]
                bench = float(parts[1].rsplit(':', maxsplit=1)[1]) # use first predict token score as benchmark
                filer_parts = list(filter(lambda x: x.rsplit(':')[0] != target_token, parts[1:]))
                for item in filer_parts[ : window_size]:
                    sub_parts = item.rsplit(':')
                    if len(sub_parts) == 2:
                        context_tokens.append(sub_parts[0])
                        top_scores.append(float(sub_parts[1]))

                if reciprocal:
                    scores = np.reciprocal(np.linspace(1, len(context_tokens), num=len(context_tokens)))

                elif divide:
                    top_scores = [item / bench for item in top_scores]
                    scores = top_scores

                for index, pair in enumerate(zip([target_token] * len(context_tokens), context_tokens)):
                    if pair == ("cat", "cat"):
                        print('add <cat, cat> pair from: %s and target token: %s ...' % (pre_line, target_token))
                    if pair in bigram_table and scores[index] > 1e-9:
                        bigram_table[pair] += scores[index]
                    else:
                        bigram_table[pair] = scores[index]

    write_table_to_file(bigram_table, coo_path)


def cal_word_pair_count_from_bpe_pair_count(word_pair_path, bpe_coo_path, save_path, coo_scale, vocab_path, tokenizer):

    print('cal word pair count from bpe pair count...')
    if not os.path.exists(vocab_path) or not os.path.exists(word_pair_path) or not os.path.exists(bpe_coo_path):
        raise ValueError('path not exits: %s, %s, %s' % (vocab_path, word_pair_path, bpe_coo_path))

    vocab = build_vocab(vocab_path)
    bpe_coo_count = read_pair_count(bpe_coo_path)
    word_pair_count, pure_words = {}, set()

    for word in vocab:
        word_pieces = tokenizer._tokenize(word)
        if len(word_pieces) > 1:
            pure_words.add(word)
    print('pure words len: %d' % len(pure_words))

    for idx, line in enumerate(codecs.open(word_pair_path, 'r', 'utf-8')):
        if (idx + 1) % 1e6 == 0:
            print('processing %d number lines...' % (idx + 1))
            sys.stdout.flush()

        item = line.strip().split('\t')
        c_word, t_word, coo_count = item[0], item[1], item[2]

        if coo_count < 1: # ignore rare word co-occurrence count
            continue

        c_word_bpes = tokenizer._tokenize(c_word)
        t_word_bpes = tokenizer._tokenize(t_word)

        sum, pair_count = 0, 0
        for c_bpe in c_word_bpes:
            for t_bpe in t_word_bpes:
                pair_count += 1
                if (c_bpe, t_bpe) in bpe_coo_count:
                    sum += bpe_coo_count[(c_bpe, t_bpe)]

        if pair_count > 0 and sum / pair_count >= 1e-8:
            if len(c_word_bpes) > 1 or len(t_word_bpes) > 1:
                val = (sum * coo_scale / pair_count) ## scale word from bpe
            else:
                val = sum / pair_count
            word_pair_count[(c_word, t_word)] = val

    addition_count = 0
    for k, v in bpe_coo_count.items():
        if k[0] in pure_words and k[1] in pure_words and k not in word_pair_count:
            addition_count += 1
            word_pair_count[k] = v

    print('add addition count %d.' % addition_count)

    fout = codecs.open(save_path, mode='w+', encoding='utf-8')
    print('writing %d word pair count to %s...' % (len(word_pair_count), save_path))
    for pair, count in word_pair_count.items():
        fout.write('%s\t%s\t%.8f\n' % (pair[0], pair[1], count))

    fout.close()

#################### self attention based glove #########################################################

def weight_sum(input):
    #start_time = time.time()
    batch_weights, item_idx, word_i, word_j, start_i, end_i, start_j, end_j = input
    weight_ij = batch_weights[item_idx, start_i: end_i+1, start_j: end_j+1].sum() / ((end_i-start_i+1) * (end_j - start_j + 1))
    #print("consumed time:", time.time() - start_time)
    return (item_idx, word_i, word_j, weight_ij)

def extract_word_word_attn_weights(pool, total_weights, total_offsets, total_lines, total_lengths):
    write_res = []
    for (batch_weights, batch_offsets, batch_lines, batch_lengths) in zip(total_weights, total_offsets, total_lines, total_lengths):
        batch_size, params, batch_write_res = len(batch_lines), [], []
        for item_idx in range(batch_size):
            cur_offset = batch_offsets[item_idx]
            for word_i in range(batch_lengths[item_idx]):
                start_i, end_i = cur_offset[word_i]
                for word_j in range(batch_lengths[item_idx]):
                    start_j, end_j = cur_offset[word_j]
                    params.append((batch_weights, item_idx, word_i, word_j, start_i, end_i, start_j, end_j))
        
        weight_res = {item[:-1]: item[-1] for item in list(pool.map(weight_sum, params))}
        # print(weight_res)
        for item_idx in range(batch_size):
            length = batch_lengths[item_idx]
            batch_write_res.append('###' + batch_lines[item_idx].strip())
            for word_i in range(length):
                res = []
                for word_j in range(length):
                    res.append(f'{word_j}:{weight_res[(item_idx, word_i, word_j)]}')
                batch_write_res.append(f"{word_i}###" + ' '.join(res))
        write_res.extend(batch_write_res)

    return write_res

def dump_self_attention_weights(model_name, corpus, batch_size, outpath, model, tokenizer):

    dataset = load_data(corpus)
    custom_dataset = CustomDataset(model_name, dataset, tokenizer)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print("Finish building custom datast!")

    pool = Pool(40)
    fout = codecs.open(outpath, 'w+', 'utf-8')
    total_weights, total_offsets, total_lines, total_lengths = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if (batch_idx+1) % 1e2 == 0:
                buffer = extract_word_word_attn_weights(pool, total_weights, total_offsets, total_lines, total_lengths)
                [fout.write(item + '\n') for item in buffer]
                fout.flush()
                total_weights, total_offsets, total_lines, total_lengths = [], [], [], []

            input_ids, masks, line_texts = batch_data['wordpiece_ids'], batch_data['wordpiece_masks'], batch_data['line_text']
            lengths, offsets = batch_data["lengths"], batch_data['offsets'] #[batch_size, max_word, 2]
            batch_size = len(lengths)

            outputs = model(input_ids=input_ids, attention_mask=masks, output_attentions=True)
            attentions = outputs.attentions # (batch_size, head_num, max_len, max_len) * layer_num
            layer_num = len(attentions)  
            layer_san_weights = torch.zeros(layer_num, batch_size, attentions[0].size(-1), attentions[0].size(-1))
            # sum all head weights
            for layer_idx in range(layer_num):
                layer_san_weights[layer_idx] = attentions[layer_idx].sum(1)
            batch_weights = layer_san_weights.sum(0).numpy()  # (layer_num, batch_size, max_len, max_len) -> (batch_size, max_len, max_len)
            total_weights.append(batch_weights)
            total_offsets.append(offsets)
            total_lines.append(line_texts)
            total_lengths.append(lengths)

    print('writing final buffer data......')
    buffer = extract_word_word_attn_weights(pool, total_weights, total_offsets, total_lines, total_lengths)
    for item in buffer: fout.write(item + '\n')
    
    sys.stdout.flush()
    pool.close()
    fout.close()


def write_table_to_file(table, path):
    print('writing table to:%s' % path)
    fout = codecs.open(path, mode='w+', encoding='utf-8')
    for k, v in table.items():
        fout.write("%s\t%s\t%.8f\n" % (k[0], k[1], v))
    fout.close()


def cal_san_word_coo(dump_file, coo_path, window_size, use_divide, use_reciprocal):
    if not os.path.exists(dump_file):
        print('dump file does not exit: ', dump_file)
        exit()

    start = time.time()
    bigram_table, line_num = {}, 0
    for line in codecs.open(dump_file, 'r', 'utf-8'):
        if line.startswith('###'):
            line_num += 1
            line_words = line[3:].strip().split()
            if line_num % 1e5 == 0:
                print('%.2fs processing %d line text.' % (time.time() - start, line_num))
                sys.stdout.flush()
        else:
            parts = line.strip().split('###')
            if len(parts) < 2:
                print('wrong line: ', line)
                print(f'###{line_words}')
                continue

            word_position = int(parts[0])
            target_token = line_words[word_position]
            parts = parts[1].split()
            sen_len = len(parts)
            if sen_len != len(line_words):
                print('length not equal!')
                print('parts:', parts)
                print(f'###{line_words}')
                continue
            
            left = 0 if (word_position - window_size) < 0 else (word_position - window_size)
            right = sen_len if (word_position + window_size + 1) > sen_len else (
                        word_position + window_size + 1)
            context = parts[left: word_position] + parts[word_position + 1: right]
            context_scores = np.array([float(item.rsplit(':', maxsplit=1)[1]) for item in context])
            context_words = [item.rsplit(':', maxsplit=1)[0] for item in context]
            context_words = [line_words[int(item)] for item in context_words]

            top_scores_idx = context_scores.argsort()[::-1][: window_size + 1]
            top_scores = context_scores[top_scores_idx]
            top_tokens = [context_words[idx] for idx in top_scores_idx]
            if len(top_scores) < 1:
                print("wrong line:", line_words)
                print("wrong weight:", line)
                continue
            
            bench = top_scores[0]  # use first predict token score as benchmark

            if use_reciprocal:
                scores = np.reciprocal(np.linspace(1, len(top_tokens), num=len(top_scores)))

            elif use_divide:
                scores = [item / bench for item in top_scores]

            for index, pair in enumerate(zip([target_token] * len(top_tokens), top_tokens)):
                if pair in bigram_table and scores[index] > 1e-9:
                    bigram_table[pair] += scores[index]
                else:
                    bigram_table[pair] = scores[index]

    write_table_to_file(bigram_table, coo_path)


def init_model(model_name, bert_path):
    model_class, masked_model_class,  config_class, tokenizer_class,  path = MODELS[model_name]
    print('Model path:', path)
    config = config_class.load(path)
    tokenizer = tokenizer_class.load(path)
    model = model_class.load(path)
    masked_model = masked_model_class.load(path)
    model.set_train(False)
    masked_model.set_train(False)
    print("Model type:", type(model))
    print('Finish loading pre-trained model.')
    return model, masked_model, tokenizer

def self_attention_sem_glove(model_name, corpus_name, corpus_path, dump_path, coo_path, batch_size, model, tokenizer, vocab_path,
                             window_size, use_divide, use_reciprocal):
    word_dump_path = os.path.join(dump_path, f'{model_name}.{corpus_name}.word.san.dump')
    word_coo_path = os.path.join(coo_path, f'{model_name}.window{window_size}.{corpus_name}.word.san.coo')
    print('Corpus file:', corpus_path)
    print('Word dump path:', word_dump_path)
    print('Word coo path:', word_coo_path)
    #dump_self_attention_weights(model_name, corpus_path, batch_size, word_dump_path, model, tokenizer)
    # cal_san_word_coo(word_dump_path, word_coo_path, window_size, use_divide, use_reciprocal)
    # merge_coo_matrix(coo_path)
    word_coo_path = "/home/ganleilei/data/BertGloVe/wiki/cooccur/window10/roberta_large_san_word_coo_divide_window10_all_merge.txt"
    convert_txt_to_bin(vocab_path, word_coo_path, word_coo_path + '.bin')
    

def mlm_sem_glove(corpus_name, corpus_path, dump_path, coo_path, batch_size, model, tokenizer, window_size, reciprocal, 
                  divide, vocab_path, wordpairpath):
    if reciprocal:
        bpe_dump_path = os.path.join(dump_path, "mlm.bpe.dump.%s.windowsize%d.reciprocal.txt" % (corpus_name, window_size)) # mlm.bpe.coo.xaa.windowsize10.reciprocal.txt
        bpe_coo_path = os.path.join(coo_path, "mlm.bpe.coo.%s.windowsize%d.reciprocal.txt" % (corpus_name, window_size)) # mlm.bpe.coo.xaa.windowsize10.reciprocal.txt
        word_coo_path = os.path.join(coo_path, "mlm.word.coo.%s.windowsize%d.reciprocal.txt" % (corpus_name, window_size)) # mlm.word.coo.xaa.windowsize10.reciprocal.txt
    elif divide:
        bpe_dump_path = os.path.join(dump_path, "mlm.bpe.dump.%s.windowsize%d.divide.txt" % (corpus_name, window_size)) # mlm.bpe.coo.xaa.windowsize10.reciprocal.txt
        bpe_coo_path = os.path.join(coo_path, "mlm.bpe.coo.%s.windowsize%d.divide.txt" % (corpus_name, window_size))  # mlm.bpe.xaa.coo.windowsize10.reciprocal.txt
        word_coo_path = os.path.join(coo_path, "mlm.word.coo.%s.windowsize%d.divide.txt" % (corpus_name, window_size))  # mlm.word.xaa.coo.windowsize10.reciprocal.txt
    else:
        raise ValueError('Please specific reweight method!')

    print('corpus file path:', corpus_path)
    print('bpe dump path:', bpe_dump_path)
    print('bpe coo path:', bpe_coo_path)
    print('word coo path:', word_coo_path)
    dump_mlm_predictions(corpus_path, bpe_dump_path, batch_size, model, tokenizer, window_size)
    # get_mlm_bpe_cooccurr_from_dump_file(window_size, divide, reciprocal, bpe_dump_path, bpe_coo_path)
    # cal_word_pair_count_from_bpe_pair_count(wordpairpath, bpe_coo_path, word_coo_path, 1, vocab_path, tokenizer)
    # convert_txt_to_bin(vocab_path, word_coo_path, word_coo_path + '.bin')


if __name__=='__main__':

    print(datetime.datetime.now())
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="SemGloVe: Semantic Co-occurrences for GloVe from BERT")
    parser.add_argument('--corpus_name', default='xaa')
    parser.add_argument('--corpus_path', default='/home/ganleilei/data/BertGloVe/wiki/')
    parser.add_argument('--model_name', default='bert_large')
    parser.add_argument('--bert_path', default='/home/ganleilei/data/bert')
    parser.add_argument('--save_path', default='/home/ganleilei/data/BertGloVe/wiki/')
    parser.add_argument('--vocab', default='data/vocab/vocab.wiki.word.txt')
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--benchposition', default=0, type=int)
    parser.add_argument('--divide', action='store_true')
    parser.add_argument('--reciprocal', action='store_true')
    parser.add_argument('--word_pair_path', default='')
    parser.add_argument('--san_glove', action='store_true')
    parser.add_argument('--mlm_glove', action='store_true')
    parser.add_argument('--txt2bin', action='store_true')

    args = parser.parse_args()

    corpus_name = args.corpus_name
    corpus_path = args.corpus_path
    model_name = args.model_name
    bert_path = args.bert_path
    save_path = args.save_path

    window_size = args.window_size
    
    corpus_path = os.path.join(corpus_path, corpus_name)
    batch_size = args.batch_size
    vocab_path = args.vocab
    use_divide = args.divide
    use_reciprocal = args.reciprocal
    word_pair_path = args.word_pair_path
    san_glove = args.san_glove
    mlm_glove = args.mlm_glove

    sys.stdout.flush()

    print('model name:', model_name)
    model, masked_model, tokenizer = init_model(model_name, bert_path)
    if args.txt2bin:
        convert_txt_to_bin(vocab_path, word_pair_path, os.path.join(word_pair_path, '.bin'))
    elif san_glove:
        print('-' * 50 + 'SAN GLOVE' + '-' * 50)
        dump_path = os.path.join(save_path, model_name, 'san', 'dump_weights')
        coo_path = os.path.join(save_path, model_name, 'san', 'cooccur', f'window{window_size}')
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        if not os.path.exists(coo_path):
            os.makedirs(coo_path)
        self_attention_sem_glove(model_name, corpus_name, corpus_path, dump_path, coo_path,
                                 batch_size, model, tokenizer, vocab_path, window_size, use_divide, use_reciprocal)
    elif mlm_glove:
        print('-' * 50 + 'MLM GLOVE' + '-' * 50)
        dump_path = os.path.join(save_path, model_name, 'mlm', 'dump_weights')
        coo_path = os.path.join(save_path, model_name, 'mlm', 'cooccur', f'window{window_size}')
        print(f"dump path:{dump_path}")
        print(f"coo path:{coo_path}")
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        if not os.path.exists(coo_path):
            os.makedirs(coo_path)
        
        mlm_sem_glove(corpus_name, corpus_path, dump_path, coo_path, batch_size, masked_model,
                      tokenizer, window_size, use_reciprocal, use_divide, vocab_path, word_pair_path)
