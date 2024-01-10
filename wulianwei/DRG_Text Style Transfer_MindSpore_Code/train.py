# 已修改部分

import sys

import json
import numpy as np
import logging
import argparse
import os
import time
import numpy as np
import glob

# import torch
# import torch.nn as nn
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
# import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
# from src.cuda import CUDA
import src.data as data
import src.models as models

# 在config 文件中，原训练 epoch 数为70

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    # required=False
    required=True
)
parser.add_argument(
    "--bleu",
    help="do BLEU eval",
    action='store_true'
)
parser.add_argument(
    "--overfit",
    help="train continuously on one batch of data",
    action='store_true'
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))

working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')
src, tgt = data.read_nmt_data(
    src=config['data']['src'],
    config=config,
    tgt=config['data']['tgt'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes']
)

src_test, tgt_test = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes'],
    train_src=src,
    train_tgt=tgt
)
logging.info('...done!')


batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = len(src['tok2id'])
tgt_vocab_size = len(tgt['tok2id'])

print(tgt_vocab_size)
tgt_vocab_size = int(tgt_vocab_size)
# weight_mask = torch.ones(tgt_vocab_size)
# 修改
ones = ops.Ones()
weight_mask = ones(tgt_vocab_size, mindspore.float32)
# weight_mask = ops.Ones(tgt_vocab_size)
print("**********")
weight_mask[tgt['tok2id']['<pad>']] = 0
# loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
# 修改 2, 调试时 删除 函数内参数
loss_criterion = mindspore.nn.SoftmaxCrossEntropyWithLogits()
# if CUDA:
#     weight_mask = weight_mask.cuda()
#     loss_criterion = loss_criterion.cuda()
# 修改，删除 上面3行

# torch.manual_seed(config['training']['random_seed'])
# 待修改 ，待核对，无映射，可能为下一行代码所示（ms.set_seed 函数）
mindspore.set_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])

model = models.SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=src['tok2id']['<pad>'],
    pad_id_tgt=tgt['tok2id']['<pad>'],
    config=config,
    # 修改 2 调试
    # config=bool(config)
)

logging.info('MODEL HAS %s params' %  model.count_params())
model, start_epoch = models.attempt_load_model(
    model=model,
    checkpoint_dir=working_dir)
# start_epoch = 0 # 新增的代码，删除 上面的函数得到
# if CUDA:
#     model = model.cuda()
# 修改，删除 上面2行

writer = SummaryWriter(working_dir)


if config['training']['optimizer'] == 'adam':
    lr = config['training']['learning_rate']
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # 修改
    optimizer = mindspore.nn.Adam(model.get_parameters(), learning_rate=lr)
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['learning_rate']
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # 修改
    optimizer = mindspore.nn.SGD(model.get_parameters(), learning_rate=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

epoch_loss = []
start_since_last_report = time.time()
words_since_last_report = 0
losses_since_last_report = []
best_metric = 0.0
best_epoch = 0
cur_metric = 0.0 # log perplexity or BLEU
num_examples = min(len(src['content']), len(tgt['content']))
num_batches = num_examples / batch_size

STEP = 0
for epoch in range(start_epoch, config['training']['epochs']):
    if cur_metric > best_metric:
        # rm old checkpoint
        for ckpt_path in glob.glob(working_dir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        # replace with new checkpoint
        # torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)
        # 修改
        mindspore.save_checkpoint(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)

        best_metric = cur_metric
        best_epoch = epoch - 1

    losses = []
    for i in range(0, num_examples, batch_size):

        if args.overfit:
            i = 50

        batch_idx = i / batch_size

        input_content, input_aux, output = data.minibatch(
            src, tgt, i, batch_size, max_length, config['model']['model_type'])
        input_lines_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output

        # decoder_logit, decoder_probs = model(input_lines_src, input_lines_tgt, srcmask, srclens, input_ids_aux, auxlens, auxmask)
        # 修改
        decoder_logit, decoder_probs = model.construct(input_lines_src, input_lines_tgt, srcmask, srclens, input_ids_aux, auxlens, auxmask)



        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, tgt_vocab_size),
            output_lines_tgt.view(-1)
        )

        losses.append(loss.item())
        losses_since_last_report.append(loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        # norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])
        norm = nn.ClipByNorm_(model.get_parameters(), config['training']['max_norm'])
        # nn.ClipByNorm
        # 可能对应该算子，核对 mindspore.nn.ClipByNorm
        # clip_by_global_norm
        # https://gitee.com/mindspore/mindspore/issues/I59419?from=project-issue

        writer.add_scalar('stats/grad_norm', norm, STEP)

        optimizer.step()

        if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - start_since_last_report)
            eps = (batch_size * config['training']['batches_per_report']) / s
            avg_loss = np.mean(losses_since_last_report)
            info = (epoch, batch_idx, num_batches, eps, avg_loss, cur_metric)
            writer.add_scalar('stats/EPS', eps, STEP)
            writer.add_scalar('stats/loss', avg_loss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s EPS: %.2f LOSS: %.4f METRIC: %.4f' % info)
            start_since_last_report = time.time()
            words_since_last_report = 0
            losses_since_last_report = []

        # NO SAMPLING!! because weird train-vs-test data stuff would be a pain
        STEP += 1
    if args.overfit:
        continue

    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)
    start = time.time()
    model.eval()
    dev_loss = evaluation.evaluate_lpp(
            model, src_test, tgt_test, config)

    writer.add_scalar('eval/loss', dev_loss, epoch)

    if args.bleu and epoch >= config['training'].get('inference_start_epoch', 1):
        cur_metric, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
            model, src_test, tgt_test, config)

        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        writer.add_scalar('eval/edit_distance', edit_distance, epoch)
        writer.add_scalar('eval/bleu', cur_metric, epoch)

    else:
        cur_metric = dev_loss

    model.train()

    logging.info('METRIC: %s. TIME: %.2fs CHECKPOINTING...' % (
        cur_metric, (time.time() - start)))
    avg_loss = np.mean(epoch_loss)
    epoch_loss = []

writer.close()

