import os
import sys
from utils_onehot import create_scatter, optimizer_step
from models_onehot import Decoder, Nu, Encoder

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.train import Model
from mindspore.nn import optim
import mindspore.ops as ops
from mindspore import context
from mindspore.common import set_seed

from itertools import chain
import argparse
import logging
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser()

# global parameters
parser.add_argument('--results_folder', default='results')
parser.add_argument('--train_from', default='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--log_prefix', default='')

# global training parameters
parser.add_argument('--num_epochs', default=80000, type=int)
parser.add_argument('--num_particles', default=64, type=int)
parser.add_argument('--test_num_particles', default=500, type=int)
parser.add_argument('--num_nu_updates', default=5, type=int)  # converge nu faster and better to avoid NA

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")
parser.add_argument('--plot_freq', type=int, default=5000)

# model and optimizer parameters
parser.add_argument('--nu_lr', default=1e-5, type=float)
parser.add_argument('--end2end_lr', default=1e-5, type=float)

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd
print(args)
#sys.exit(0)
# training starts
set_seed(args.seed)
if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)
logging.basicConfig(filename=os.path.join(args.results_folder, args.log_prefix + 'eval.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')

# Device selection
if context.get_context("device_target") != "GPU":
    args.no_gpu = True

context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if not args.no_gpu else "CPU")

gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

# Create tensors
x_test = Tensor(np.zeros((4 * args.test_num_particles, 4), dtype=np.float32)) #(2000, 4)
#print("[tlog] " + str(x_test) + "\t" + str(x_test.shape) )

# Note: MindSpore does not support LongTensor explicitly. You'll use Int32 instead.
idx_test = Tensor([j for i in range(4) for j in [i] * args.test_num_particles], ms.int32)
#print("[tlog] " + str(idx_test.shape) + "\t" + str(idx_test)) #(2000, 1)

# Setting up operations
equal = ops.Equal()
greater = ops.Greater()
reduce_mean = ops.ReduceMean()
reduce_sum = ops.ReduceSum()
log = ops.Log()
shape = ops.Shape()

def ones_like(tensor):
    return ms.Tensor(np.ones(tensor.shape, dtype=np.float32))

a = idx_test.view(-1, 1).repeat(4, axis=1)
b = ones_like(a)
x_test = ops.scatter(input=x_test, axis=1, index=a, src=b)
#print("[tlog] a: ", a)
#print("[tlog] x_test" + str(x_test))

# Generate random tensor
from mindspore.nn.probability.distribution import Normal

mean = Tensor([0.0], dtype=ms.float32)
stddev = Tensor([1.0], dtype=ms.float32)
normal_distr = Normal(mean, stddev)
eps_test_shape = (4 * args.test_num_particles, 2)
eps_test = normal_distr.sample((4 * args.test_num_particles, 2))
eps_test = eps_test.view(eps_test_shape)
#eps_test = Tensor(normal(eps_test_shape, mindspore.float32))
print(args.train_from)
if args.train_from == "":
    decoder = Decoder()
    nu = Nu()
    encoder = Encoder()
    nu_optimizer = optim.Adam(nu.trainable_params(), learning_rate=args.nu_lr)
    end2end_optimizer = optim.Adam(decoder.trainable_params() + encoder.trainable_params(), learning_rate=args.end2end_lr)
else:
    logging.info('load model from' + args.train_from)
    checkpoint = mindspore.load_checkpoint(args.train_from)
    decoder = Decoder()
    nu = Nu()
    encoder = Encoder()
    mindspore.load_param_into_net(nu, checkpoint)
    mindspore.load_param_into_net(encoder, checkpoint)
    mindspore.load_param_into_net(decoder, checkpoint)
    args = checkpoint['args']
    logging.info(str(args))
    nu = checkpoint['nu']
    nu_optimizer = checkpoint['nu_optimizer']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    end2end_optimizer = checkpoint['end2end_optimizer']
#sys.exit(0)

def evaluation(x, eps, epo=None):
    z_x = encoder(x, eps)

    z_plt = ops.stop_gradient(z_x).asnumpy()
    if epo is not None:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, '%06d.png' % epo))
    else:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, 'evaluation.png'))

    pi = decoder(z_x)
    
    
    # Convert the lines
    pi_pos_indices = equal(pi, 1)
    pi_neg_indices = equal(pi, 0)
    rec = -reduce_mean(log(reduce_mean(pi * pi_pos_indices))) - reduce_sum(log(1.0 - pi * pi_neg_indices)) / shape(x)[0]
    logging.info('rec: %.4f' % rec)
    z = normal_distr.sample(shape(eps)).view(eps.shape)
    kl = reduce_mean(nu(x, z_x)) - reduce_mean(ops.Exp()(nu(x, z))) + 1.0
    logging.info('kl with nu: %.4f' % kl)
    logging.info('neg_ELBO with nu: %.4f' % (rec + kl))


def check_point(epo=None):
    check_pt = {
        'args': args,
        'nu': nu,
        'nu_optimizer': nu_optimizer,
        'encoder': encoder,
        'decoder': decoder,
        'end2end_optimizer': end2end_optimizer
    }
    '''
    if epo is not None:
        save_checkpoint(check_pt, os.path.join(args.results_folder, '%06d.pt' % epo))
    else:
        save_checkpoint(check_pt, os.path.join(args.results_folder, 'checkpoint.pt'))
    '''

print(args.test)
if args.test:
    evaluation(x_test, eps_test)
    exit()

class Similarity(nn.Cell):
    """
        Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = ops.cosine_similarity

    def construct(self, x, y):
        return self.cos(x, y) / self.temp

pbar = tqdm(range(args.num_epochs + 1))

sim_func = Similarity(0.05)
self_sup_loss_fct = nn.CrossEntropyLoss()

def kl_approx_loss(a, b): 
    return ops.mean(ops.Exp()(a) - b)

def kl_approx_loss_logexp(a,b): 
    zero = ms.Tensor([0])
    cat = ops.Concat(axis=-1)
    new_a = cat([zero, ops.Exp()(a).view(-1)])
    new_b = cat([zero, -b.view(-1)])
    return ops.Logsumexp()(new_a, axis=-1) + ops.Logsumexp()(new_b, axis=-1)
    
for epo in pbar:

    if epo % args.plot_freq == 0:
        logging.info('------------------------------------------------------')
        logging.info("the current epo is %d" % epo)
        evaluation(x_test, eps_test, epo=epo)
        check_point(epo)

    x = Tensor(np.zeros((4 * args.num_particles, 4), dtype=np.float32)) #(2000, 4)

    rand_N = ops.randint(0, 4, (4,))
    rand = [j for i in rand_N.asnumpy() for j in [i] * args.num_particles]
    idx = Tensor(rand).view(-1, 1).repeat(4, axis=1)
    
    x = ops.scatter(input=x, axis=1, index=idx, src=ones_like(idx))

    eps = Tensor(np.random.normal(0, 1, (4 * args.num_particles, 2)), ms.float32) #(2000, 2)
    z_x = encoder(x, eps)
    
    eps2 = Tensor(np.random.normal(0, 1, (4 * args.num_particles, 2)), ms.float32) #(2000, 2)
    z_x2 = encoder(x, eps2)
    
    cos_sim = sim_func(z_x.unsqueeze(1), z_x2.unsqueeze(0))
    '''
    batch_size, feat_size = z_x.size()
    rand_vec = torch.normal(0, 1, size=(3*batch_size, feat_size)).to(z_x.device)
    reg_cos_sim = sim_func(z_x.unsqueeze(1), rand_vec.unsqueeze(0)) 
    cos_sim = torch.cat((cos_sim, reg_cos_sim), dim=1)
    '''
    labels = Tensor([i for i in range(cos_sim.shape[0])], ms.int32)
    self_sup_loss = self_sup_loss_fct(cos_sim, labels)

    # nu update
    for k in range(args.num_nu_updates):
        def train_fn(x, z_x, z_x2): 
            z_x_nu = ops.stop_gradient(z_x)
            z = Tensor(np.random.normal(0, 1, (4 * args.num_particles, 2)), ms.float32)
            
            nu_loss = kl_approx_loss_logexp(nu(x, z),  nu(x, z_x_nu)) ###tzy: train nu only
            
            z_x_nu2 = ops.stop_gradient(z_x2)
            z2 = Tensor(np.random.normal(0, 1, (4 * args.num_particles, 2)), ms.float32)
            
            nu_loss += kl_approx_loss_logexp(nu(x, z2), nu(x, z_x_nu2)) ###tzy: train nu only
            return nu_loss

        grad_fn = ms.value_and_grad(train_fn, None, nu_optimizer.parameters, has_aux=False)
        nu_loss, grads = grad_fn(x, z_x, z_x2)
        nu_optimizer(grads)

    # end2end update
    pi = decoder(z_x)
    pi2 = decoder(z_x2)
    '''
    print("[tlog] pi: " + str(pi.size()))
    print("[tlog] pi: " + str(pi))
    print("[tlog] x==1: " + str(x==1))
    print("[tlog] pi1: " + str(pi[x==1].size()))
    print("[tlog] pi: " + str(pi[x==1]))
    print("[tlog] pi0: " + str(pi[x==0].size()))
    print("[tlog] pi: " + str(pi[x==0]))
    '''
    #sys.exit(0)
    pi_pos_indices = equal(pi, 1)
    pi_neg_indices = equal(pi, 0)
    rec = -reduce_mean(log(reduce_mean(pi * pi_pos_indices))) - reduce_sum(log(1.0 - pi * pi_neg_indices)) / shape(x)[0]
    pi2_pos_indices = equal(pi2, 1)
    pi2_neg_indices = equal(pi2, 0)
    rec2 = -reduce_mean(log(reduce_mean(pi2 * pi2_pos_indices))) - reduce_sum(log(1.0 - pi2 * pi2_neg_indices)) / shape(x)[0]
    rec = rec + rec2
    
    loss = rec + ops.mean(nu(x, z_x)) + ops.mean(nu(x, z_x2))
    loss += self_sup_loss
    
    def e2e_train_fn(x, z_x):
        return loss
    e2e_grad_fn = ms.value_and_grad(e2e_train_fn, None, end2end_optimizer.parameters, has_aux=False)
    loss, grads = e2e_grad_fn(x, z_x)
    end2end_optimizer(grads)
