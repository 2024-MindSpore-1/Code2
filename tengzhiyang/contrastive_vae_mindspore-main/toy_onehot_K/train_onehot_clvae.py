
import os
import sys
from utils_onehot import create_scatter, optimizer_step
from models_onehot import Decoder, Nu, Encoder
import torch
from itertools import chain
from torch import optim
import argparse
from torch.autograd import grad
import logging
from tqdm import tqdm
import torch.nn as nn

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
parser.add_argument('--num_class', default=4, type=float)



if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])   # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd
print(args)
#sys.exit(0)
# training starts
num_class = int(args.num_class)
torch.manual_seed(args.seed)
if not os.path.exists(args.results_folder):
    os.makedirs(args.results_folder)
logging.basicConfig(filename=os.path.join(args.results_folder, args.log_prefix + 'eval.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')
if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

x_test = torch.zeros((num_class * args.test_num_particles, num_class), device='cuda' if gpu else 'cpu') # 4*64, 4
print("[tlog] " + str(x_test))
idx_test = torch.LongTensor([j for i in range(num_class) for j in [i] * args.test_num_particles]).to(x_test.device)
print("[tlog] " + str(idx_test))
x_test.scatter_(1, idx_test.view(-1, 1), 1) # one hot??
print("[tlog] " + str(x_test))
eps_test = torch.zeros((num_class * args.test_num_particles, 2), device=x_test.device).normal_(0, 1) # why 2??
print("[tlog] " + str(eps_test))
#sys.exit(0)
print(args.train_from)
if args.train_from == "":
    decoder = Decoder(num_class=num_class)
    nu = Nu(num_class=num_class)
    encoder = Encoder(num_class=num_class)
    if gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        nu = nu.cuda()
    nu_optimizer = optim.Adam(nu.parameters(), lr=args.nu_lr) ### tzy: different optimizier
    end2end_optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.end2end_lr)
else:
    print("haha")
    logging.info('load model from' + args.train_from)
    checkpoint = torch.load(args.train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')
    is_test = args.test
    args = checkpoint['args']
    logging.info(str(args))
    nu = checkpoint['nu']
    nu_optimizer = checkpoint['nu_optimizer']
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    end2end_optimizer = checkpoint['end2end_optimizer']
    args.test = is_test
#sys.exit(0)

def evaluation(x, eps, epo=None):
    z_x = encoder(x, eps)

    z_plt = z_x.detach().cpu().numpy()
    if epo is not None:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, '%06d.png' % epo), num_class=num_class)
    else:
        create_scatter(z_plt, save_path=os.path.join(args.results_folder, 'evaluation.png'), num_class=num_class)

    pi = decoder(z_x)
    rec = -torch.mean(torch.log(pi[x == 1])) - torch.sum(torch.log(1 - pi[x == 0])) / x.shape[0]
    logging.info('rec: %.4f' % rec)
    z = torch.randn_like(eps)
    kl = torch.mean(nu(x, z_x)) - torch.mean(torch.exp(nu(x, z))) + 1.0
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
    if epo is not None:
        torch.save(check_pt, os.path.join(args.results_folder, '%06d.pt' % epo))
    else:
        torch.save(check_pt, os.path.join(args.results_folder, 'checkpoint.pt'))

print(args.test)
if args.test:
    evaluation(x_test, eps_test)
    exit()

class Similarity(nn.Module):
    """
        Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

pbar = tqdm(range(args.num_epochs + 1))

sim_func = Similarity(0.05)
self_sup_loss_fct = nn.CrossEntropyLoss()

def kl_approx_loss(a, b): 
    return torch.mean(torch.exp(a) - b)

def kl_approx_loss_logexp(a,b): 
    zero = torch.tensor([0.0], device=a.device)
    new_a = torch.cat([zero, torch.exp(a).view(-1)], dim=-1)
    new_b = torch.cat([zero, -b.view(-1)], dim=-1)
    return torch.logsumexp(new_a, dim=-1) + torch.logsumexp(new_b, dim=-1)
    
for epo in pbar:

    if epo % args.plot_freq == 0:
        logging.info('------------------------------------------------------')
        logging.info("the current epo is %d" % epo)
        evaluation(x_test, eps_test, epo=epo)
        check_point(epo)

    x = torch.zeros((num_class * args.num_particles, num_class), device='cuda' if gpu else 'cpu')
    rand_N = torch.LongTensor(num_class).random_(0, num_class).tolist()
    rand = [j for i in rand_N for j in [i] * args.num_particles]
    idx = torch.LongTensor(rand).to(x.device)
    x.scatter_(1, idx.view(-1, 1), 1.0)
    eps = torch.zeros((num_class * args.num_particles, 2), device=x.device).normal_(0, 1)
    z_x = encoder(x, eps)
    
    eps2 = torch.zeros((num_class * args.num_particles, 2), device=x.device).normal_(0, 1)
    z_x2 = encoder(x, eps2)
    
    cos_sim = sim_func(z_x.unsqueeze(1), z_x2.unsqueeze(0))
    '''
    batch_size, feat_size = z_x.size()
    rand_vec = torch.normal(0, 1, size=(3*batch_size, feat_size)).to(z_x.device)
    reg_cos_sim = sim_func(z_x.unsqueeze(1), rand_vec.unsqueeze(0)) 
    cos_sim = torch.cat((cos_sim, reg_cos_sim), dim=1)
    '''
    #print(cos_sim.size())
    #print(cos_sim)
    labels = torch.arange(cos_sim.size(0)).long().to(z_x.device)
    self_sup_loss = self_sup_loss_fct(cos_sim, labels)

    # nu update
    for k in torch.arange(args.num_nu_updates):
        z_x_nu = z_x.detach() ###tzy: note there is a detach here, and there are several epochs, andfor each k, the z_x is the same
        z = torch.zeros((num_class * args.num_particles, 2), device=x.device).normal_(0, 1)
        
        nu_loss = kl_approx_loss_logexp(nu(x, z),  nu(x, z_x_nu)) ###tzy: train nu only
        
        z_x_nu2 = z_x2.detach() ###tzy: note there is a detach here, and there are several epochs, andfor each k, the z_x is the same
        z2 = torch.zeros((num_class * args.num_particles, 2), device=x.device).normal_(0, 1)
        
        nu_loss += kl_approx_loss_logexp(nu(x, z2), nu(x, z_x_nu2)) ###tzy: train nu only
        
        optimizer_step(nu_loss, nu_optimizer)

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
    rec = - torch.mean(torch.log(pi[x == 1])) - torch.sum(torch.log(1 - pi[x == 0])) / x.shape[0]
    rec2 = - torch.mean(torch.log(pi2[x == 1])) - torch.sum(torch.log(1 - pi2[x == 0])) / x.shape[0]
    rec = rec + rec2
    
    #loss = rec + torch.mean(nu(x, z_x))
    loss = rec + torch.mean(nu(x, z_x)) + torch.mean(nu(x, z_x2))
    loss += self_sup_loss
    
    optimizer_step(loss, end2end_optimizer) #end2end parameters does not contain any parameter of nu

    torch.cuda.empty_cache()
