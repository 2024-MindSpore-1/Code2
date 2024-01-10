from .. import batch_miner
from mindspore import nn, Tensor, ops, Parameter, dtype
from . import register_criteria
import math

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batch_miner._dict.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True


@register_criteria
# MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Margin(nn.Cell):
    def __init__(self, opt, batch_miner, **kwargs):
        super(Margin, self).__init__()
        self.n_classes          = opt.n_classes

        self.margin             = opt.loss_margin_margin
        self.nu                 = opt.loss_margin_nu
        self.beta_constant      = opt.loss_margin_beta_constant
        self.beta_val           = opt.loss_margin_beta

        self.stable = opt.stable

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = Parameter(ops.ones(opt.n_classes)*opt.loss_margin_beta)

        self.batch_miner = batch_miner

        self.lr    = opt.loss_margin_beta_lr

        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def construct(self, batch, labels, **kwargs):
        sampled_triplets = self.batch_miner(batch, labels)

        if len(sampled_triplets):
            d_ap, d_an = [], []
            for triplet in sampled_triplets:

                train_triplet = {'Anchor': batch[triplet[0], :], 'Positive': batch[triplet[1], :], 'Negative': batch[triplet[2], :]}

                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = ops.stack(d_ap), ops.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = ops.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(dtype.float32)  # .to(d_ap.device)

            pos_loss = ops.relu(d_ap-beta+self.margin)
            if self.stable:
                pos_loss = math.sqrt(2) * ops.log(1 + pos_loss)
            neg_loss = ops.relu(beta-d_an+self.margin)

            pair_count = ops.sum((pos_loss > 0.).to(dtype.float32) + (neg_loss > 0.).to(dtype.float32)) # .to(d_ap.device)

            if pair_count == 0.:
                loss = ops.sum(pos_loss + neg_loss)
            else:
                loss = ops.sum(pos_loss + neg_loss) / pair_count

            if self.nu: 
                beta_regularization_loss = ops.sum(beta)
                loss += self.nu * beta_regularization_loss.to(dtype.float32)  # .to(d_ap.device)
        else:
            loss = Tensor(0.).to(dtype.float32)  # .to(batch.device)

        return loss
