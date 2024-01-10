from mindspore import Tensor, ops, nn, Parameter
from . import register_criteria


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


@register_criteria
class ProxyNca(nn.Cell):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(ProxyNca, self).__init__()

        ####
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        self.proxies            = Parameter(ops.randn(self.num_proxies, self.embed_dim)/8)
        self.class_idxs         = ops.arange(self.num_proxies)

        self.optim_dict_list = [{'params': self.proxies, 'lr': opt.lr * opt.loss_proxynca_lrmulti}]

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def forward(self, batch, labels, **kwargs):
        # Empirically, multiplying the embeddings during the computation of the loss seem to allow for more stable training;
        # Acts as a temperature in the NCA objective.
        batch   = 3 * ops.L2Normalize(batch, axis=1)
        proxies = 3 * ops.L2Normalize(self.proxies, axis=1)
        # Group required proxies
        pos_proxies = ops.stack([proxies[pos_label:pos_label+1, :] for pos_label in labels])
        neg_proxies = ops.stack([ops.cat([self.class_idxs[:class_label], self.class_idxs[class_label+1:]]) for class_label in labels])
        neg_proxies = ops.stack([proxies[neg_labels,:] for neg_labels in neg_proxies])
        # Compute Proxy-distances
        dist_to_neg_proxies = ops.sum((batch[:, None, :]-neg_proxies).pow(2), dim=-1)
        dist_to_pos_proxies = ops.sum((batch[:, None, :]-pos_proxies).pow(2), dim=-1)
        # Compute final proxy-based NCA loss
        loss = ops.mean(dist_to_pos_proxies[:, 0] + ops.logsumexp(-dist_to_neg_proxies, axis=1))

        return loss
