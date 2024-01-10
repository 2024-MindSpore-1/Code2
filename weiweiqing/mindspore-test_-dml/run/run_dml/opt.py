import os


# === training parameters

multi_loss = True
m_loss = "Softmax"
fuse = 0.1
stable = True

# --- Dataset-related Parameters
dataset = "Food"  # Cars196, Cub200, StanfordOnlineProducts
use_tv_split = False
tv_split_by_samples = True
tv_split_perc = 0.8
augmentation = "base"  # big, adv, red

# --- General Training Parameters
lr = 1e-5
fc_lr = -1
decay = 1e-4
n_epochs = 150  # --------------------
kernels = 4   # number of workers for dataloader
bs = 112  # batch size  # ---------------------
seed = 1
scheduler = "step"  # learning rate scheduler
tau = [1000]  # step size before reducing learning rate
gamma = [0.3]  # learning rate reduction after tau epochs/steps


# --- loss specific settings
optim = "adam"
loss = "Margin"  # "Contrastive"  # "Margin"  # ------------------------
batch_mining = "RhoDistance"  # "Distance"


# ---- network-related flags
embed_dim = 128
not_pretrained = False
arch = "Vit"   # "Resnet50"


# --- evaluation parameters
no_train_metric = True
evaluate_on_gpu = False  # --------------------------
# evaluation_metrics = [
# 	"E_Recall@1", "E_Recall@2", "E_Recall@4",
# 	"NMI", "F1", "MAP_1000", "MAP_Lim", "MAP_C",
# 	"Dists@intra", "Dists@inter", "Dists@intra_over_inter",
# 	"Rho_Spectrum@0", "Rho_Spectrum@-1", "Rho_Spectrum@2", "Rho_Spectrum@10"
# ]
evaluation_metrics = [
	"E_Recall@1", "E_Recall@2", "E_Recall@4",
	# "NMI", "F1", "MAP_1000", "MAP_C",
	# "Dists@intra", "Dists@inter", "Dists@intra_over_inter",
	# "Rho_Spectrum@0", "Rho_Spectrum@-1", "Rho_Spectrum@2", "Rho_Spectrum@10"
]
storage_metrics = ['E_Recall@1']
evaltypes = ['discriminative']


# --- Setup Parameters
gpu = [0]
save_name = "group_plus_seed"  # using by log online
# source_path = r"/home/yang/food172"  # os.getcwd()+'/../../Datasets'  # -----------------------
source_path = r"/home/yang/food172"
save_path = os.getcwd()+'/Training_Results'


# === wandb parameters

# --- online logging/Wandb Log arguments
log_online = False
wandb_key = "<your_api_key_here>"
project = "Sample_Project"
group = "Sample_Group"


# === loss specific parameters

# --- Contrastive loss
loss_contrastive_pos_margin = 0
loss_contrastive_neg_margin = 1

# --- Triplet-based losses
loss_triplet_margin = 0.2

# --- Margin loss
loss_margin_margin = 0.2
loss_margin_beta_lr = 0.0005
loss_margin_beta = 0.6  # 1.2  ------------------------
loss_margin_nu = 0
loss_margin_beta_constant = False

# --- ProxyNCA, NOTE: The number of proxies is determined by the number of data classes
loss_proxynca_lrmulti = 50

# --- NPair
loss_npair_l2 = 0.005

# --- Angular loss
loss_angular_alpha = 45
loss_angular_npair_ang_weight = 2
loss_angular_npair_l2 = 0.005

# --- Multisimilary loss
loss_multisimilarity_pos_weight = 2
loss_multisimilarity_neg_weight = 40
loss_multisimilarity_margin = 0.1
loss_multisimilarity_thres = 0.5

# --- Lifted Structure loss
loss_lifted_neg_margin = 1
loss_lifted_l2 = 0.005

# --- Quadruplet loss
loss_quadruplet_margin_alpha_1 = 0.2
loss_quadruplet_margin_alpha_2 = 0.2

# --- Soft-Triple loss
loss_softtriplet_n_centroids = 2
loss_softtriplet_margin_delta = 0.01
loss_softtriplet_gamma = 0.1
loss_softtriplet_lambda = 8
loss_softtriplet_reg_weight = 0.2
loss_softtriplet_lrmulti = 1

# --- Normalized Softmax Loss
loss_softmax_lr = 0.00001
loss_softmax_temperature = 0.05

# --- Histogram Loss
loss_histogram_nbins = 65

# --- SNR Triplet (with learnable margin) Loss
loss_snr_margin = 0.2
loss_snr_reg_lambda = 0.005

# --- ArcFace
loss_arcface_lr = 0.005
loss_arcface_angular_margin = 0.5
loss_arcface_feature_scale = 16


# === batch mining specific parameters
miner_distance_lower_cutoff = 0.5
miner_distance_upper_cutoff = 1.4
# Spectrum-Regularized Miner (as proposed in our paper) - utilizes a distance-based sampler that is regularized.
miner_rho_distance_lower_cutoff = 0.5
miner_rho_distance_upper_cutoff = 1.4
miner_rho_distance_cp = 0.4  # 0.2  # ----------------------

# === batch creation parameters
data_sampler = "ClassRandomSampler"
samples_per_class = 2
# Batch-Sample Flags - Have no relevance to default SPC-N sampling
data_batchmatch_bigbs = 512
data_batchmatch_ncomps = 10
data_storage_no_update = False
data_d2_coreset_lambda = 1
data_gc_coreset_lim = 1e-9
data_sampler_lowproj_dim = -1
data_sim_measure = "euclidean"
data_gc_softened = False
data_idx_full_prec = False
data_mb_mom = -1
data_mb_lr = 1
