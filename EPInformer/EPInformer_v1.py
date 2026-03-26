import os

import sys
import argparse

# from datetime import datetime
import os
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np

# from EPInformer.models_v2 import EPInformer_v2, enhancer_predictor_256bp
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import Subset, Dataset
import h5py

# from torch import Tensor
import torch
import torch.nn as nn
# from torch.nn import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
import math
from sklearn.metrics import mean_squared_error
import copy
from sklearn.model_selection import train_test_split
import numpy as np
# import time
# import tqdm
import warnings
# import torch.nn.functional as F
# import einops
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')
# from EPInformer.multihead_diffattn import MultiheadDiffAttn
from dataclasses import dataclass


class promoter_enhancer_dataset(Dataset):
    def __init__(self, cell_type='GM12878', expr_type='RNA', n_enh_feats=3, disable_enh=False, distance_thr=None, max_n_enh=200):
        # self.data_h5 = h5py.File('/mnt/usb1/jiecong/{}_220CREs-gene_strand.hdf5'.format(cell_type), 'r') # /mnt/usb1/jiecong
        self.data_h5 = h5py.File('/dev/shm/jiecong_data/{}_200CREs-gene_RPM_4feats.hdf5'.format(cell_type), 'r')
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.expr_df = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso.csv', index_col='gene_id')
        self.cell2tok = {'K562':0, 'GM12878':1, 'HepG2':2, 'NHEK': 3, 'HUVEC': 4, 'H1': 5}
        if cell_type == 'K562':
            promoter_df = pd.read_csv('./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'GM12878':
            promoter_df = pd.read_csv('./data/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'HepG2':
            promoter_df = pd.read_csv('./data/HepG2/DNase_ENCFF691HJY_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'NHEK':
            promoter_df = pd.read_csv('./data/NHEK/DNase_ENCFF862NDZ_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'HUVEC':
            promoter_df = pd.read_csv('./data/HUVEC/DNase_ENCFF091KTX_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'H1':
            promoter_df = pd.read_csv('./data/H1/DNase_ENCFF761ZRE_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        else:
            print('Cell not found!')
        promoter_df['promoter_activity'] = np.sqrt(promoter_df['DHS.RPKM.TSS1Kb']*promoter_df['H3K27ac.RPKM.TSS1Kb'])
        self.promoter_df = promoter_df

    def __len__(self):
        return len(self.data_h5['ensid'])
    def __getitem__(self, idx):
        sample_ensid = self.data_h5['ensid'][idx].decode()
        enh_ohe = self.data_h5['enhancers_ohe'][idx]
        prm_ohe = self.data_h5['promoter_ohe'][idx][np.newaxis,:]
        enh_feats = self.data_h5['enhancers_feat'][idx][:,:]
        enh_feats = np.concatenate([enh_feats[:, [0]], enh_feats[:, [3]], enh_feats[:, [-1]]], axis=1)[:,:self.n_enh_feats]
        cell_tok = np.array([self.cell2tok[self.cell_type]])
        rna_feats = np.array(self.expr_df.loc[sample_ensid][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore',
                            'UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float)).flatten()
        if self.expr_df.loc[sample_ensid, 'is_hk']:
            hk_tok = np.array([1])
        else:
            hk_tok = np.array([0])
       #  prm_signal = np.array([self.promoter_df.loc[sample_ensid,'promoter_activity']])
        if self.distance_thr is not None: 
            enh_ohe_new = np.zeros((self.max_n_enh, 2000, 4))
            enh_feats_new = np.zeros((self.max_n_enh, enh_feats.shape[-1]))
            new_i = 0
            for i in range(enh_ohe.shape[0]):
                if abs(enh_feats[i][0])<=self.distance_thr:
                    # enh_ohe[i] = np.zeros_like(enh_ohe[i])
                    enh_ohe_new[new_i] = enh_ohe[i]
                    enh_feats_new[new_i] = enh_feats[i]
                    new_i += 1
                if new_i >= self.max_n_enh:
                    break
            enh_ohe = enh_ohe_new
            enh_feats = enh_feats_new
        if self.disable_enh:
            enh_ohe = np.zeros_like(enh_ohe)
            enh_feats = np.zeros_like(enh_feats)
        if self.expr_type == 'CAGE':
            expr = np.log(1+self.data_h5['expr'][idx][1])
        else:
            expr = self.data_h5['expr'][idx][0]
        return prm_ohe, enh_ohe, rna_feats, enh_feats, hk_tok, expr, sample_ensid


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        
class L1KLmixed(nn.Module):
    
    def __init__(self, reduction='batchmean', alpha=1.0, beta=0.5):

        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
    
    def forward(self, preds, targets):

        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size) 
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)  

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)  
        return out
        
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class seq_256bp_encoder(nn.Module):
    def __init__(self, base_size=4, out_dim=128, conv_dim=256):
        super(seq_256bp_encoder, self).__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        # cropped_len = 46
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels = base_size, out_channels = self.conv_dim, kernel_size = (1, 12), stride = 1, padding='same'),
            nn.ReLU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dim = [self.conv_dim, 128, 64, 64, 64]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i], out_channels=conv_dim[i+1], kernel_size=(1, 3), padding=(0, 1)),
                # nn.BatchNorm2d(conv_dim[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ))
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i+1], out_channels=conv_dim[i+1], kernel_size=(1, 1)),
            ))

    def forward(self, enhancers_input):
        if enhancers_input.shape[2] == 1:
            x_enhancer = enhancers_input
        else:
            x_enhancer = enhancers_input.permute(0, 3, 1, 2).contiguous()  
        x_enhancer = self.stem_conv(x_enhancer)
#         print(x_enhancer.shape)
        for i in range(0, len(self.conv_tower), 2):
            x_enhancer = self.conv_tower[i](x_enhancer)
            x_enhancer = self.conv_tower[i+1](x_enhancer) + x_enhancer
        return x_enhancer

class enhancer_predictor_256bp(nn.Module):
    def __init__(self):
        super(enhancer_predictor_256bp, self).__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128*16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )  
    def forward(self, enhancer_seq):
        if len(enhancer_seq.shape) < 4:
            enhancer_seq = enhancer_seq.unsqueeze(2)
        seq_embed = self.encoder(enhancer_seq)
        epi_out = self.embedToAct(seq_embed)
        return epi_out.squeeze(-1)

class MHAttention_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super(MHAttention_encoderLayer, self).__init__()
        # self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_attn = MultiheadDiffAttn(embed_dim=d_model, depth=12, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout),
        )
    # self-attention block
    def _sa_block(self, x, key_padding_mask=None, attn_mask=None):
        x, w = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # x, w = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
        
    def forward(self, x, enhancers_padding_mask=None, attn_mask=None):
        xt, attention_w = self._sa_block(self.norm1(x), key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
        x = xt + x
        x = x + self.ff(self.norm2(x))
        return x, attention_w

class EPInformer_v1(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, d_model=128, n_enh_feats=3, device='cuda', n_heads = 4, use_rna_feats=False, use_hk_tok=True):
        super(EPInformer_v1, self).__init__()
        # self.n_enhancers = n_enhancers
        self.d_model = d_model
        # self.n_cell = n_cell
        self.use_rna_feats = use_rna_feats
        self.use_hk_tok = use_hk_tok
        self.model_name = 'EPInformer_v1'
        # self.usePromoterSignal = usePromoterSignal
        self.n_heads = n_heads
        self.n_extraFeat = n_enh_feats
        # self.useBN = useBN
        self.base_size = base_size
        # self.useLN = useLN
        self.promoter_seq_encoder = seq_256bp_encoder(base_size=base_size)
        self.enhancer_seq_encoder = seq_256bp_encoder(base_size=base_size)
        self.p_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(38, 4)
        )
        self.e_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(113, 4)
        )
        # if pre_trained_encoder is not None:
        #     self.seq_encoder = pre_trained_encoder
        #     self.name = 'EPInformerV2.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        # else:
        #     self.seq_encoder = seq_256bp_encoder(base_size=base_size)
        #     self.name = 'EPInformerV2.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=self.d_model, nhead=self.n_heads), self.n_encoder)

        # attn_mask = (~np.identity(self.n_enhancers+1).astype(bool))
        # attn_mask[:, 0] = False
        # attn_mask[0, :] = False
        # attn_mask = torch.from_numpy(attn_mask)
        # attn_mask.masked_fill(attn_mask, float('-inf'))
        # self.attn_mask = attn_mask
        self.hk_embed = nn.Embedding(2, self.d_model)
        self.rna_embed = nn.Linear(8, self.d_model)
        # self.promter_signal_embed = nn.Linear(1, 64)
        # if self.use_rna_feats:
        #     self.pToExpr = nn.Sequential(
        #                 nn.Linear(self.d_model+8, self.d_model),
        #                 nn.ReLU(),
        #                 nn.Linear(self.d_model, 1),
        #             )
        # else:
        #     self.pToExpr = nn.Sequential(
        #             nn.Linear(self.d_model, self.d_model),
        #             nn.ReLU(),
        #             nn.Linear(self.d_model, 1),
        #         )
        self.pToExpr = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, 1),
                )
        self.add_feats = nn.Sequential(
                nn.Conv1d(in_channels = self.d_model+self.n_extraFeat, out_channels=self.d_model, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.d_model, out_channels=self.d_model, kernel_size=1),
                nn.ReLU(),
        )
        
    def forward(self, p_seq, e_seqs, e_feats, hk_tok=None, rna_feats=None):
        enhancers_padding_mask = ~(e_seqs.sum(-1).sum(-1) > 0).bool()
        padding_mask = torch.cat((torch.zeros(enhancers_padding_mask.shape[0], 1).to(self.device), enhancers_padding_mask), axis=1).bool()
        # attn_mask = (~np.identity(e_seqs.shape[1]+1).astype(bool))
        # attn_mask[:, 0] = False
        # attn_mask[0, :] = False
        # attn_mask = torch.from_numpy(attn_mask)
        # attn_mask.masked_fill(attn_mask, float('-inf'))
        # attn_mask = attn_mask.to(self.device)
        # p_embed = self.p_out(self.promoter_seq_encoder(p_seq)).permute(0, 2, 1, 3) # + self.cell_embed(cell_tok)
        # print(hk_tok.shape)
        # p_embed = self.p_out(self.promoter_seq_encoder(p_seq)).permute(0, 2, 1, 3)
        # p_embed = torch.flatten(p_embed, start_dim=2) 
        if rna_feats is not None:
            p_embed = self.rna_embed(rna_feats).unsqueeze(1) # + p_embed
        if hk_tok is not None:
            p_embed = p_embed + self.hk_embed(hk_tok)
        # print(self.hk_embed(hk_tok).shape)
        e_embed = self.e_out(self.enhancer_seq_encoder(e_seqs)).permute(0, 2, 1, 3)
        e_embed = torch.cat((torch.flatten(e_embed, start_dim=2), e_feats), axis=-1)
        # print(e_embed.shape)
        e_embed = self.add_feats(e_embed.permute(0, 2, 1)).permute(0, 2, 1)
        e_embed = torch.flatten(e_embed, start_dim=2)
        pe_embed = torch.cat((p_embed, e_embed), axis=1)
        # print(p_embed.shape, e_embed.shape, pe_embed.shape)
        attn_list = []
        for i in range(self.n_encoder):
            pe_embed, attn = self.attn_encoder[i](pe_embed, enhancers_padding_mask=padding_mask, attn_mask=None)
            attn_list.append(attn.unsqueeze(0))
        p_embed = pe_embed[:, 0, :]
        # if rna_feats is not None:
        #     p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, torch.cat(attn_list)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Logger():
    """A logging class that can report or save metrics.

    This class contains a simple utility for saving statistics as they are
    generated, saving a report to a text file at the end, and optionally
    print the report to screen one line at a time as it is being generated.
    Must begin using the `start` method, which will reset the logger.

    Parameters
    ----------
    names: list or tuple
        An iterable containing the names of the columns to be logged.

    verbose: bool, optional
        Whether to print to screen during the logging.
    """

    def __init__(self, names, verbose=False):
        self.names = names
        self.verbose = verbose

    def start(self):
        """Begin the recording process."""

        self.data = {name: [] for name in self.names}

        if self.verbose:
            print("\t".join(self.names))

    def add(self, row):
        """Add a row to the log.

        This method will add one row to the log and, if verbosity is set,
        will print out the row to the log. The row must be the same length
        as the names given at instantiation.

        Parameters
        ----------
        args: tuple or list
            An iterable containing the statistics to be saved.
        """

        assert len(row) == len(self.names)

        for name, value in zip(self.names, row):
            self.data[name].append(value)

        if self.verbose:
            print("\t".join(map(str, [round(x, 4) if isinstance(x, float) else x
                for x in row])))

    def save(self, name):
        """Write a log to disk.


        Parameters
        ----------
        name: str
            The filename to save the logs to.
        """
        pd.DataFrame(self.data).to_csv(name, sep='\t', index=False)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 6
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch_i):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}', 'best_score', self.best_score)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch_i):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
                },
                self.path)
        print('Saving ckpt at', self.path)
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train(net, training_dataset, fold_i, saved_model_path='./models/', learning_rate=1e-4, model_logger=None, fixed_encoder = False, valid_dataset = None, model_name = '', batch_size = 64, device = 'cuda', stratify=None, class_weight=None, EPOCHS=100, valid_size=1000):
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    if valid_dataset is not None:
        train_ds = training_dataset
        valid_ds = valid_dataset
    else:
        train_idx, val_idx = train_test_split(list(range(len(training_dataset))), test_size=valid_size, shuffle=True, random_state=66, stratify=stratify)
        train_ds = Subset(training_dataset, train_idx)
        valid_ds = Subset(training_dataset, val_idx)

    # fix encoder parameter
    if fixed_encoder:
        print('fixed parameter of encoder')
        for name, value in net.named_parameters():
            if name.startswith('seq_encoder'):
                value.requires_grad = False
    
    print("fold", fold_i ,"training data:", len(train_ds), "validated data:", len(valid_ds), 'total data:', len(training_dataset))
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    early_stopping = EarlyStopping(patience=5,
               verbose=True, path= saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt")

    L_expr = L1KLmixed()# nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print('model_name:', net.model_name)
    lrs = []
    # last_loss = None
    net.train()
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    for epoch in range(EPOCHS):
        net.train()
        print('learning rate:', get_lr(optimizer))
        running_loss = 0
        loss_e = 0
        # print('model training mode is:', net.training)
        for data in tqdm(trainloader):
            # print(inputs.size())
            optimizer.zero_grad()
            P_seq, E_seqs, rna_feats, E_feats, hk_tok, y_expr, eid = data
            P_seq = P_seq.float().to(device)
            E_seqs = E_seqs.float().to(device)
            if net.use_hk_tok:
                hk_tok = hk_tok.long().to(device)
            else:
                hk_tok = None
            if net.use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            E_feats = E_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(P_seq, E_seqs, E_feats, hk_tok, rna_feats)
            loss_expr = L_expr(pred_expr, y_expr)
            loss_e += loss_expr.item()
            loss = loss_expr
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
            running_loss += loss.item()

        print('[Epoch %d] loss: %.9f' %
                      (epoch + 1, running_loss/len(trainloader)))
        print('Training Loss: expression loss:', loss_e/len(trainloader))
        
        val_mse_all, val_r2_all, val_pr_all = validate(net, valid_ds, device=device)
        val_r2 = val_r2_all
        val_pr_wE, val_r2_wE = val_pr_all, val_r2_all
        print('Valdaition R square all:', val_r2_all)
        early_stopping(-val_r2, net, epoch)
        if model_logger is not None:
            label_type = net.name.split('.')[-1]
            model_logger.add([fold_i, epoch, running_loss/len(trainloader), val_mse_all, val_pr_all, val_r2_all, val_pr_wE, val_r2_wE, early_stopping.counter, label_type])
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return lrs

def validate(net, valid_ds, batch_size=16, device = 'cuda'):
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=32)
    net.eval()
    L_expr = L1KLmixed()
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in tqdm(validloader):
            P_seq, E_seqs, rna_feats, E_feats, hk_tok, y_expr, eid = data
            P_seq = P_seq.float().to(device)
            E_seqs = E_seqs.float().to(device)
            if net.use_hk_tok:
                hk_tok = hk_tok.long().to(device)
            else:
                hk_tok = None
            if net.use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            E_feats = E_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(P_seq, E_seqs, E_feats, hk_tok, rna_feats)
            outputs = list(pred_expr.flatten().cpu().detach().numpy())
            labels = list(y_expr.flatten().cpu().detach().numpy())
            loss_expr = L_expr(pred_expr, y_expr)
            loss_e += loss_expr.item()
            preds += outputs
            actual += labels
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(preds, actual)
        peasonr, pvalue = stats.pearsonr(preds, actual)
    except:
        peasonr = 0
        r_value = 0
    mse = mean_squared_error(preds, actual)
    print('Validation loss expression loss:', loss_e/len(validloader))
    print("valid: mse", mse, "R_sqaure", r_value**2, 'peasonr', peasonr)
    return mse, r_value**2, peasonr

def test(net, test_ds, fold_i, model_name = None, saved_model_path=None, batch_size=64, device = 'cuda', model_type='best'):
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    if saved_model_path is not None:
        checkpoint = torch.load(saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt")
        net.load_state_dict(checkpoint['model_state_dict'])
        print(model_name,'loaded!')
    net.eval()
    with torch.no_grad():
        preds = []
        actual = []
        ensid_list = []
        for data in tqdm(testloader):
            P_seq, E_seqs, rna_feats, E_feats, hk_tok, y_expr, eid = data
            P_seq = P_seq.float().to(device)
            E_seqs = E_seqs.float().to(device)
            if net.use_hk_tok:
                hk_tok = hk_tok.long().to(device)
            else:
                hk_tok = None
            if net.use_rna_feats:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            E_feats = E_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(P_seq, E_seqs, E_feats, hk_tok, rna_feats)
            
            outputs = list(pred_expr.flatten().cpu().detach().numpy())
            labels = list(y_expr.flatten().cpu().detach().numpy())

            preds += outputs
            actual += labels
            ensid_list += eid

    slope, intercept, r_value, p_value, std_err = stats.linregress(preds, actual)
    peasonr, pvalue = stats.pearsonr(preds, actual)
    mse = mean_squared_error(preds, actual)
    # print(fold %s test sequence: %0.3f' % (fold_i, r_value**2))
    print('\nPearson R:', peasonr)
    sys.stdout.flush()
    df = pd.DataFrame(index=np.array(ensid_list).flatten())
    df['Pred'] = preds
    df['actual'] = actual
    df['fold_idx'] = fold_i
    pearsonr_we, pvalue = stats.pearsonr(df['Pred'], df['actual'])
    print('PearsonR:', pearsonr_we)
    if saved_model_path is not None:
        df.to_csv(saved_model_path + "/fold_" + str(fold_i) + "_"+ model_name + "_predictions.csv")
    return df