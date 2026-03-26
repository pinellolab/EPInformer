import os

import sys
import argparse

from datetime import datetime
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

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import copy
import numpy as np
# import time
# import tqdm
import warnings
import torch.nn.functional as F
# import einops
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')
from .multihead_diffattn import MultiheadDiffAttn

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        
class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='batchmean', alpha=1.0, beta=0.5):
        """
        Initialize the L1KLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
    
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining L1 and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
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
        conv_dim = [self.conv_dim, 128, 128, 64, 64, 64, 64]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i], out_channels=conv_dim[i+1], kernel_size=(1, 3), padding=(0, 1)),
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
        for i in range(0, len(self.conv_tower), 2):
            x_enhancer = self.conv_tower[i](x_enhancer)
            x_enhancer = self.conv_tower[i+1](x_enhancer) + x_enhancer
            
        return x_enhancer

# torch.Size([30, 128, 1, 500])
# torch.Size([30, 128, 1, 250])
# torch.Size([30, 64, 1, 125])
# torch.Size([30, 64, 1, 62])

class enhancer_predictor_256bp(nn.Module):
    def __init__(self):
        super(enhancer_predictor_256bp, self).__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )  
    def forward(self, enhancer_seq):
        if len(enhancer_seq.shape) < 4:
            enhancer_seq = enhancer_seq.unsqueeze(2)
        seq_embed = self.encoder(enhancer_seq)
        epi_out = self.embedToAct(seq_embed)
        return epi_out.squeeze(-1)

class enhancer_deep_predictor_256bp(nn.Module):
    def __init__(self):
        super(enhancer_deep_predictor_256bp, self).__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, 256),
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

class MHAttentionDiff_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super(MHAttentionDiff_encoderLayer, self).__init__()
        # self.activation = activation
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = MultiheadDiffAttn(embed_dim=d_model, depth=12, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.Linear(d_model*4, d_model),
            SwiGLU(d_model),
            nn.Dropout(dropout),
        )
    # self-attention block
    def _sa_block(self, x, key_padding_mask=None, attn_mask=None):
        # x, w = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x, w = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
        
    def forward(self, x, enhancers_padding_mask=None, attn_mask=None):
        xt, attention_w = self._sa_block(self.norm1(x), key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
        x = xt + x
        x = x + self.ff(self.norm2(x))
        return x, attention_w

class MHAttention_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super(MHAttention_encoderLayer, self).__init__()
        # self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_attn = MultiheadDiffAttn(embed_dim=d_model, depth=12, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.Linear(d_model*4, d_model),
            SwiGLU(d_model),
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

class EPInformerMT_diffattn(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=64, pre_trained_encoder= None, n_enhancers=220, device='cuda', 
                 use_rna_feats = False, n_heads = 8, n_enh_feats=0, use_pro_seq=False, n_cell=10):
        super(EPInformerMT_diffattn, self).__init__()
        self.n_enhancers = n_enhancers
        self.use_pro_seq = use_pro_seq
        self.out_dim = out_dim
        self.n_cell = n_cell
        self.use_rna_feats = use_rna_feats
        if use_rna_feats:
            self.model_name = 'epinformer_diffattn_rnafeats'
        else:
            self.model_name = 'epinformer_diffattn'
        if use_pro_seq:
            self.model_name = self.model_name + '_proseq'
        self.n_heads = n_heads
        self.n_extraFeat = n_enh_feats
        self.base_size = base_size
        self.p_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(38, 8)
        )
        self.e_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(113, 8)
        )
        if pre_trained_encoder is not None:
            self.cre_seq_encoder = pre_trained_encoder
            self.promoter_seq_encoder = pre_trained_encoder
            self.model_name = self.model_name + '_preTrainedEncoder'
            # self.name = 'EPInformerV2.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.cre_seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.promoter_seq_encoder = seq_256bp_encoder(base_size=base_size)
            # self.name = 'EPInformerV2.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttentionDiff_encoderLayer(d_model=256, nhead=self.n_heads), self.n_encoder)

        # attn_mask = (~np.identity(self.n_enhancers+1).astype(bool))
        # attn_mask[:, 0] = False
        # attn_mask[0, :] = False
        # attn_mask = torch.from_numpy(attn_mask)
        # attn_mask.masked_fill(attn_mask, float('-inf'))
        # self.attn_mask = attn_mask

        self.cell_embed = nn.Embedding(self.n_cell, 256)
        self.cre_type_embed = nn.Embedding(4, 256)
        self.rna_feat_embed = nn.Linear(8, 256)
        self.promter_signal_embed = nn.Linear(1, 256)
        self.pToExpr = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )

        conv_feat = nn.Sequential(
                nn.Conv1d(in_channels = 257, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = 128, out_channels=256, kernel_size=1),
        )
        self.add_feats = get_clones(conv_feat, self.n_extraFeat)
        self.attn_encoder = get_clones(MHAttentionDiff_encoderLayer(d_model=256, nhead=self.n_heads), self.n_encoder)
        
    def forward(self, p_seq=None, e_seqs=None, e_types=None, e_feats=None, cell_tok=None, rna_feats=None):
        # if enhancers_padding_mask is None:
        enhancers_padding_mask = ~(e_seqs.sum(-1).sum(-1) > 0).bool()
        padding_mask = torch.cat((torch.zeros(enhancers_padding_mask.shape[0], 1).to(self.device),
                                  enhancers_padding_mask), axis=1).bool()
        # padding_mask = (padding_mask.unsqueeze(-1).repeat(1,1,padding_mask.shape[-1]).permute(0,2,1) + padding_mask.unsqueeze(-1).repeat(1,1, padding_mask.shape[-1])).bool().int()
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1) # .repeat(1,self.n_heads*2,1,1)
        padding_mask = torch.where(padding_mask == 1, torch.tensor(float('-inf')), padding_mask).float()  
        # pad_masks = (padding_mask.unsqueeze(-1).repeat(1,1,5).permute(0,2,1) + padding_mask.unsqueeze(-1).repeat(1,1,5)).bool().int()
        # pad_masks = pad_masks.unsqueeze(1).repeat(1,head*2,1,1)
        # pad_masks = torch.where(pad_masks == 1, torch.tensor(float('-inf')), pad_masks)   
        # print(enhancers_padding_mask)
        if self.use_pro_seq:
            p_embed = self.p_out(self.promoter_seq_encoder(p_seq)).permute(0, 2, 1, 3) 
            p_embed = torch.flatten(p_embed, start_dim=2) + self.cell_embed(cell_tok) #+ self.rna_feat_embed(rna_feats)
        else:
            p_embed = self.cell_embed(cell_tok)
        if self.use_rna_feats:
            p_embed = p_embed + self.rna_feat_embed(rna_feats)

        e_embed = self.e_out(self.cre_seq_encoder(e_seqs)).permute(0, 2, 1, 3)
        e_embed = torch.flatten(e_embed, start_dim=2)
        # e_embed = torch.cat((torch.flatten(e_embed, start_dim=2), e_feats), axis=-1)
        # print(self.cre_type_embed(e_types).shape, e_types.shape)
        e_embed = e_embed + self.cre_type_embed(e_types).squeeze(2)
        for i in range(self.n_extraFeat):
            e_embed = torch.cat((e_embed, e_feats[:,:,[i]]), axis=-1)
            e_embed = self.add_feats[i](e_embed.permute(0, 2, 1)).permute(0, 2, 1)
        
        # e_embed = e_embed.permute(0, 2, 1)
        # e_embed = torch.flatten(e_embed, start_dim=2)
        # for i in range(self.n_enh_feats):
        #     feat_embed = self.feat_embeddings[i](e_feats[:,:,[i]])
        #     e_embed = e_embed + feat_embed
        # print(p_embed.shape, e_embed.shape)
        pe_embed = torch.cat((p_embed, e_embed), axis=1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_embed, attn = self.attn_encoder[i](pe_embed, enhancers_padding_mask=padding_mask, attn_mask=None)
            attn_list.append(attn.unsqueeze(0))
        p_embed = pe_embed[:, 0, :]
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, torch.cat(attn_list)



class EPInformerMT_vanilla(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=64, pre_trained_encoder= None, n_enhancers=220, device='cuda', 
                 use_rna_feats = False, n_heads = 8, n_enh_feats=0, use_pro_seq=False, n_cell=10):
        super(EPInformerMT_vanilla, self).__init__()
        self.n_enhancers = n_enhancers
        self.use_pro_seq = use_pro_seq
        self.out_dim = out_dim
        self.n_cell = n_cell
        self.use_rna_feats = use_rna_feats
        if use_rna_feats:
            self.model_name = 'epinformer_vanilla_rnafeats'
        else:
            self.model_name = 'epinformer_vanilla'
        if use_pro_seq:
            self.model_name = self.model_name + '_proseq'
        self.n_heads = n_heads
        self.n_extraFeat = n_enh_feats
        self.base_size = base_size
        self.p_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(38, 8)
        )
        self.e_out = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Linear(113, 8)
        )
        if pre_trained_encoder is not None:
            self.cre_seq_encoder = pre_trained_encoder
            # self.promoter_seq_encoder = pre_trained_encoder
            self.model_name = self.model_name + '_preTrainedEncoder'
            # self.name = 'EPInformerV2.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.cre_seq_encoder = seq_256bp_encoder(base_size=base_size)
            # self.promoter_seq_encoder = seq_256bp_encoder(base_size=base_size)
            # self.name = 'EPInformerV2.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttentionDiff_encoderLayer(d_model=256, nhead=self.n_heads), self.n_encoder)

        # attn_mask = (~np.identity(self.n_enhancers+1).astype(bool))
        # attn_mask[:, 0] = False
        # attn_mask[0, :] = False
        # attn_mask = torch.from_numpy(attn_mask)
        # attn_mask.masked_fill(attn_mask, float('-inf'))
        # self.attn_mask = attn_mask

        self.cell_embed = nn.Embedding(self.n_cell, 256)
        self.cre_type_embed = nn.Embedding(4, 256)
        self.rna_feat_embed = nn.Linear(8, 256)
        self.promter_signal_embed = nn.Linear(1, 256)
        self.pToExpr = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )
        conv_feat = nn.Sequential(
                nn.Conv1d(in_channels = 257, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = 128, out_channels=256, kernel_size=1),
        )
        self.add_feats = get_clones(conv_feat, self.n_extraFeat)
        # self.add_feats = nn.Sequential(
        #         nn.Conv1d(in_channels = 256+self.n_extraFeat, out_channels=128, kernel_size=1),
        #         nn.ReLU(),
        #         nn.Conv1d(in_channels = 128, out_channels=256, kernel_size=1),
        # )
        # self.add_feats = get_clones(conv_feat, self.n_extraFeat)
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=256, nhead=self.n_heads), self.n_encoder)
        
    def forward(self, p_seq=None, e_seqs=None, e_types=None, e_feats=None, cell_tok=None, rna_feats=None):
        # if enhancers_padding_mask is None:
        enhancers_padding_mask = ~(e_seqs.sum(-1).sum(-1) > 0).bool()
        padding_mask = torch.cat((torch.zeros(enhancers_padding_mask.shape[0], 1).to(self.device),
                                  enhancers_padding_mask), axis=1).bool()
        if self.use_pro_seq:
            p_embed = self.p_out(self.cre_seq_encoder(p_seq)).permute(0, 2, 1, 3) 
            p_embed = torch.flatten(p_embed, start_dim=2) + self.cell_embed(cell_tok) #+ self.rna_feat_embed(rna_feats)
        else:
            p_embed = self.cell_embed(cell_tok)
        if self.use_rna_feats:
            p_embed = p_embed + self.rna_feat_embed(rna_feats)

        e_embed = self.e_out(self.cre_seq_encoder(e_seqs)).permute(0, 2, 1, 3)
        e_embed = torch.flatten(e_embed, start_dim=2)

        # e_embed = e_embed + self.cre_type_embed(e_types).squeeze(2)
        for i in range(self.n_extraFeat):
            e_embed = torch.cat((e_embed, e_feats[:,:,[i]]), axis=-1)
            e_embed = self.add_feats[i](e_embed.permute(0, 2, 1)).permute(0, 2, 1)
        
        pe_embed = torch.cat((p_embed, e_embed), axis=1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_embed, attn = self.attn_encoder[i](pe_embed, enhancers_padding_mask=padding_mask, attn_mask=None)
            attn_list.append(attn.unsqueeze(0))
        p_embed = pe_embed[:, 0, :]
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, torch.cat(attn_list)
