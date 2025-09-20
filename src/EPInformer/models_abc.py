from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
import math
from sklearn.metrics import mean_squared_error
import copy
import numpy as np
import pandas as pd
from scipy import stats
# import time
# import tqdm
import warnings
import torch.nn.functional as F
# import einops
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class seq_256bp_encoder(nn.Module):
    def __init__(self, base_size=4, out_dim=128, conv_dim=256):
        super(seq_256bp_encoder, self).__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        # cropped_len = 46
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels = base_size, out_channels = self.conv_dim, kernel_size = (1, 8), stride = 1, padding='same'),
            nn.ELU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dim = [self.conv_dim, 128, 64, 64, 128]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i], out_channels=conv_dim[i+1], kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(conv_dim[i+1]),
                nn.ELU(),                   
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ))
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i+1], out_channels=conv_dim[i+1], kernel_size=(1, 1)),
                nn.ELU(),
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
    def __init__(self, d_model=128, nhead=8, dropout=0.):
        super(MHAttention_encoderLayer, self).__init__()
        # self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, 4*d_model) might cause loading problem, this parameter is not neccessary
        # self.linear2 = nn.Linear(4*d_model, d_model) might cause loading problem, this parameter is not neccessary
        # self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
    # self-attention block
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
        
    def forward(self, x, enhancers_padding_mask=None, attn_mask=None):
        x2 = self.norm1(x)
        x2, attention_w = self._sa_block(x2, key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
        x = x2 + x
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x, attention_w

# class MHAttention_encoderLayer_addFeat(nn.Module):
#     def __init__(self, n_extraFeat, d_model=128, nhead=8, dropout=0.1):
#         super(MHAttention_encoderLayer_addFeat, self).__init__()
#         # self.activation = activation
#         self.n_extraFeat = n_extraFeat
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_model*4),
#             nn.ReLU(),
#             nn.Linear(d_model*4, d_model)
#         )
#         feat_encoder_list = []
#         for _ in range(self.n_extraFeat):
#             feat_encoder_list.append(ContinuousValueEncoder(d_model=16, max_value=50_000))
#         self.feat_encoder = nn.ModuleList(feat_encoder_list)
#     # self-attention block
#     def _sa_block(self, x, key_padding_mask, attn_mask):
#         x, w = self.self_attn(x, x, x,
#                            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         return x, w

#     def forward(self, x, x_feat=None, enhancers_padding_mask=None, attn_mask=None):
#         x2 = self.norm1(x)
#         x2, attention_w = self._sa_block(x2, key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
#         # if x_feat is not None:
#         #     for i in range(x_feat.shape[-1]):
#         #         x_feat_emb = self.feat_encoder[i](x_feat[:, :, i])
#         #         x2[:, :1, :] = x_feat_emb.permute(0, 2, 1) @ x2
#         x = x2 + x
#         x2 = self.norm2(x)
#         x = x + self.ff(x2)
#         return x, attention_w

class EPInformer_abc(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=False, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_abc, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer-abc-v0.1.preTrainedConv'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer-abc-v0.1'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        # feat_encoder_list = []
        # for _ in range(self.n_extraFeat):
        #     feat_encoder_list.append(ContinuousValueEncoder(d_model=32, max_value=50_000))
        # self.feat_encoder = nn.ModuleList(feat_encoder_list)
        self.feat_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                # nn.LayerNorm(64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_extraFeat)
        ])
        
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        n_feat = 0
        if self.useFeat:
            if self.usePromoterSignal:
                n_feat = 9
            else:
                n_feat = 8
        self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+n_feat, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        if n_extraFeat == 0:
            n_extraFeat = n_extraFeat + 1
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)

        # pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        pe_flatten_embed[:, 1:, :] = self.add_pos_conv(torch.concat([pe_flatten_embed[:, 1:, :], enh_feats[:, 1:, :]], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        # p_embed_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        # p_embed_list = []
        # set the first enhancer feature to be the promoter
        # enh_feats[:, :, 0] = 1 
        if self.n_extraFeat > 2:
            feats_a = enh_feats[:, 1:, 1]*~enhancers_padding_mask[:, 1:]
            feats_b = enh_feats[:, 1:, 2]*~enhancers_padding_mask[:, 1:]
            feats_a = self.feat_encoders[1](feats_a.unsqueeze(-1)) # *~enhancers_padding_mask
            feats_b = self.feat_encoders[2](feats_b.unsqueeze(-1))
            feats_w = (feats_a.squeeze(-1)* ~enhancers_padding_mask[:, 1:]) * (feats_b.squeeze(-1) * ~enhancers_padding_mask[:, 1:]) 
            feats_w = feats_w.masked_fill(enhancers_padding_mask[:, 1:], float('-inf'))
            feats_w = F.softmax(feats_w, dim=-1)
            feats_w = torch.cat((enh_feats[:, [0], 1], torch.nan_to_num(feats_w, nan=0.0)), dim=1)
            p_embed = feats_w.unsqueeze(1) @ pe_flatten_embed # ABC-enhanced Attention score
            p_embed = p_embed.flatten(start_dim=1) # ABC-enhanced Attention score

        elif self.n_extraFeat == 2:
            feats_a = enh_feats[:, 1:, 0]*~enhancers_padding_mask[:, 1:]
            feats_b = enh_feats[:, 1:, 1]*~enhancers_padding_mask[:, 1:]
            feats_a = self.feat_encoders[0](feats_a.unsqueeze(-1)) # *~enhancers_padding_mask
            feats_b = self.feat_encoders[1](feats_b.unsqueeze(-1))
            feats_w = (feats_a.squeeze(-1)* ~enhancers_padding_mask[:, 1:]) * (feats_b.squeeze(-1) * ~enhancers_padding_mask[:, 1:]) 
            feats_w = feats_w.masked_fill(enhancers_padding_mask[:, 1:], float('-inf'))
            feats_w = F.softmax(feats_w, dim=-1)
            feats_w = torch.cat((enh_feats[:, [0], 1], torch.nan_to_num(feats_w, nan=0.0)), dim=1)
            p_embed = feats_w.unsqueeze(1) @ pe_flatten_embed # ABC-enhanced Attention score
            p_embed = p_embed.flatten(start_dim=1) # ABC-enhanced Attention score
        else:
            p_embed = pe_flatten_embed[:, 0, :]
        if rna_feats is not None:
            p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, (torch.cat(attn_list), feats_w)

class EPInformer_abc_dist(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=False, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_abc_dist, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer-abc-dist-v0.6.preTrainedConv'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer-abc-dist-v0.6'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        self.feat_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                # nn.LayerNorm(64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_extraFeat)
        ])
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        n_feat = 0
        if self.useFeat:
            if self.usePromoterSignal:
                n_feat = 9
            else:
                n_feat = 8
        self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+n_feat, 128),
                        # nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        if n_extraFeat == 0:
            n_extraFeat = n_extraFeat + 1
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        # if enh_feats is not None:
        #     pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        # feat_embeds = self.feat_encoder[0](enh_feats[:, :, 0])
        # print(pe_flatten_embed.shape, feat_embeds.shape)
        # pe_flatten_embed = pe_flatten_embed + feat_embeds
        pe_flatten_embed[:, 1:, :] = self.add_pos_conv(torch.concat([pe_flatten_embed[:, 1:, :], enh_feats[:, 1:, :]], axis=-1).permute(0,2,1)).permute(0,2,1)
        # pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        # e_embed_list = []
        # feat_w_list = []
        # for i in range(1, self.n_extraFeat):
        if self.n_extraFeat > 1:
            if self.n_extraFeat == 3:
                feats_a =  enh_feats[:, 1:, 1]
                feats_w = enh_feats[:, 1:, 1]*enh_feats[:, 1:, 2]*~enhancers_padding_mask[:, 1:]   # add promomter
            elif self.n_extraFeat == 2:
                feats_w = enh_feats[:, 1:, 1]*~enhancers_padding_mask[:, 1:]
                # print(feats_w)
                # feats_w = self.feat_encoders[0](feats_w.unsqueeze(-1)).squeeze(-1)  
                    # softmax with mask
                    # feats_w = feats - torch.max(feats, dim=-1, keepdim=True).values
            feats_w = feats_w.masked_fill(enhancers_padding_mask[:, 1:], float('-inf'))
            feats_w = F.softmax(feats_w, dim=-1)
                # print(feats_w.shape, enh_feats.shape)
                # add promoter feats
            feats_w = torch.cat((enh_feats[:, [0], 0], feats_w), dim=1) # add promoter feats
                # print(feats_w.shape) 
                # feats_w = F.softmax(feats_w, dim=-1) # normalize the feature weights
            p_embed = feats_w.unsqueeze(1) @ pe_flatten_embed
            p_embed = p_embed.flatten(start_dim=1) # ABC-enhanced Attention score
        else:
            p_embed = pe_flatten_embed[:, 0, :]
            # feat_w_list.append(feats_w)
            # e_embed_list.append(p_embed)
            # e_embed = torch.cat(e_embed_list, axis=1).sum(1) # ABC-enhanced Attention score
            # p_embed = pe_flatten_embed[:, 0, :] + e_embed
        if rna_feats is not None:
            p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, (torch.cat(attn_list), feats_w)
    
class EPInformer_abc_v2(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=True, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_abc_v2, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer-abc-v2.preTrainedConv'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer-abc-v2'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        feat_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.Linear(32, 1),
            nn.Sigmoid())
        
        feat_encoder_list = []
        for _ in range(self.n_extraFeat):
            feat_encoder_list.append(feat_encoder())
        self.feat_encoders = nn.ModuleList(feat_encoder_list)
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.combine_feat = nn.Sequential(nn.Linear(3, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 1))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        n_feat = 0
        if self.useFeat:
            if self.usePromoterSignal:
                n_feat = 9
            else:
                n_feat = 8
        self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+n_feat, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        if n_extraFeat == 0:
            n_extraFeat = n_extraFeat + 1
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        enh_feat_embeds = []
        for i in range(self.n_extraFeat):
            feat_embed = self.feat_encoder[i](enh_feats[:, 1:, i])
            # print(feat_embed.shape, enh_feats[:, :1, [i]].shape)
            feat_embed = torch.cat([enh_feats[:, :1, [i]], feat_embed], axis=1)
            enh_feat_embeds.append(feat_embed)
        # enh_feat_embeds =  self.combine_feat(torch.cat(enh_feat_embeds, axis=-1))
        attn_list = []
        for i in range(self.n_encoder):
            for j in range(self.n_extraFeat):
                pe_flatten_embed = enh_feat_embeds[j] * pe_flatten_embed
            # pe_flatten_embed = pe_flatten_embed * enh_feat_embeds
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        p_embed = pe_flatten_embed[:,0,:]
        if rna_feats is not None:
            p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, torch.cat(attn_list)


class EPInformer_v1(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=True, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_v1, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer_v1.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer_v1.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        if useLN:
            self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        else:
            self.attn_encoder = get_clones(MHAttention_encoderLayer_noLN(d_model=out_dim, nhead=head), self.n_encoder)
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        # attn_mask.to(self.device)
        self.attn_mask = attn_mask# .to(self.device)
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        if self.useFeat:
            if self.usePromoterSignal:
                feat_n = 9
            else:
                feat_n = 8
            self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+feat_n, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        else:
            self.pToExpr = nn.Sequential(
                    nn.Linear(self.out_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        # if enhancers_padding_mask is None:
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
#         print(enhancers_padding_mask)
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        if enh_feats is not None:
            pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        p_embed = torch.flatten(pe_flatten_embed[:,0,:], start_dim=1)
        if self.useFeat:
            p_embed = torch.cat([p_embed, rna_feats], dim=-1)
        p_expr = self.pToExpr(p_embed)
        return p_expr, torch.cat(attn_list)

class EPInformer_v2(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=True, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_v2, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer-v2.preTrainedConv'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer-v2'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        n_feat = 0
        if self.useFeat:
            if self.usePromoterSignal:
                n_feat = 9
            else:
                n_feat = 8
        self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+n_feat, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        if n_extraFeat == 0:
            n_extraFeat = n_extraFeat + 1
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )
    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        if enh_feats is not None:
            pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        # Guide attention module

        p_embed = pe_flatten_embed[:,0,:]
        if rna_feats is not None:
            p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, torch.cat(attn_list)
    
class EPInformer_abc_dist_v2(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=False, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_abc_dist_v2, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformer-abc-dist-v2.2.preTrainedConv'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformer-abc-dist-v2.2'# .{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        self.feat_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                # nn.LayerNorm(64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_extraFeat)
        ])
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 3)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(109, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        n_feat = 0
        if self.useFeat:
            if self.usePromoterSignal:
                n_feat = 9
            else:
                n_feat = 8
        self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+n_feat, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        if n_extraFeat == 0:
            n_extraFeat = n_extraFeat + 1
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feats=None, enh_feats=None):
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        if self.usePromoterSignal:
            pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, enh_feats], axis=-1).permute(0,2,1)).permute(0,2,1)
        else:
            pe_flatten_embed[:, 1:, :] = self.add_pos_conv(torch.concat([pe_flatten_embed[:, 1:, :], enh_feats[:, 1:, :]], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))

        feats_w = None
        if self.n_extraFeat > 1:
            if self.n_extraFeat == 3:
                feats_w = torch.log(enh_feats[:, 1:, 1]*enh_feats[:, 1:, 2]+1)
                feats_w = feats_w*~enhancers_padding_mask[:, 1:]   # add promomter
                feats_w = feats_w.masked_fill(enhancers_padding_mask[:, 1:], float('-inf'))
                feats_w = F.softmax(feats_w, dim=-1)
            elif self.n_extraFeat == 2:
                feats_w = enh_feats[:, 1:, 1]/(enh_feats[:, 1:, 0]+1)*1000
                feats_w = feats_w*~enhancers_padding_mask[:, 1:]
                feats_w = feats_w.masked_fill(enhancers_padding_mask[:, 1:], 0.0)
                feats_w_sum = torch.sum(feats_w, dim=-1, keepdim=True)
                feats_w = feats_w / (feats_w_sum + 1e-8)

            feats_w = torch.cat((enh_feats[:, [0], 1], torch.nan_to_num(feats_w, nan=0.0)), dim=1) # add promoter feats
            p_embed = feats_w.unsqueeze(1) @ pe_flatten_embed
            p_embed = p_embed.flatten(start_dim=1) 
        else:
            p_embed = pe_flatten_embed[:, 0, :]
        if rna_feats is not None:
            p_embed = torch.cat((p_embed, rna_feats), axis=-1)
        expr_out = self.pToExpr(p_embed).squeeze(-1)
        return expr_out, (torch.cat(attn_list), feats_w)
