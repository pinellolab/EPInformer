import sys
import argparse

from datetime import datetime
import os
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np

from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import Subset, Dataset

parser = argparse.ArgumentParser()
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--cell', type=str, help='cell line (support K562 and GM12878)', choices=['K562', 'GM12878'])  
parser.add_argument("--fold", type=list_of_strings, help="test fold", default='all')
parser.add_argument("--model_type", type=str, help='EPInformer type', default='EPInformer-PE-Activity', choices=['EPInformer-PE', 'EPInformer-PE-Activity', 'EPInformer-PE-Activity-HiC'])  
parser.add_argument('--distance_threshold', type=int, help='max distance to TSS', default=100_000) 
parser.add_argument('--hic_threshold', type=int, help='hic loop thresold', default=-1) 
parser.add_argument('--expr_assay', type=str, help='expression_assay', choices=['CAGE', 'RNA'])
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--n_interact_enc',type=int, help='layers of interaction encoder', default=3)
parser.add_argument('--epochs',type=int, help='training epochs', default=100)
parser.add_argument('--cuda', help='use cuda', action='store_true')
parser.add_argument('--use_pretrained_encoder', help='use pretrained sequence encoder', action='store_true')

# example
# python train_EPInformer.py --cell K562  --model_type EPInformer-PE-Activity --expr_assay CAGE --use_pretrained_encoder --batch_size 16

##### parameter ######
args = parser.parse_args()

cell = args.cell

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'
distance_threshold = args.distance_threshold
n_epoch = args.epochs
hic_threshold = args.hic_threshold
if hic_threshold == -1:
    hic_threshold = None

if args.model_type == 'EPInformer-PE': 
    n_extraFeat = 1
elif args.model_type == 'EPInformer-PE-Activity':
    n_extraFeat = 2
elif args.model_type == 'EPInformer-PE-Activity-HiC':
    n_extraFeat = 3

use_pretrained = args.use_pretrained_encoder
fold_list = args.fold 
n_encoder = args.n_interact_enc
batch_size = args.batch_size 
expr_type = args.expr_assay
n_enhancers = 60
#################

today = datetime.now()   # Get date

datetime_str = today.strftime("%Y-%m-%d-%H")
split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
saved_model_path = './trained_models/{}/'.format(datetime_str)

if 'all' in fold_list:
    fold_list = list(range(1, 13))
else:
    fold_list = fold_list
for fi in fold_list:
    print("-"*10, 'fold', fi, '-'*10)
    fold_i = 'fold_' + str(fi)

    train_ensid = split_df[split_df[fold_i] == 'train'].index
    valid_ensid = split_df[split_df[fold_i] == 'valid'].index
    test_ensid = split_df[split_df[fold_i] == 'test'].index

    all_ds = utils.promoter_enhancer_dataset(data_folder= './data/', expr_type=expr_type, cell_type=cell, n_extraFeat=n_extraFeat, usePromoterSignal=True, n_enhancers=n_enhancers, hic_threshold=hic_threshold, distance_threshold=distance_threshold)
    ensid_list = [eid.decode() for eid in all_ds.data_h5['ensid'][:]]
    ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
    ensid_df['idx'] = np.arange(len(ensid_list))
    ensid_df = ensid_df.set_index('ensid')
    train_idx = ensid_df.loc[train_ensid]['idx']
    valid_idx = ensid_df.loc[valid_ensid]['idx']

    test_idx = ensid_df.loc[test_ensid]['idx']

    train_ds = Subset(all_ds, train_idx)
    valid_ds = Subset(all_ds, valid_idx)
    test_ds = Subset(all_ds, test_idx)

    if use_pretrained:
        pretrained_convNet = enhancer_predictor_256bp()
        pt_model_name = '{}_seq2activityLog2_leaveChrOut_combinedRS_2bins_bs64_H3K27ac_adamW_erisxdl_r0'.format(cell)
        checkpoint = torch.load("./trained_models/pretrained_enhancer_encoder/{}_best_{}_checkpoint.pt".format(fold_i, pt_model_name))
        print('Loading pretrained model ...', pt_model_name)
        model = EPInformer_v2(n_encoder=n_encoder, pre_trained_encoder=pretrained_convNet.encoder, n_enhancer=n_enhancers, out_dim=64, n_extraFeat=n_extraFeat, device=device).to(device)
    else:
        model = EPInformer_v2(n_encoder=n_encoder, pre_trained_encoder=None, n_enhancer=n_enhancers, out_dim=64, n_extraFeat=n_extraFeat, device=device).to(device)

    model = model.to(device)
    model.name = model.name.replace('EPInformerV2', args.model_type) + '.' +  cell + '.' + expr_type
    utils.train(model, train_ds, valid_dataset=valid_ds, EPOCHS=n_epoch, model_name = model.name, fold_i=fi, batch_size=batch_size, device=device, saved_model_path=saved_model_path)
    test_df = utils.test(model, test_ds, model_name = model.name, saved_model_path=saved_model_path, fold_i=fi, batch_size=batch_size, device=device)
