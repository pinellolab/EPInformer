import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
from datetime import datetime
import os
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np

from scipy import stats
from tqdm import tqdm
import torch
import h5py
from torch import Tensor
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error
import copy
import numpy as np
import warnings
import torch.nn.functional as F

import torch.utils.data as data_utils
from torch.utils.data import Subset, Dataset, DataLoader
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from kipoiseq import Interval
import pyfaidx
import kipoiseq
from EPInformer.models_abc import EPInformer_abc, EPInformer_v2, EPInformer_abc_dist, EPInformer_abc_dist_v2, enhancer_predictor_256bp

# def df_to_pyranges(df, start_col='start', end_col='end', chr_col='chr', start_slop=0, end_slop=0):
#     df['Chromosome'] = df[chr_col]
#     df['Start'] = df[start_col] - start_slop
#     df['End'] = df[end_col] + end_slop
#     return(pr.PyRanges(df))

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream
        
    def close(self):
        return self.fasta.close()
    
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence, neutral_value=0.0).astype(np.float32)

fasta_extractor = FastaStringExtractor("./data/hg38.fa")

class promoter_enhancer_dataset(Dataset):
    def __init__(self, cell_type='K562', expr_type='RNA', n_enh_feats=3, disable_enh=False, distance_thr=None, max_n_enh=200, use_prm_signal=False, rm_prm_seq=False):
        # self.data_h5 = h5py.File('/mnt/usb1/jiecong/{}_220CREs-gene_strand.hdf5'.format(cell_type), 'r') # /mnt/usb1/jiecong
        self.data_h5 = h5py.File('/dev/shm/data/{}_200CREs-gene_RPM_4feats.hdf5'.format(cell_type), 'r')
        self.rm_prm_seq = rm_prm_seq
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        # self.expr_df = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso.csv', index_col='gene_id')
        self.expr_df = pd.read_csv('./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv', index_col='gene_id')
        self.cell2tok = {'K562':0, 'GM12878':1, 'HepG2':2, 'NHEK': 3, 'HUVEC': 4, 'H1': 5}
        if cell_type == 'K562':
            promoter_df = pd.read_csv('./data/K562/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'GM12878':
            promoter_df = pd.read_csv('./epinformer_data_20250503/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
        elif cell_type == 'HepG2':
            promoter_df = pd.read_csv('./epinformer_data_20250503/HepG2/DNase_ENCFF691HJY_Neighborhoods/GeneList.txt', sep='\t', index_col='name')
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
        enh_feats = self.data_h5['enhancers_feat'][idx][:,:]
        prm_seq = self.expr_df.loc[sample_ensid, 'promoter_2k']
        prm_ohe = one_hot_encode(prm_seq)[np.newaxis,:]
        prm_signal = np.log(1+np.array([self.promoter_df.loc[sample_ensid, 'promoter_activity']]))
        if self.n_enh_feats == 0:
            enh_feats = np.zeros_like(np.concatenate([abs(enh_feats[:, [0]]), enh_feats[:, [3]], enh_feats[:, [-1]]], axis=1)[:,:1])
        else:
            enh_feats = np.concatenate([abs(enh_feats[:, [0]]), enh_feats[:, [3]], enh_feats[:, [-1]]], axis=1)[:,:self.n_enh_feats]
        rna_feats = np.array(self.expr_df.loc[sample_ensid][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore',
                             'UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float)).flatten()
        if self.use_prm_signal:
            rna_feats = np.concatenate([rna_feats, prm_signal])
        if self.distance_thr is not None: 
            enh_ohe_new = np.zeros((self.max_n_enh, 2000, 4))
            enh_feats_new = np.zeros((self.max_n_enh, enh_feats.shape[-1]))
            new_i = 0
            for i in range(enh_ohe.shape[0]):
                if not self.rm_prm_seq:
                    if abs(enh_feats[i][0])<=self.distance_thr:
                        # enh_ohe[i] = np.zeros_like(enh_ohe[i])
                        enh_ohe_new[new_i] = enh_ohe[i]
                        enh_feats_new[new_i] = enh_feats[i]
                        new_i += 1
                else:
                    if abs(enh_feats[i][0])<=self.distance_thr and abs(enh_feats[i][0])>=1000:
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
            expr = np.log10(self.expr_df.loc[sample_ensid, self.cell_type + '_CAGE_128*3_sum']+1)
        else:
            expr = self.expr_df.loc[sample_ensid, 'Actual_' + self.cell_type]
        pe_ohe = np.concatenate([prm_ohe, enh_ohe], axis=0)
        prm_feats = np.ones_like(enh_feats[[0]])
        if self.use_prm_signal and self.n_enh_feats == 3:
            prm_feats[0, 1] = prm_signal
        pe_feats = np.concatenate([prm_feats, enh_feats], axis=0)
        return pe_ohe, rna_feats, pe_feats, expr, sample_ensid
    
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
        self.val_loss_min = np.inf
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
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    early_stopping = EarlyStopping(patience=5,
               verbose=True, path= saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt")

    L_expr = nn.SmoothL1Loss() # L1KLmixed()# nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    # print('model_name:', net.model_name)
    lrs = []
    for epoch in range(EPOCHS):
        net.train()
        print('learning rate:', get_lr(optimizer))
        running_loss = 0
        loss_e = 0
        # print('model training mode is:', net.training)
        for data in tqdm(trainloader):
            # print(inputs.size())
            optimizer.zero_grad()
            pe_seqs, rna_feats, enh_feats, y_expr, eid = data
            # print('pe_seqs shape:', pe_seqs.shape, 'enh_feats shape:', enh_feats.shape, 'y_expr shape:', y_expr.shape)
            pe_seqs = pe_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(pe_seqs, enh_feats=enh_feats, rna_feats=rna_feats)
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
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    net.eval()
    L_expr = nn.SmoothL1Loss() #L1KLmixed()
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in tqdm(validloader):
            pe_seqs, rna_feats, enh_feats, y_expr, eid = data
            pe_seqs = pe_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(pe_seqs, enh_feats=enh_feats, rna_feats=rna_feats)
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

def test(net, test_ds, fold_i, model_name = None, saved_model_path=None, batch_size=64, device='cuda', model_type='best'):
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    if saved_model_path is not None:
        checkpoint = torch.load(saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt", weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(model_name,'loaded!')
    net.eval()
    with torch.no_grad():
        preds = []
        actual = []
        ensid_list = []
        for data in testloader:
            pe_seqs, rna_feats, enh_feats, y_expr, eid = data
            pe_seqs = pe_seqs.float().to(device)
            if net.useFeat:
                rna_feats = rna_feats.float().to(device)
            else:
                rna_feats = None
            enh_feats = enh_feats.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr, _ = net(pe_seqs, enh_feats=enh_feats, rna_feats=rna_feats)
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

split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes_addBorzoi_enhancer_annot.csv', index_col=0)
# split_df = split_df[split_df['with_enhancer_100kb'] == True]
device = 'cuda'
parser = argparse.ArgumentParser()
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--cuda_id', type=int, help='cuda id', default=0)
parser.add_argument('--model_type', type=str, help='model type', default='EPInformer-v2', choices=['EPInformer-v2', 'EPInformer-abc', 'EPInformer-abc-dist', 'EPInformer-abc-dist-v2'])
parser.add_argument('--expr_type', type=str, help='expression type', default='RNA', choices=['CAGE', 'RNA'])
parser.add_argument('--n_enh_feats', type=int, help='number of enhancer features', default=3, choices=[1, 2, 3])
parser.add_argument('--cell', type=str, help='cell type', default='K562', choices=['K562', 'GM12878', 'HepG2'])
parser.add_argument('--use_prm_signal', type=bool, help='use promoter signal', default=False)
parser.add_argument('--use_pretrained_encoder', type=bool, help='use pretrained encoder', default=False)
parser.add_argument('--rm_prm_seq', type=bool, help='remove promoter sequence', default=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_id)
os.makedirs('/dev/shm/data/', exist_ok=True)
# copy data 

results = []
expr_type = args.expr_type
batch_size = 50
max_n_enh = 60
dist_thr = 100_000
lr = 0.0001
model_type = args.model_type
use_pretrained_encoder = args.use_pretrained_encoder
cell_type = args.cell
# use_prm_signal = args.use_prm_signal
use_prm_signal = args.use_prm_signal
print('use_prm_signal:', use_prm_signal)
model_dist = {'EPInformer-abc': EPInformer_abc, 'EPInformer-v2': EPInformer_v2, 'EPInformer-abc-dist': EPInformer_abc_dist, 'EPInformer-abc-dist-v2':EPInformer_abc_dist_v2}
for fi in ['borzoi']:# range(1, 13):
    fold_i = 'fold_{}'.format(fi)
    for use_rna_feats, rm_prm_seq in [(True, args.rm_prm_seq)]:
        for cell in [cell_type]:
            if not os.path.exists('/dev/shm/data/{}_200CREs-gene_RPM_4feats.hdf5'.format(cell)):
                print('copying data into /dev/shm/')
                os.system('cp ./data/{}_200CREs-gene_RPM_4feats.hdf5 /dev/shm/data/'.format(cell))
                print(os.path.exists('/dev/shm/data/{}_200CREs-gene_RPM_4feats.hdf5'.format(cell)))
            for n_enh_feats in [args.n_enh_feats]:
                ds = promoter_enhancer_dataset(cell_type=cell, expr_type=expr_type, n_enh_feats=n_enh_feats, distance_thr=dist_thr, max_n_enh=max_n_enh, use_prm_signal=use_prm_signal, rm_prm_seq=rm_prm_seq)
                train_ensid = split_df[split_df[fold_i] == 'train'].index
                valid_ensid = split_df[split_df[fold_i] == 'valid'].index
                test_ensid = split_df[split_df[fold_i] == 'test'].index
                ensid_list = [eid.decode('utf-8') for eid in ds.data_h5['ensid'][:]]    
                ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
                ensid_df['idx'] = np.arange(len(ensid_list))
                ensid_df = ensid_df.set_index('ensid')
                train_idx = ensid_df.loc[train_ensid]['idx']
                valid_idx = ensid_df.loc[valid_ensid]['idx']
                test_idx = ensid_df.loc[test_ensid]['idx']
                train_ds = Subset(ds, train_idx)
                valid_ds = Subset(ds, valid_idx)
                test_ds = Subset(ds, test_idx)
                # Set up the model
                if use_pretrained_encoder:
                    print('Using pre-trained encoder')
                    pt_model_name = './pretrained_seqencoder_h3k27ac/fold_{}_best_enhancer_predictor_H3K27ac_256bp_{}_checkpoint.pt'.format(fi, cell)
                    checkpoint = torch.load(pt_model_name, weights_only=False)
                    pretrained_convNet = enhancer_predictor_256bp()
                    pretrained_convNet.load_state_dict(checkpoint['model_state_dict'])
                    model = model_dist[model_type](n_extraFeat=n_enh_feats,  pre_trained_encoder=pretrained_convNet.encoder, useFeat=use_rna_feats, out_dim=64, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal).to(device)
                    # freeze the encoder parameters
                    print('freezing the encoder parameters')
                    for name, value in model.named_parameters():
                        if name.startswith('seq_encoder'):
                            value.requires_grad = False
                else:
                    model = model_dist[model_type](n_extraFeat=n_enh_feats,  pre_trained_encoder=None, useFeat=use_rna_feats, out_dim=64, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal).to(device)
                # pt_model_name = '{}_seq2activityLog2_leaveChrOut_combinedRS_2bins_bs64_H3K27ac_adamW_erisxdl_r0'.format(cell)
                # checkpoint = torch.load("./trained_models/pretrained_enhancer_encoder/{}_best_{}_checkpoint.pt".format(fold_i, pt_model_name))
                # pretrained_convNet = enhancer_predictor_256bp() 
                # checkpoint = torch.load("./trained_models/EPInformer_PE_Activity/K562/fold_1_EPInformer_PE_Activity_CAGE_K562_checkpoint.pt")
                # model.load_state_dict(checkpoint['model_state_dict']) 
                # print('load model from', "./trained_models/EPInformer_PE_Activity/K562/fold_1_EPInformer_PE_Activity_CAGE_K562_checkpoint.pt")
                use_rna_feats_flag = 'rnafeats' if use_rna_feats else 'nornafeats'
                use_prm_signal_flag = 'prmsig' if use_prm_signal else 'nonprmsig'
                rm_prm_signal_flag = 'rmprmseq' if rm_prm_seq else 'nonrmprmseq'
                model.name = model.name + '.{}.{}.{}enhs.{}feats.{}.{}.{}.{}kb2TSS'.format(cell, expr_type, max_n_enh, n_enh_feats, use_rna_feats_flag, use_prm_signal_flag, rm_prm_signal_flag, str(int(dist_thr/1000)))
                # Test loading pre-trained EPInformer
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())  
                total_params = sum(np.prod(p.size()) for p in model_parameters)  
                print(cell, 'fold', fi, 'total', total_params/1_000_000, 'M params')
                print(model.name)
                saved_model_path = './EPInformer_models_20250629/'
                # Train the model
                train(model, train_ds, valid_dataset=valid_ds, learning_rate=lr, EPOCHS=50, model_name = model.name, fold_i=fi, batch_size=batch_size, device=device, saved_model_path=saved_model_path)
                # Test the model
                test_df = test(model, test_ds, model_name = model.name, saved_model_path=saved_model_path, fold_i=fi, batch_size=batch_size, device=device)
                test_df['cell'] = cell
                test_df['fold'] = fi
                test_df['use_rna_feats'] = use_rna_feats
                test_df['use_prm_signal'] = use_prm_signal_flag
                test_df['rm_prm_seq'] = rm_prm_signal_flag
                test_df['n_enh_feats'] = n_enh_feats
                results.append(test_df)

results_df = pd.concat(results)
results_df.to_csv('./EPInformer_predictions_20250629/{}_results.csv'.format(model.name), index=False)