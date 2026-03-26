import os
import sys
import argparse
from datetime import datetime
import os
import scripts.utils_forTraining as utils
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.model_selection import train_test_split
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

from Bio.Seq import Seq
import glob
import random
from EPInformer.models import enhancer_predictor_256bp
from epinformer_preprocessing import one_hot_encode

class dnase_dataset(Dataset):
    def __init__(self, cell_name, chrom_list, strand = 'both'):
        self.cell_name = cell_name 
        self.strand = strand
        dnase_df = pd.read_csv('/home/bingxing2/gpuuser926/project/EPInformer/data/enhancer_sequences/{}_peak_5bins_around_summit_activity_sequence.csv'.format(cell_name))
        dnase_df = dnase_df.rename(columns={'Offset_to_summit': 'Pos'})
        # dnase_df['Activity'] = np.sqrt(dnase_df['H3K27ac_1_RPM'] * dnase_df['DNase_RPM'])
        self.dnase_df = dnase_df[dnase_df['Chromosome'].isin(chrom_list)].reset_index(drop=True)
    def __len__(self):
        return len(self.dnase_df)

    def __getitem__(self, idx):
        sample = self.dnase_df.iloc[idx]
        activity = np.log2(0.1+sample['Activity'])
        # activity = sample['Activity']
        seq = sample['Sequence']
        seq_name = sample['Name'] + '_' + str(sample['Pos'])
        if self.strand == 'both':
            if random.random() > 0.5:
                seq = str(Seq(seq).reverse_complement())
        elif self.strand == 'reverse':
            seq = str(Seq(seq).reverse_complement())
        ohe_seq = one_hot_encode(seq)[None, :, :]
        return ohe_seq, activity, seq_name

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
        
def train(net, training_dataset, fold_i, saved_model_path='./models/', learning_rate=1e-4, model_logger=None, valid_dataset = None, model_name = '', batch_size = 64, device = 'cuda', EPOCHS=100, valid_size=1000):
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    if valid_dataset is not None:
        train_ds = training_dataset
        valid_ds = valid_dataset
    else:
        train_idx, val_idx = train_test_split(list(range(len(training_dataset))), test_size=valid_size, shuffle=True, random_state=66, stratify=stratify)
        train_ds = Subset(training_dataset, train_idx)
        valid_ds = Subset(training_dataset, val_idx)
    
    print("fold", fold_i ,"training data:", len(train_ds), "validated data:", len(valid_ds), 'total data:', len(training_dataset))
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    early_stopping = EarlyStopping(patience=5,
               verbose=True, path= saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt")

    L_expr = L1KLmixed() # nn.SmoothL1Loss()
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
            ohe_seq, y_expr, seq_name = data
            ohe_seq = ohe_seq.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr = net(ohe_seq)
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

def validate(net, valid_ds, batch_size=1024, device = 'cuda'):
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    net.eval()
    L_expr = L1KLmixed()
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in tqdm(validloader):
            ohe_seq, y_expr, seq_name = data
            ohe_seq = ohe_seq.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr = net(ohe_seq)
            loss_expr = L_expr(pred_expr, y_expr)
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
        checkpoint = torch.load(saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt", weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(model_name, fold_i, 'loaded!')
    net.eval()
    L_expr = L1KLmixed()
    with torch.no_grad():
        preds = []
        actual = []
        ensid_list = []
        for data in tqdm(testloader):
            ohe_seq, y_expr, seq_name = data
            ohe_seq = ohe_seq.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr = net(ohe_seq)
            loss_expr = L_expr(pred_expr, y_expr)
            outputs = list(pred_expr.flatten().cpu().detach().numpy())
            labels = list(y_expr.flatten().cpu().detach().numpy())
            preds += outputs
            actual += labels
            ensid_list += list(seq_name)

    slope, intercept, r_value, p_value, std_err = stats.linregress(preds, actual)
    peasonr, pvalue = stats.pearsonr(preds, actual)
    # mse = mean_squared_error(preds, actual)
    # print(fold %s test sequence: %0.3f' % (fold_i, r_value**2))\
    ensid_list = np.array(ensid_list).flatten()
    preds_df = pd.DataFrame({'preds': preds, 'actual': actual, 'ensid': ensid_list})
    preds_df['fold'] = fold_i
    # preds_df.to_csv(saved_model_path + "/fold_" + str(fold_i) + "_"+ model_name + "_predictions.csv")
    print('\nPearson R:', peasonr)
    return preds_df


split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
split_df['chrom'] = 'chr' + split_df['chrom']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cell', type=str, default='HepG2', help='cell type')
cells = parser.parse_args().cell
device = 'cuda'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if cells == 'all':
    cell_list = [ 'NHEK','HUVEC', 'HepG2', 'H1', 'GM12878', 'K562v2']
else:
    cell_list = [cells]
os.makedirs('./pretrained_seqencoder_h3k27ac_predictions/', exist_ok=True)
test_strand = 'reverse'
for cell in cell_list:
    cell_results = []
    for fi in range(1, 13):
        fold_i = 'fold_{}'.format(fi)
        # Set up the dataset
        train_chrom = list(split_df[split_df[fold_i] == 'train']['chrom'].unique())
        valid_chrom = list(split_df[split_df[fold_i] == 'valid']['chrom'].unique())
        test_chrom = list(split_df[split_df[fold_i] == 'test']['chrom'].unique())
        train_ds = dnase_dataset(cell, train_chrom, strand='both')
        valid_ds = dnase_dataset(cell, valid_chrom, strand='forward')
        test_ds = dnase_dataset(cell, test_chrom, strand=test_strand)
        model = enhancer_predictor_256bp().to(device)
        model.name = 'enhancer_predictor_log2_H3K27ac_256bp_{}'.format(cell)
        print('model name:', model.name)
        saved_model_path = './pretrained_seqencoder_h3k27ac/'
        os.makedirs(saved_model_path, exist_ok=True)
        # train(model, train_ds, valid_dataset=valid_ds, learning_rate=0.0005, EPOCHS=50, model_name = model.name, fold_i=fi, batch_size=256, device=device, saved_model_path=saved_model_path)
        # save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'fold_i': fi,
            'cell': cell
        }, saved_model_path + "/fold_" + str(fold_i) + "_last_"+ model.name+"_checkpoint.pt")
        # print('Model saved at:', saved_model_path + '/fold_{}_best_enhancer
        preds_df = test(model, test_ds, model_name = model.name, saved_model_path=saved_model_path, fold_i=fi, batch_size=128, device=device)
        preds_df['cell'] = cell
        cell_results.append(preds_df)
    cell_results_df = pd.concat(cell_results, axis=0)
    cell_results_df.to_csv('./pretrained_seqencoder_h3k27ac_predictions/enhancer_predictor_H3K27ac_256bp_{}_{}_predictions.csv'.format(cell, test_strand))
    peasonr, pvalue = stats.pearsonr(cell_results_df['preds'], cell_results_df['actual'])
    print(cell, len(cell_results_df), 'Pearson R:', peasonr)
    print('Results saved at:', './pretrained_seqencoder_h3k27ac_predictions/enhancer_predictor_H3K27ac_256bp_{}_{}_predictions.csv'.format(cell, test_strand))