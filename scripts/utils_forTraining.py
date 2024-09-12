# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import pandas as pd
# torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim
import torch.utils.data as data_utils
from torch.utils.data import Subset, Dataset

import sys
import argparse

from scipy import stats
# import sklearn
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# logging
from tqdm import tqdm
# from model.EPInformer import EPInformer_v2, enhancer_predictor_256bp
import h5py
import glob

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

def train(net, training_dataset, fold_i, saved_model_path='../models', learning_rate=5e-4, model_logger=None, fixed_encoder = False, n_enhancers = 50, valid_dataset = None, model_name = '', batch_size = 64, device = 'cuda', stratify=None, class_weight=None, EPOCHS=100, valid_size=1000):
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
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    early_stopping = EarlyStopping(patience=3,
               verbose=True, path= saved_model_path + "/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt")

    L_expr = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print('Model name:', net.name)
    lrs = []
    # last_loss = None
    net.train()
    for epoch in range(EPOCHS):
        net.train()
        print('learning rate:', get_lr(optimizer))
        running_loss = 0
        loss_e = 0
        # print('model training mode is:', net.training)
        for data in tqdm(trainloader):
            # print(inputs.size())
            optimizer.zero_grad()
            input_PE, input_feat, input_dist, y_expr, eid = data
            input_PE = input_PE.float().to(device)
            input_feat = input_feat.float().to(device)
            # input_dist = input_dist.long().to(device)
            input_dist = input_dist.float().to(device)
            # input_PEmask = ~(input_PE.sum(-1).sum(-1) > 0).bool().to(device)
            y_expr = y_expr.float().to(device)
            # print(input_P.shape, input_E.shape, input_Emask.shape)
            # print(input_dist.shape, input_dist)
            # if net_type == 'seq_feat':
            #     pred_expr, _ = net(input_PE, input_feat)
            # elif net_type == 'seq':
            #     pred_expr, _ = net(input_PE)
            # elif net_type == 'seq_feat_dist':
            pred_expr, _ = net(input_PE, input_feat, input_dist)
            loss_expr = L_expr(pred_expr, y_expr)
            loss_e += loss_expr.item()

            loss = loss_expr# + loss_intensity + loss_contact
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
            running_loss += loss.item()

        print('[Epoch %d] loss: %.9f' %
                      (epoch + 1, running_loss/len(trainloader)))
        print('Training Loss: expression loss:', loss_e/len(trainloader))
        # log_cols = ['Epoch', 'Training_Loss', 'Validation_Loss', 'Validation_PearsonR_allGene',
        #             'Validation_R2_allGene', 'Validation_PearsonR_weGene', 'Validation_R2_weGene', 'Saved?']


        val_mse_all, val_r2_all, val_pr_all = validate(net, valid_ds, n_enhancers=n_enhancers, device=device)
        val_r2 = val_r2_all
        val_pr_wE, val_r2_wE = val_pr_all, val_r2_all
        print('Valdaition R square all:', val_r2_all)
        early_stopping(-val_r2, net, epoch)
        if model_logger is not None:
            label_type = net.name.split('.')[-1]
            model_logger.add([fold_i, epoch, running_loss/len(trainloader), val_mse_all, val_pr_all, val_r2_all, val_pr_wE, val_r2_wE, early_stopping.counter, label_type])
            # model_logger.save("./EPInfomrer_log/{}.crossValid.log".format(net.name.replace('.'+label_type, '')))
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return lrs

def validate(net, valid_ds,  net_type = 'seq_feat_dist', n_enhancers=50, batch_size=16, device = 'cuda'):
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    net.eval()
    L_expr = nn.MSELoss()
    
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in validloader:
            # print(inputs.size())
            input_PE, input_feat, input_dist, y_expr, eid = data
            input_PE = input_PE.float().to(device)
            input_feat = input_feat.float().to(device)
            # input_dist = input_dist.long().to(device)
            input_dist = input_dist.float().to(device)
            # print(input_dist.shape, input_dist)
            # input_PEmask = ~(input_PE.sum(-1).sum(-1) > 0).bool().to(device)
            y_expr = y_expr.float().to(device)
            # print(input_P.shape, input_E.shape, input_Emask.shape)
            pred_expr, _ = net(input_PE, input_feat, input_dist)

            outputs = list(pred_expr.flatten().cpu().detach().numpy())
            labels = list(y_expr.flatten().cpu().detach().numpy())

            loss_expr = L_expr(pred_expr, y_expr)
            loss_e += loss_expr.item()
            preds += outputs
            actual += labels

    slope, intercept, r_value, p_value, std_err = stats.linregress(preds, actual)
    peasonr, pvalue = stats.pearsonr(preds, actual)
    mse = mean_squared_error(preds, actual)
    print('Validation loss expression loss:', loss_e/len(validloader))
    print("valid: mse", mse, "R_sqaure", r_value**2, 'peasonr', peasonr)
    return mse, r_value**2, peasonr

def test(net, test_ds, fold_i, model_name = None, saved_model_path=None, batch_size=64, device = 'cuda', model_type='best'):
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    # checkpoint = torch.load(saved_model_path + "/fold_" + str(fold_i) + "_"+model_name+"_checkpoint.pt")
    # net.load_state_dict(checkpoint['model_state_dict'])
    # except:
    # net = nn.DataParallel(net, device_ids=[0,1])
    # net.load_state_dict(checkpoint['model_state_dict'])
    # net.load_state_dict(torch.load("./K562_10crx_models/fold_" + str(fold_i) + "_best_"+model_name+"_checkpoint.pt"))
    # print("Load the best model from fold_" + str(fold_i) + "_"+model_type+"_"+model_name+"_checkpoint.pt", )
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
            input_PE, input_feat, input_dist, y_expr, eid = data
            input_PE = input_PE.float().to(device)
            input_feat = input_feat.float().to(device)
            # input_dist = input_dist.long().to(device)
            input_dist = input_dist.float().to(device)
            # input_PEmask = ~(input_PE.sum(-1).sum(-1) > 0).bool().to(device)
            y_expr = y_expr.float().to(device)
            # print(input_P.shape, input_E.shape, input_Emask.shape)
            pred_expr, _ = net(input_PE, input_feat, input_dist)

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

class promoter_enhancer_dataset(Dataset):
    def __init__(self, data_folder = '/content/drive/MyDrive/EPInformer/github/EPInformer/data/', expr_type='CAGE', usePromoterSignal=True, first_signal='distance', signal_type='H3K27ac', cell_type='K562', distance_threshold=None, hic_threshold=None, n_enhancers=50, n_extraFeat=1):
        self.expr_type = expr_type
        self.cell_type = cell_type
        self.data_folder = data_folder
        self.first_signal = first_signal
        self.n_enhancers = n_enhancers
        self.signal_type = signal_type
        self.n_extraFeat = n_extraFeat
        self.usePromoterSignal = usePromoterSignal
        self.distance_threshold = distance_threshold
        self.hic_threshold = hic_threshold
        if cell_type == 'K562':
            # promoter_df = pd.read_csv('/content/drive/MyDrive/EPInformer/EPInformer_activity/data/K562/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
            promoter_df = pd.read_csv(self.data_folder + '/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
            promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
            self.promoter_df = promoter_df
            self.data_h5 = h5py.File(self.data_folder + '/K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5', 'r')
            # self.data_h5 = h5py.File('/content/drive/MyDrive/EPInformer/EPInformer_activity/data/K562/K562_DNase_ENCFF257HEE_2kb_noCutOff_hic_noFlankSeq_150kb60e_AllPutative_signals_False_v2.h5')
        elif cell_type == 'GM12878':
            promoter_df = pd.read_csv(self.data_folder + '/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
            promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
            self.promoter_df = promoter_df 
            self.data_h5 = h5py.File(self.data_folder + '/GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5', 'r')
        self.expr_df = pd.read_csv(self.data_folder + '/GM12878_K562_18377_gene_expr_fromXpresso.csv', index_col='ENSID')
    def __len__(self):
        return len(self.data_h5['ensid'])

    def __getitem__(self, idx):
        sample_ensid = self.data_h5['ensid'][idx].decode()
        seq_code = self.data_h5['pe_code'][idx]
        enhancer_distance = self.data_h5['distance'][idx,1:]
        enhancer_intensity = self.data_h5['activity'][idx,1:]
        enhancer_contact = self.data_h5['hic'][idx,1:]

        if self.signal_type == 'H3K27ac':
            promoter_activity = self.promoter_df.loc[sample_ensid]['PromoterActivity']
        elif self.signal_type == 'DNase':
            promoter_activity = self.promoter_df.loc[sample_ensid]['normalized_dhs']
            # enhancer_intensity = dhs_intensity
        promoter_code = seq_code[:1]
        enhancers_code = seq_code[1:]

        rnaFeat = list(self.expr_df.loc[sample_ensid][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore','UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float))
        pe_activity = np.concatenate([[0], enhancer_intensity]).flatten()

        if self.usePromoterSignal and self.n_extraFeat > 1:
            rnaFeat = np.array(rnaFeat + [promoter_activity])
        else:
            rnaFeat = np.array(rnaFeat + [0])

        if self.distance_threshold is not None:
            enhancer_distance = enhancer_distance.flatten()
            enhancers_zero = np.zeros_like(enhancers_code)
            enhancers_zero[abs(enhancer_distance) < self.distance_threshold] = enhancers_code[abs(enhancer_distance) < self.distance_threshold]
            enhancers_code = enhancers_zero

            enhancer_distance_zero = np.zeros_like(enhancer_distance)
            enhancer_distance_zero[abs(enhancer_distance) < self.distance_threshold] = enhancer_distance[abs(enhancer_distance) < self.distance_threshold]
            enhancer_distance = enhancer_distance_zero

        if self.hic_threshold is not None:
            enhancer_contact = enhancer_contact.flatten()
            enhancers_zero = np.zeros_like(enhancers_code)
            enhancers_zero[enhancer_contact > self.hic_threshold] = enhancers_code[enhancer_contact > self.hic_threshold]
            enhancers_code = enhancers_zero

            enhancer_contact_zero = np.zeros_like(enhancer_contact)
            enhancer_contact_zero[enhancers_code[enhancer_contact > self.hic_threshold ]] = enhancer_contact[enhancers_code[enhancer_contact > self.hic_threshold]]
            enhancer_contact = enhancer_contact_zero

        pe_hic = np.concatenate([[0], enhancer_contact]).flatten()
        pe_hic = np.log10(1+pe_hic)
        pe_distance = np.concatenate([[0], enhancer_distance/1000]).flatten()
        # print(pe_distance)
        if self.n_extraFeat == 1:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis]],axis=-1)
        elif self.n_extraFeat == 2:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis], pe_activity[:,np.newaxis]],axis=-1)
        elif self.n_extraFeat == 3:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis], pe_hic[:,np.newaxis], pe_activity[:,np.newaxis], ],axis=-1)
        else:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis]],axis=-1)

        promoter_code_tensor = torch.from_numpy(promoter_code).float()
        pe_feat_tensor = torch.from_numpy(pe_feat[:self.n_enhancers+1])
        if self.n_extraFeat == 0: # Use promoter only
            enhancers_code = np.zeros_like(enhancers_code[:self.n_enhancers, :])
        enhancers_code_tensor = torch.from_numpy(enhancers_code[:self.n_enhancers, :]).float()
        pe_code_tensor = torch.concat([promoter_code_tensor, enhancers_code_tensor])
        rnaFeat_tensor = torch.from_numpy(rnaFeat).float()
        # print(pe_distance_tensor)

        if self.expr_type == 'CAGE':
            cage_expr = np.log10(self.expr_df.loc[sample_ensid][self.cell_type + '_CAGE_128*3_sum']+1)
            expr_tensor = torch.from_numpy(np.array([cage_expr])).float()
        elif self.expr_type == 'RNA':
            rna_expr = self.expr_df.loc[sample_ensid]['Actual_' + self.cell_type]
            expr_tensor = torch.from_numpy(np.array([rna_expr])).float()
        else:
            assert False, 'label not exists!'
        return pe_code_tensor, rnaFeat_tensor, pe_feat_tensor, expr_tensor, sample_ensid
