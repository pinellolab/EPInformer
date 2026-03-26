import os
import sys
import argparse
from datetime import datetime
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
import warnings
import torch.nn.functional as F

import torch.utils.data as data_utils
from torch.utils.data import Subset, Dataset, DataLoader
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from EPInformer.models_abc import EPInformer_abc, EPInformer_v2, EPInformer_abc_dist, EPInformer_abc_dist_v2, enhancer_predictor_256bp


class promoter_enhancer_dataset(Dataset):
    """Dataset that reads from the factored HDF5 format (samples.h5).

    The HDF5 contains:
      Gene-level:
        - promoter_seq:  (N, L, 4)     promoter one-hot sequences
        - gene_enh_idx:  (N, n_enh)    index into enhancer_seq (-1 = padding)
        - activity:      (N, n_enh)    enhancer activity_base
        - dhs:           (N, n_enh)    normalized DNase signal
        - distance:      (N, n_enh)    distance_relative to TSS
        - contact:       (N, n_enh)    HiC contact
        - ensid:         (N,)          gene IDs
      Enhancer-level:
        - enhancer_seq:  (M, L, 4)     unique enhancer one-hot sequences
        - enhancer_name: (M,)          enhancer element IDs
    """

    def __init__(self, h5_path, expr_csv, cell_type='K562', expr_type='RNA',
                 n_enh_feats=3, disable_enh=False, distance_thr=None,
                 max_n_enh=60, use_prm_signal=False, rm_prm_seq=False,
                 rm_self_promoter=False):
        self.data_h5 = h5py.File(h5_path, 'r')
        self.expr_df = pd.read_csv(expr_csv, index_col='gene_id')
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        self.rm_prm_seq = rm_prm_seq
        self.rm_self_promoter = rm_self_promoter

        # Preload all data into shared-memory tensors (zero-copy across DataLoader workers)
        print('Loading HDF5 data into shared memory...')
        self._ensid = [e.decode() if isinstance(e, bytes) else e for e in self.data_h5['ensid'][:]]
        self._promoter_seq = torch.from_numpy(self.data_h5['promoter_seq'][:]).share_memory_()
        self._gene_enh_idx = torch.from_numpy(self.data_h5['gene_enh_idx'][:].astype(np.int64)).share_memory_()
        self._enhancer_seq = torch.from_numpy(self.data_h5['enhancer_seq'][:]).share_memory_()
        self._distance = torch.from_numpy(self.data_h5['distance'][:].astype(np.float64)).share_memory_()
        self._activity = torch.from_numpy(self.data_h5['activity'][:].astype(np.float64)).share_memory_()
        self._contact = torch.from_numpy(self.data_h5['contact'][:].astype(np.float64)).share_memory_()
        self.data_h5.close()

        # Precompute per-gene expression features (avoids slow pandas .loc in workers)
        rna_cols = ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
                    'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        rna_list = []
        expr_list = []
        for ensid in self._ensid:
            row = self.expr_df.loc[ensid]
            rna = row[rna_cols].values.astype(np.float64).flatten()
            if use_prm_signal:
                rna = np.concatenate([rna, [0.0]])
            rna_list.append(rna)
            if expr_type == 'CAGE':
                expr_list.append(float(np.log10(row[cell_type + '_CAGE_128*3_sum'] + 1)))
            else:
                expr_list.append(float(row['Actual_' + cell_type]))
        self._rna_feats = torch.from_numpy(np.array(rna_list)).share_memory_()
        self._expr = torch.from_numpy(np.array(expr_list, dtype=np.float64)).share_memory_()

        print(f'Loaded: {len(self._ensid)} genes, {self._enhancer_seq.shape[0]} enhancers')

    def __len__(self):
        return len(self._ensid)

    def __getitem__(self, idx):
        sample_ensid = self._ensid[idx]

        # Promoter sequence (shared-memory tensor → numpy)
        prm_seq = self._promoter_seq[idx].numpy().astype(np.float64)  # (L, 4)

        # Gather enhancer sequences by index (torch indexing on shared memory)
        enh_indices = self._gene_enh_idx[idx]  # (n_enh,)
        valid_mask = enh_indices >= 0
        L = prm_seq.shape[0]
        enh_seqs = np.zeros((self.max_n_enh, L, 4), dtype=np.float64)
        if valid_mask.any():
            enh_seqs[valid_mask.numpy()] = self._enhancer_seq[enh_indices[valid_mask]].numpy()

        # Stack promoter + enhancers
        pe_ohe = np.concatenate([prm_seq[np.newaxis], enh_seqs], axis=0)  # (1+n_enh, L, 4)

        # Build enhancer features: [abs(distance), activity, contact]
        dist = self._distance[idx].numpy()
        activity = self._activity[idx].numpy()
        contact = self._contact[idx].numpy()

        if self.n_enh_feats == 0:
            enh_feats = np.zeros((dist.shape[0], 1), dtype=np.float32)
        else:
            enh_feats = np.stack([np.abs(dist), activity, contact], axis=1)[:, :self.n_enh_feats]

        # Precomputed mRNA sequence features + expression
        rna_feats = self._rna_feats[idx].numpy()

        # Distance thresholding
        if self.distance_thr is not None:
            n_enh = pe_ohe.shape[0] - 1
            enh_ohe = pe_ohe[1:]
            enh_ohe_new = np.zeros((self.max_n_enh, pe_ohe.shape[1], 4), dtype=np.float32)
            enh_feats_new = np.zeros((self.max_n_enh, enh_feats.shape[-1]), dtype=np.float32)
            min_d = 1000 if (self.rm_prm_seq or self.rm_self_promoter) else 0
            new_i = 0
            for i in range(n_enh):
                d = abs(dist[i]) if i < len(dist) else 0
                if d <= self.distance_thr and d >= min_d:
                    enh_ohe_new[new_i] = enh_ohe[i]
                    enh_feats_new[new_i] = enh_feats[i]
                    new_i += 1
                if new_i >= self.max_n_enh:
                    break
            pe_ohe = np.concatenate([pe_ohe[[0]], enh_ohe_new], axis=0)
            enh_feats = enh_feats_new

        if self.disable_enh:
            pe_ohe[1:] = 0
            enh_feats[:] = 0

        # Expression target (precomputed)
        expr = self._expr[idx].item()

        # Promoter features row (ones) prepended to enhancer features
        prm_feats = np.ones_like(enh_feats[[0]])
        pe_feats = np.concatenate([prm_feats, enh_feats], axis=0)

        return pe_ohe, rna_feats, pe_feats, expr, sample_ensid


class promoter_enhancer_dataset_legacy(Dataset):
    """Dataset that reads the old flat HDF5 format (e.g. K562_200CREs-gene_RPM_4feats.hdf5).

    HDF5 contains:
      - enhancers_ohe:  (N, 200, 2000, 4)  per-gene enhancer one-hot sequences
      - enhancers_feat: (N, 200, 5)         [distance, feat1, feat2, feat3, feat4]
      - promoter_ohe:   (N, 1000, 4)        1kb promoter one-hot (zero-padded to 2kb)
      - ensid:          (N,)                gene IDs
      - expr:           (N, 1)              expression values (optional, can use CSV)

    Features extracted as: [abs(distance), feat[3], feat[-1]][:, :n_enh_feats]
    """

    def __init__(self, h5_path, expr_csv, cell_type='K562', expr_type='RNA',
                 n_enh_feats=3, disable_enh=False, distance_thr=None,
                 max_n_enh=60, use_prm_signal=False, rm_prm_seq=False):
        self.expr_df = pd.read_csv(expr_csv, index_col='gene_id')
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        self.rm_prm_seq = rm_prm_seq

        # Preload all data into memory for fast training
        print('Loading legacy HDF5 data into memory...')
        f = h5py.File(h5_path, 'r')
        self._ensid = [e.decode() if isinstance(e, bytes) else e for e in f['ensid'][:]]
        self._enhancers_feat = f['enhancers_feat'][:]  # (N, 200, 5)
        self._enhancers_ohe = f['enhancers_ohe'][:].astype(np.float32)  # (N, 200, 2000, 4)
        # Load 1kb promoter from HDF5 and zero-pad to 2kb (500bp each side)
        prm_1k = f['promoter_ohe'][:].astype(np.float32)  # (N, 1000, 4)
        self._promoter_ohe = np.pad(prm_1k, ((0, 0), (500, 500), (0, 0)), mode='constant')  # (N, 2000, 4)
        f.close()
        print(f'Loaded: {len(self._ensid)} genes, {self._enhancers_feat.shape[1]} CREs per gene, '
              f'enhancers_ohe: {self._enhancers_ohe.nbytes/1e9:.1f} GB, '
              f'promoter_ohe: {self._promoter_ohe.nbytes/1e9:.1f} GB')

    def __len__(self):
        return len(self._ensid)

    def __getitem__(self, idx):
        sample_ensid = self._ensid[idx]

        # Promoter: preloaded 2kb one-hot
        prm_ohe = self._promoter_ohe[idx][np.newaxis, :]  # (1, 2000, 4)

        # Enhancer sequences: preloaded
        enh_ohe = self._enhancers_ohe[idx]  # (200, 2000, 4)

        # Extract features: [abs(distance), feat[3], feat[-1]]
        raw_feats = self._enhancers_feat[idx]  # (200, 5)
        if self.n_enh_feats == 0:
            enh_feats = np.zeros_like(np.concatenate(
                [abs(raw_feats[:, [0]]), raw_feats[:, [3]], raw_feats[:, [-1]]], axis=1)[:, :1])
        else:
            enh_feats = np.concatenate(
                [abs(raw_feats[:, [0]]), raw_feats[:, [3]], raw_feats[:, [-1]]], axis=1)[:, :self.n_enh_feats]

        # mRNA sequence features
        rna_feats = np.array(self.expr_df.loc[sample_ensid][
            ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
             'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        ].values.astype(float)).flatten()

        if self.use_prm_signal:
            rna_feats = np.concatenate([rna_feats, np.array([0.0])])

        # Distance thresholding — filter to nearest max_n_enh enhancers
        if self.distance_thr is not None:
            enh_ohe_new = np.zeros((self.max_n_enh, 2000, 4), dtype=np.float32)
            enh_feats_new = np.zeros((self.max_n_enh, enh_feats.shape[-1]), dtype=np.float32)
            new_i = 0
            for i in range(enh_ohe.shape[0]):
                d = abs(enh_feats[i][0]) if i < len(enh_feats) else 0
                if not self.rm_prm_seq:
                    if d <= self.distance_thr:
                        enh_ohe_new[new_i] = enh_ohe[i]
                        enh_feats_new[new_i] = enh_feats[i]
                        new_i += 1
                else:
                    if d <= self.distance_thr and d >= 1000:
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

        # Expression target
        if self.expr_type == 'CAGE':
            expr = float(np.log10(self.expr_df.loc[sample_ensid, self.cell_type + '_CAGE_128*3_sum'] + 1))
        else:
            expr = float(self.expr_df.loc[sample_ensid, 'Actual_' + self.cell_type])

        pe_ohe = np.concatenate([prm_ohe, enh_ohe], axis=0)
        pe_feats = np.concatenate([np.zeros_like(enh_feats[[0]]), enh_feats], axis=0)

        return pe_ohe, rna_feats, pe_feats, expr, sample_ensid


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Logger():
    def __init__(self, names, verbose=False):
        self.names = names
        self.verbose = verbose

    def start(self):
        self.data = {name: [] for name in self.names}
        if self.verbose:
            print("\t".join(self.names))

    def add(self, row):
        assert len(row) == len(self.names)
        for name, value in zip(self.names, row):
            self.data[name].append(value)
        if self.verbose:
            print("\t".join(map(str, [round(x, 4) if isinstance(x, float) else x
                for x in row])))

    def save(self, name):
        pd.DataFrame(self.data).to_csv(name, sep='\t', index=False)

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', hparams=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.hparams = hparams or {}

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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
                'hparams': self.hparams,
                },
                self.path)
        print('Saving ckpt at', self.path)
        self.val_loss_min = val_loss

def train(net, training_dataset, fold_i, saved_model_path='./models/', learning_rate=1e-4, model_logger=None, fixed_encoder=False, valid_dataset=None, model_name='', batch_size=64, device='cuda', stratify=None, class_weight=None, EPOCHS=100, valid_size=1000, hparams=None):
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path, exist_ok=True)
    if valid_dataset is not None:
        train_ds = training_dataset
        valid_ds = valid_dataset
    else:
        train_idx, val_idx = train_test_split(list(range(len(training_dataset))), test_size=valid_size, shuffle=True, random_state=66, stratify=stratify)
        train_ds = Subset(training_dataset, train_idx)
        valid_ds = Subset(training_dataset, val_idx)

    if fixed_encoder:
        print('fixed parameter of encoder')
        for name, value in net.named_parameters():
            if name.startswith('seq_encoder'):
                value.requires_grad = False

    print("fold", fold_i, "training data:", len(train_ds), "validated data:", len(valid_ds), 'total data:', len(training_dataset))
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    early_stopping = EarlyStopping(patience=5,
               verbose=True, path=saved_model_path + "/fold_" + str(fold_i) + "_best_" + model_name + "_checkpoint.pt",
               hparams=hparams)

    # Per-fold training log
    fold_log = Logger(['epoch', 'lr', 'train_loss', 'val_mse', 'val_r2', 'val_pearsonr', 'early_stop_counter'])
    fold_log.start()

    L_expr = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    lrs = []
    for epoch in range(EPOCHS):
        net.train()
        cur_lr = get_lr(optimizer)
        print('learning rate:', cur_lr)
        running_loss = 0
        loss_e = 0
        for data in tqdm(trainloader):
            optimizer.zero_grad()
            pe_seqs, rna_feats, enh_feats, y_expr, eid = data
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
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        print('[Epoch %d] loss: %.9f' % (epoch + 1, train_loss))
        print('Training Loss: expression loss:', loss_e / len(trainloader))
        val_mse_all, val_r2_all, val_pr_all = validate(net, valid_ds, device=device)
        val_r2 = val_r2_all
        val_pr_wE, val_r2_wE = val_pr_all, val_r2_all
        print('Validation R square all:', val_r2_all)
        early_stopping(-val_r2, net, epoch)

        # Log this epoch
        fold_log.add([epoch + 1, cur_lr, train_loss, val_mse_all, val_r2_all, val_pr_all, early_stopping.counter])

        if model_logger is not None:
            label_type = net.name.split('.')[-1]
            model_logger.add([fold_i, epoch, train_loss, val_mse_all, val_pr_all, val_r2_all, val_pr_wE, val_r2_wE, early_stopping.counter, label_type])
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save per-fold training log
    log_path = os.path.join(saved_model_path, f'fold_{fold_i}_training_log.csv')
    fold_log.save(log_path)
    print(f'Training log saved to {log_path}')

    return lrs

def validate(net, valid_ds, batch_size=16, device='cuda'):
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers=4, persistent_workers=True)
    net.eval()
    L_expr = nn.SmoothL1Loss()
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
    print('Validation loss expression loss:', loss_e / len(validloader))
    print("valid: mse", mse, "R_sqaure", r_value**2, 'peasonr', peasonr)
    return mse, r_value**2, peasonr

def test(net, test_ds, fold_i, model_name=None, saved_model_path=None, batch_size=64, device='cuda', model_type='best'):
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=4)
    if saved_model_path is not None:
        checkpoint = torch.load(saved_model_path + "/fold_" + str(fold_i) + "_best_" + model_name + "_checkpoint.pt", weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(model_name, 'loaded!')
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
    print('\nPearson R:', peasonr)
    sys.stdout.flush()
    df = pd.DataFrame(index=np.array(ensid_list).flatten())
    df['Pred'] = preds
    df['actual'] = actual
    df['fold_idx'] = fold_i
    pearsonr_we, pvalue = stats.pearsonr(df['Pred'], df['actual'])
    print('PearsonR:', pearsonr_we)
    if saved_model_path is not None:
        df.to_csv(saved_model_path + "/fold_" + str(fold_i) + "_" + model_name + "_predictions.csv")
    return df, {'fold': fold_i, 'pearsonr': peasonr, 'r2': r_value**2, 'mse': mse, 'n_test': len(preds)}


# ── CLI & main loop ──────────────────────────────────────────────────────────

if __name__ == '__main__':

    split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, help='cuda id', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default='EPInformer-v2', choices=['EPInformer-v2', 'EPInformer-abc', 'EPInformer-abc-dist', 'EPInformer-abc-dist-v2'])
    parser.add_argument('--expr_type', type=str, help='expression type', default='RNA', choices=['CAGE', 'RNA'])
    parser.add_argument('--n_enh_feats', type=int, help='number of enhancer features', default=3, choices=[0, 1, 2, 3])
    parser.add_argument('--cell', type=str, help='cell type', default='K562')
    parser.add_argument('--use_prm_signal', action='store_true', help='use promoter signal')
    parser.add_argument('--use_pretrained_encoder', action='store_true', help='use pretrained encoder')
    parser.add_argument('--rm_prm_seq', action='store_true', help='remove promoter sequence')
    parser.add_argument('--h5_path', type=str, default='./training_data/k562_run/samples.h5', help='path to preprocessed HDF5')
    parser.add_argument('--expr_csv', type=str, default='./data_EPInformer/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv', help='gene expression CSV')
    parser.add_argument('--output_dir', type=str, default='./EPInformer_models/', help='directory for saved models and predictions')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--rm_self_promoter', action='store_true', help='remove self-promoter elements (distance < 1000bp) at training time')
    args = parser.parse_args()

    if device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    results = []
    fold_metrics = []
    expr_type = args.expr_type
    batch_size = args.batch_size
    max_n_enh = 60
    dist_thr = 100_000
    lr = 0.0001
    model_type = args.model_type
    use_pretrained_encoder = args.use_pretrained_encoder
    cell_type = args.cell
    use_prm_signal = args.use_prm_signal
    saved_model_path = args.output_dir
    os.makedirs(saved_model_path, exist_ok=True)

    print('device:', device)
    print('use_prm_signal:', use_prm_signal)
    model_dist = {'EPInformer-abc': EPInformer_abc, 'EPInformer-v2': EPInformer_v2, 'EPInformer-abc-dist': EPInformer_abc_dist, 'EPInformer-abc-dist-v2': EPInformer_abc_dist_v2}

    # Auto-detect HDF5 format
    with h5py.File(args.h5_path, 'r') as _f:
        is_legacy = 'enhancers_ohe' in _f
    print(f'HDF5 format: {"legacy (flat)" if is_legacy else "factored"}')

    for fi in range(1, 13):
        fold_i = 'fold_{}'.format(fi)
        for use_rna_feats, rm_prm_seq in [(True, args.rm_prm_seq)]:
            for cell in [cell_type]:
                for n_enh_feats in [args.n_enh_feats]:
                    if is_legacy:
                        ds = promoter_enhancer_dataset_legacy(
                            h5_path=args.h5_path,
                            expr_csv=args.expr_csv,
                            cell_type=cell,
                            expr_type=expr_type,
                            n_enh_feats=n_enh_feats,
                            distance_thr=dist_thr,
                            max_n_enh=max_n_enh,
                            use_prm_signal=use_prm_signal,
                            rm_prm_seq=rm_prm_seq,
                        )
                    else:
                        ds = promoter_enhancer_dataset(
                            h5_path=args.h5_path,
                            expr_csv=args.expr_csv,
                            cell_type=cell,
                            expr_type=expr_type,
                            n_enh_feats=n_enh_feats,
                            distance_thr=dist_thr,
                            max_n_enh=max_n_enh,
                            use_prm_signal=use_prm_signal,
                            rm_prm_seq=rm_prm_seq,
                            rm_self_promoter=args.rm_self_promoter,
                        )
                    train_ensid = split_df[split_df[fold_i] == 'train'].index
                    valid_ensid = split_df[split_df[fold_i] == 'valid'].index
                    test_ensid = split_df[split_df[fold_i] == 'test'].index
                    if is_legacy:
                        ensid_list = ds._ensid
                    else:
                        ensid_list = [eid.decode('utf-8') for eid in ds.data_h5['ensid'][:]]
                    ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
                    ensid_df['idx'] = np.arange(len(ensid_list))
                    ensid_df = ensid_df.set_index('ensid')
                    # Only keep ENSIDs that exist in both split_df and HDF5
                    train_idx = ensid_df.loc[ensid_df.index.intersection(train_ensid)]['idx']
                    valid_idx = ensid_df.loc[ensid_df.index.intersection(valid_ensid)]['idx']
                    test_idx = ensid_df.loc[ensid_df.index.intersection(test_ensid)]['idx']
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
                        model = model_dist[model_type](n_extraFeat=n_enh_feats, pre_trained_encoder=pretrained_convNet.encoder, useFeat=use_rna_feats, out_dim=64, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal).to(device)
                        print('freezing the encoder parameters')
                        for name, value in model.named_parameters():
                            if name.startswith('seq_encoder'):
                                value.requires_grad = False
                    else:
                        model = model_dist[model_type](n_extraFeat=n_enh_feats, pre_trained_encoder=None, useFeat=use_rna_feats, out_dim=64, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal).to(device)
                    use_rna_feats_flag = 'rnafeats' if use_rna_feats else 'nornafeats'
                    use_prm_signal_flag = 'prmsig' if use_prm_signal else 'nonprmsig'
                    rm_prm_signal_flag = 'rmprmseq' if rm_prm_seq else 'nonrmprmseq'
                    model.name = model.name + '.{}.{}.{}enhs.{}feats.{}.{}.{}.{}kb2TSS'.format(cell, expr_type, max_n_enh, n_enh_feats, use_rna_feats_flag, use_prm_signal_flag, rm_prm_signal_flag, str(int(dist_thr / 1000)))
                    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                    total_params = sum(np.prod(p.size()) for p in model_parameters)
                    print(cell, 'fold', fi, 'total', total_params / 1_000_000, 'M params')
                    print(model.name)
                    # Train
                    hparams = {
                        'model_type': model_type,
                        'cell': cell,
                        'expr_type': expr_type,
                        'n_enh_feats': n_enh_feats,
                        'max_n_enh': max_n_enh,
                        'dist_thr': dist_thr,
                        'use_rna_feats': use_rna_feats,
                        'use_prm_signal': use_prm_signal,
                        'rm_prm_seq': rm_prm_seq,
                        'rm_self_promoter': args.rm_self_promoter,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'epochs': args.epochs,
                        'h5_path': args.h5_path,
                    }
                    train(model, train_ds, valid_dataset=valid_ds, learning_rate=lr, EPOCHS=args.epochs, model_name=model.name, fold_i=fi, batch_size=batch_size, device=device, saved_model_path=saved_model_path, hparams=hparams)
                    # Test
                    test_df, metrics = test(model, test_ds, model_name=model.name, saved_model_path=saved_model_path, fold_i=fi, batch_size=batch_size, device=device)
                    test_df['cell'] = cell
                    test_df['fold'] = fi
                    test_df['use_rna_feats'] = use_rna_feats
                    test_df['use_prm_signal'] = use_prm_signal_flag
                    test_df['rm_prm_seq'] = rm_prm_signal_flag
                    test_df['n_enh_feats'] = n_enh_feats
                    results.append(test_df)
                    metrics['n_train'] = len(train_idx)
                    metrics['n_valid'] = len(valid_idx)
                    fold_metrics.append(metrics)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(saved_model_path, '{}_results.csv'.format(model.name)), index=False)

    # Save fold summary
    summary_df = pd.DataFrame(fold_metrics)
    summary_path = os.path.join(saved_model_path, 'fold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\nFold summary saved to {summary_path}')
    print(summary_df.to_string(index=False))
    print(f'\nMean Pearson R: {summary_df["pearsonr"].mean():.4f} +/- {summary_df["pearsonr"].std():.4f}')
