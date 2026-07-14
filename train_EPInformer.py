import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from scipy import stats
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

# Workaround for numpy 2.x + torch < 2.4 incompatibility
# torch.from_numpy fails with "expected np.ndarray (got numpy.ndarray)"
_NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32, np.float64: torch.float64,
    np.int32: torch.int32, np.int64: torch.int64, np.bool_: torch.bool,
}
_orig_from_numpy = torch.from_numpy
def _from_numpy_compat(arr):
    try:
        return _orig_from_numpy(arr)
    except TypeError:
        arr = np.ascontiguousarray(arr)
        td = _NP_TO_TORCH_DTYPE.get(arr.dtype.type)
        if td is None:
            raise
        return torch.frombuffer(arr, dtype=td).reshape(arr.shape)
torch.from_numpy = _from_numpy_compat

from dataclasses import dataclass
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from EPInformer.models import EPInformer_v2, enhancer_predictor_256bp


def _resolve_expr_col(df, cell_type):
    """Resolve expression column: try {cell}_RPKM, then Actual_{cell}."""
    col = cell_type + '_RPKM'
    if col in df.columns:
        return col
    alt = 'Actual_' + cell_type
    if alt in df.columns:
        print(f'  Expression column: {alt} (fallback)')
        return alt
    raise KeyError(f"No expression column found: tried '{col}' and '{alt}'")


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
                 rm_self_promoter=False, promoter_activity_df=None,
                 strand_aware=False):
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
        self.promoter_activity_df = promoter_activity_df

        # Preload all data into shared-memory tensors (zero-copy across DataLoader workers)
        print('Loading HDF5 data into shared memory...')
        _all_ensid = [e.decode() if isinstance(e, bytes) else e for e in self.data_h5['ensid'][:]]
        # Filter to genes present in expression CSV (v2 HDF5 may have more genes than expr CSV)
        _expr_index = set(self.expr_df.index)
        _keep = [i for i, e in enumerate(_all_ensid) if e in _expr_index]
        if len(_keep) < len(_all_ensid):
            print(f'  Filtered {len(_all_ensid)} HDF5 genes to {len(_keep)} in expression CSV')
            _keep = np.array(_keep)
            self._ensid = [_all_ensid[i] for i in _keep]
            self._promoter_seq = torch.from_numpy(self.data_h5['promoter_seq'][_keep])
            self._gene_enh_idx = torch.from_numpy(self.data_h5['gene_enh_idx'][_keep].astype(np.int64))
        else:
            self._ensid = _all_ensid
            self._promoter_seq = torch.from_numpy(self.data_h5['promoter_seq'][:])
            self._gene_enh_idx = torch.from_numpy(self.data_h5['gene_enh_idx'][:].astype(np.int64))
        self._enhancer_seq = torch.from_numpy(self.data_h5['enhancer_seq'][:])
        def _nan_to_num_feat(name, arr):
            a = np.asarray(arr, dtype=np.float64)
            n_nan = int(np.isnan(a).sum())
            if n_nan:
                print(
                    f"  Warning: {n_nan} NaN(s) in HDF5 '{name}' — replacing with 0.0 "
                    "(common for missing Hi-C contact)."
                )
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.from_numpy(a)
        if len(_keep) < len(_all_ensid):
            self._distance = _nan_to_num_feat('distance', self.data_h5['distance'][_keep])
            self._activity = _nan_to_num_feat('activity', self.data_h5['activity'][_keep])
            self._contact = _nan_to_num_feat('contact', self.data_h5['contact'][_keep])
        else:
            self._distance = _nan_to_num_feat('distance', self.data_h5['distance'][:])
            self._activity = _nan_to_num_feat('activity', self.data_h5['activity'][:])
            self._contact = _nan_to_num_feat('contact', self.data_h5['contact'][:])
        self.data_h5.close()
        del self.data_h5

        # Precompute per-gene expression features (avoids slow pandas .loc in workers)
        _all_rna_cols = ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
                         'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        rna_cols = [c for c in _all_rna_cols if c in self.expr_df.columns]
        self.has_rna_feats = len(rna_cols) > 0
        if not self.has_rna_feats:
            print("  No Xpresso feature columns found — rna_feats will be zeros")
        expr_col = _resolve_expr_col(self.expr_df, cell_type)
        rna_list = []
        expr_list = []
        n_rna_dim = len(rna_cols) if rna_cols else 0
        for ensid in self._ensid:
            row = self.expr_df.loc[ensid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if rna_cols:
                rna = row[rna_cols].values.astype(np.float32).flatten()
            else:
                rna = np.zeros(n_rna_dim, dtype=np.float32)
            if use_prm_signal:
                prm_act = 0.0
                if promoter_activity_df is not None and ensid in promoter_activity_df.index:
                    prm_act = float(np.log(1 + promoter_activity_df.loc[ensid, 'promoter_activity']))
                rna = np.concatenate([rna, [prm_act]])
            rna_list.append(rna)
            if expr_type == 'CAGE':
                expr_list.append(float(np.log10(row[cell_type + '_CAGE_128*3_sum'] + 1)))
            else:
                expr_list.append(float(row[expr_col]))
        self._rna_feats = torch.from_numpy(np.array(rna_list, dtype=np.float32))
        self._expr = torch.from_numpy(np.array(expr_list, dtype=np.float32))
        # Extract strand info before freeing expr_df
        self._strand_map = {}
        if strand_aware and 'strand' in self.expr_df.columns:
            for ensid in self._ensid:
                if ensid in self.expr_df.index:
                    val = self.expr_df.loc[ensid, 'strand']
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    self._strand_map[ensid] = val
        del self.expr_df  # free memory; no longer needed after precomputation

        # Precompute per-gene promoter signal for pe_feats injection
        self._prm_signal = None
        if use_prm_signal and promoter_activity_df is not None:
            prm_sig_list = []
            for ensid in self._ensid:
                if ensid in promoter_activity_df.index:
                    prm_sig_list.append(float(np.log(1 + promoter_activity_df.loc[ensid, 'promoter_activity'])))
                else:
                    prm_sig_list.append(0.0)
            self._prm_signal = torch.tensor(prm_sig_list, dtype=torch.float32)

        # Pre-compute pe_ohe and pe_feats for all genes (avoids per-sample numpy work)
        print('Pre-computing pe_ohe and pe_feats for all genes...')
        N = len(self._ensid)
        L = self._promoter_seq.shape[1]
        self._pe_ohe = torch.zeros((N, 1 + self.max_n_enh, L, 4), dtype=torch.float32)
        self._pe_ohe[:, 0] = self._promoter_seq.float()
        for i in range(N):
            enh_indices = self._gene_enh_idx[i]
            valid_mask = enh_indices >= 0
            if valid_mask.any():
                self._pe_ohe[i, 1:][valid_mask] = self._enhancer_seq[enh_indices[valid_mask]].float()

        # Build pe_feats: slot 0 = ones (promoter), slots 1..n_enh = [abs(dist), activity, contact, signed_dist]
        abs_dist = self._distance.abs().float()
        activity_f = self._activity.float()
        contact_f = self._contact.float()
        signed_dist = self._distance.float()
        # Strand-aware: flip signed distance for minus-strand genes
        if self._strand_map:
            for i, ensid in enumerate(self._ensid):
                if self._strand_map.get(ensid) == '-':
                    signed_dist[i] = -signed_dist[i]
        if self.n_enh_feats == 0:
            enh_feats_all = torch.zeros((N, self.max_n_enh, 1), dtype=torch.float32)
        else:
            _all_feats = [abs_dist, activity_f, contact_f, signed_dist]
            n_avail = min(self.n_enh_feats, len(_all_feats))
            enh_feats_all = torch.stack(_all_feats[:n_avail], dim=2)[:, :, :self.n_enh_feats]
        prm_feats = torch.ones((N, 1, enh_feats_all.shape[2]), dtype=torch.float32)
        # Inject promoter activity into pe_feats[0, 1] (matches legacy behavior)
        if use_prm_signal and self._prm_signal is not None and self.n_enh_feats >= 2:
            prm_feats[:, 0, 1] = self._prm_signal
        self._pe_feats = torch.cat([prm_feats, enh_feats_all], dim=1)
        # Store raw distance for distance_thr / rm_self_promoter filtering in __getitem__
        self._raw_distance = self._distance.float()

        # Strand-aware promoter correction: RC promoter for minus-strand genes
        if self._strand_map:
            n_flipped = 0
            for i, ensid in enumerate(self._ensid):
                if self._strand_map.get(ensid) == '-':
                    prm = self._pe_ohe[i, 0]              # (L, 4)
                    mask = prm.sum(dim=-1) > 0
                    length = int(mask.sum().item())
                    if length > 0:
                        comp = prm[:, [3, 2, 1, 0]]       # complement A<->T, C<->G
                        self._pe_ohe[i, 0, :length] = comp[:length].flip(0)
                        n_flipped += 1
            print(f'  Strand-aware: flipped {n_flipped}/{len(self._ensid)} promoters (minus-strand genes)')

        # Store activity for aux head (available even when n_enh_feats=1)
        # Shape: (N, 1+max_n_enh) — slot 0 = promoter (0.0), slots 1..n_enh = activity
        self._enhancer_activity = torch.cat([
            torch.zeros((N, 1), dtype=torch.float32),
            activity_f,
        ], dim=1)

        # Free arrays no longer needed
        del self._promoter_seq, self._gene_enh_idx, self._enhancer_seq
        del self._distance, self._activity, self._contact

        print(f'Loaded: {len(self._ensid)} genes, pe_ohe: {self._pe_ohe.nbytes/1e9:.1f} GB')

    def __len__(self):
        return len(self._ensid)

    def __getitem__(self, idx):
        sample_ensid = self._ensid[idx]

        pe_ohe = self._pe_ohe[idx]      # (1+n_enh, L, 4) float32 tensor
        pe_feats = self._pe_feats[idx]   # (1+n_enh, n_feats) float32 tensor
        rna_feats = self._rna_feats[idx] # (n_rna,) float32 tensor
        expr = self._expr[idx]           # scalar float32 tensor
        activity = self._enhancer_activity[idx]  # (1+n_enh,) float32 tensor

        # Distance thresholding (only when needed)
        if self.distance_thr is not None:
            dist = self._raw_distance[idx]  # (n_enh,)
            n_feat = pe_feats.shape[1]
            L = pe_ohe.shape[1]
            enh_ohe_new = torch.zeros((self.max_n_enh, L, 4), dtype=torch.float32)
            enh_feats_new = torch.zeros((self.max_n_enh, n_feat), dtype=torch.float32)
            activity_new = torch.zeros((self.max_n_enh,), dtype=torch.float32)
            min_d = 1000 if (self.rm_prm_seq or self.rm_self_promoter) else 0
            new_i = 0
            for i in range(min(self.max_n_enh, dist.shape[0])):
                d = abs(dist[i].item())
                if d <= self.distance_thr and d >= min_d:
                    enh_ohe_new[new_i] = pe_ohe[1 + i]
                    enh_feats_new[new_i] = pe_feats[1 + i]
                    activity_new[new_i] = activity[1 + i]
                    new_i += 1
                if new_i >= self.max_n_enh:
                    break
            pe_ohe = torch.cat([pe_ohe[:1], enh_ohe_new], dim=0)
            pe_feats = torch.cat([pe_feats[:1], enh_feats_new], dim=0)
            activity = torch.cat([activity[:1], activity_new], dim=0)

        if self.disable_enh:
            pe_ohe = pe_ohe.clone()
            pe_ohe[1:] = 0
            pe_feats = pe_feats.clone()
            pe_feats[1:] = 0

        return pe_ohe, rna_feats, pe_feats, expr, activity, sample_ensid


class promoter_enhancer_dataset_legacy(Dataset):
    """Dataset that reads the old flat HDF5 format (e.g. K562_200CREs-gene_RPM_4feats.hdf5).

    HDF5 contains:
      - enhancers_ohe:  (N, 200, 2000, 4)  per-gene enhancer one-hot sequences
      - enhancers_feat: (N, 200, 5)         [distance, feat1, feat2, feat3, feat4]
      - promoter_ohe:   (N, 1000, 4)        1kb promoter one-hot (zero-padded to 2kb)
      - ensid:          (N,)                gene IDs
      - expr:           (N, 1)              expression values (optional, can use CSV)

    Features extracted as: [abs(distance), activity, contact][:, :n_enh_feats]
    """

    def __init__(self, h5_path, expr_csv, cell_type='K562', expr_type='RNA',
                 n_enh_feats=3, disable_enh=False, distance_thr=None,
                 max_n_enh=60, use_prm_signal=False, rm_prm_seq=False,
                 use_h5_expr=False, promoter_activity_df=None):
        self.expr_df = pd.read_csv(expr_csv, index_col='gene_id')
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        self.rm_prm_seq = rm_prm_seq
        self.use_h5_expr = use_h5_expr
        self.promoter_activity_df = promoter_activity_df
        if use_h5_expr and expr_type != 'RNA':
            raise ValueError('use_h5_expr is only supported with expr_type RNA')
        self._expr_col = None if use_h5_expr else _resolve_expr_col(self.expr_df, cell_type)

        # Load small arrays into memory; keep large sequence arrays on disk
        print('Loading legacy HDF5 metadata...')
        self._h5_path = h5_path
        f = h5py.File(h5_path, 'r')
        self._ensid = [e.decode() if isinstance(e, bytes) else e for e in f['ensid'][:]]
        self._enhancers_feat = f['enhancers_feat'][:]  # (N, 200, 5) — small
        # Load 1kb promoter from HDF5 and zero-pad to 2kb (500bp each side)
        prm_1k = f['promoter_ohe'][:].astype(np.float32)  # (N, 1000, 4)
        self._promoter_ohe = np.pad(prm_1k, ((0, 0), (500, 500), (0, 0)), mode='constant')  # (N, 2000, 4)
        # Keep enhancers_ohe on disk — read per-sample in __getitem__
        self._enhancers_ohe = None  # lazy-loaded from HDF5
        n_cre = f['enhancers_ohe'].shape[1]
        self._expr_h5 = None
        if use_h5_expr:
            if 'expr' not in f:
                f.close()
                raise ValueError('use_h5_expr=True but HDF5 has no expr dataset')
            self._expr_h5 = f['expr'][:].astype(np.float64).reshape(-1)
            if len(self._expr_h5) != len(self._ensid):
                f.close()
                raise ValueError(
                    f'expr length {len(self._expr_h5)} != ensid length {len(self._ensid)}'
                )
            print('Using expression targets from HDF5 expr (RNA).')
        f.close()
        enh_size_gb = len(self._ensid) * n_cre * 2000 * 4 * 4 / 1e9
        print(f'Loaded: {len(self._ensid)} genes, {n_cre} CREs per gene, '
              f'enhancers_ohe: {enh_size_gb:.1f} GB (lazy, on disk), '
              f'promoter_ohe: {self._promoter_ohe.nbytes/1e9:.1f} GB (in memory)')

    def __len__(self):
        return len(self._ensid)

    def __getitem__(self, idx):
        sample_ensid = self._ensid[idx]

        # Promoter: preloaded 2kb one-hot
        prm_ohe = self._promoter_ohe[idx][np.newaxis, :]  # (1, 2000, 4)

        # Enhancer sequences: lazy-load from HDF5 (avoids ~29 GB in memory)
        if self._enhancers_ohe is None:
            # Open per-worker file handle (h5py is not fork-safe)
            if not hasattr(self, '_h5_handle') or self._h5_handle is None:
                self._h5_handle = h5py.File(self._h5_path, 'r')
            enh_ohe = self._h5_handle['enhancers_ohe'][idx].astype(np.float32)
        else:
            enh_ohe = self._enhancers_ohe[idx]

        # Extract features: [abs(distance), DNase, H3K27ac]
        raw_feats = self._enhancers_feat[idx]  # (200, 5)
        if self.n_enh_feats == 0:
            enh_feats = np.zeros_like(np.concatenate(
                [abs(raw_feats[:, [0]]), raw_feats[:, [3]], raw_feats[:, [-1]]], axis=1)[:, :1])
        else:
            enh_feats = np.concatenate(
                [abs(raw_feats[:, [0]]), raw_feats[:, [3]], raw_feats[:, [-1]]], axis=1)[:, :self.n_enh_feats]

        # mRNA sequence features (Xpresso) — use available columns or zeros
        _all_rna_cols = ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
                         'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        rna_cols = [c for c in _all_rna_cols if c in self.expr_df.columns]
        if rna_cols:
            rna_feats = np.array(self.expr_df.loc[sample_ensid][rna_cols]
                                 .values.astype(float)).flatten()
        else:
            rna_feats = np.zeros(len(_all_rna_cols), dtype=np.float64)

        if self.use_prm_signal:
            prm_act = 0.0
            if self.promoter_activity_df is not None and sample_ensid in self.promoter_activity_df.index:
                prm_act = float(np.log(1 + self.promoter_activity_df.loc[sample_ensid, 'promoter_activity']))
            rna_feats = np.concatenate([rna_feats, np.array([prm_act])])

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

        # Expression target (HDF5 or CSV)
        if self.use_h5_expr:
            expr = float(self._expr_h5[idx])
        elif self.expr_type == 'CAGE':
            expr = float(np.log10(self.expr_df.loc[sample_ensid, self.cell_type + '_CAGE_128*3_sum'] + 1))
        else:
            expr = float(self.expr_df.loc[sample_ensid, self._expr_col])

        pe_ohe = np.concatenate([prm_ohe, enh_ohe], axis=0)
        prm_feat_slot = np.ones_like(enh_feats[[0]])
        if self.use_prm_signal and self.n_enh_feats >= 2:
            prm_act = 0.0
            if self.promoter_activity_df is not None and sample_ensid in self.promoter_activity_df.index:
                prm_act = float(np.log(1 + self.promoter_activity_df.loc[sample_ensid, 'promoter_activity']))
            prm_feat_slot[0, 1] = prm_act
        pe_feats = np.concatenate([prm_feat_slot, enh_feats], axis=0)

        # Activity for aux head (slot 0 = promoter = 0, slots 1+ = DNase activity)
        activity = np.concatenate([[0.0], raw_feats[:self.max_n_enh, 3]], axis=0).astype(np.float32)

        return pe_ohe, rna_feats, pe_feats, expr, activity, sample_ensid


class promoter_enhancer_dataset_pecode(Dataset):
    """Dataset for pe_code HDF5 format (combined promoter+enhancer sequences).

    HDF5 contains:
      - pe_code:   (N, 1+n_enh, 2000, 4)  index 0 = promoter, 1..n_enh = enhancers
      - activity:  (N, 1+n_enh)
      - distance:  (N, 1+n_enh)
      - hic:       (N, 1+n_enh)
      - ensid:     (N,)
    """

    def __init__(self, h5_path, expr_csv, cell_type='K562', expr_type='RNA',
                 n_enh_feats=3, disable_enh=False, distance_thr=None,
                 max_n_enh=60, use_prm_signal=False, rm_prm_seq=False,
                 rm_self_promoter=False, promoter_activity_df=None):
        self.cell_type = cell_type
        self.n_enh_feats = n_enh_feats
        self.expr_type = expr_type
        self.disable_enh = disable_enh
        self.distance_thr = distance_thr
        self.max_n_enh = max_n_enh
        self.use_prm_signal = use_prm_signal
        self.rm_prm_seq = rm_prm_seq
        self.rm_self_promoter = rm_self_promoter

        print('Loading pe_code HDF5 data into shared memory...')
        f = h5py.File(h5_path, 'r')
        self._ensid = [e.decode() if isinstance(e, bytes) else e for e in f['ensid'][:]]
        pe_code = f['pe_code'][:].astype(np.float32)  # (N, 1+n_enh, 2000, 4)
        activity = f['activity'][:].astype(np.float32)  # (N, 1+n_enh)
        distance = f['distance'][:].astype(np.float32)  # (N, 1+n_enh)
        hic = f['hic'][:].astype(np.float32)            # (N, 1+n_enh)
        f.close()

        n_enh_in_file = pe_code.shape[1] - 1  # exclude promoter slot
        n_enh = min(n_enh_in_file, max_n_enh)

        # Split into promoter (slot 0) and enhancers (slots 1..n_enh)
        self._pe_ohe = torch.from_numpy(pe_code[:, :1+n_enh].astype(np.float32))        # Build features: [abs(distance), activity, hic]
        dist_feat = np.abs(distance[:, 1:1+n_enh, np.newaxis])
        act_feat = activity[:, 1:1+n_enh, np.newaxis]
        hic_feat = hic[:, 1:1+n_enh, np.newaxis]
        enh_feats = np.concatenate([dist_feat, act_feat, hic_feat], axis=-1)[:, :, :n_enh_feats]
        # Prepend promoter placeholder features
        prm_feats = np.ones((enh_feats.shape[0], 1, enh_feats.shape[2]), dtype=np.float32)
        # Inject promoter activity into pe_feats[0, 1] (matches legacy behavior)
        if use_prm_signal and promoter_activity_df is not None and n_enh_feats >= 2:
            for i, ensid in enumerate(self._ensid):
                if ensid in promoter_activity_df.index:
                    prm_feats[i, 0, 1] = float(np.log(1 + promoter_activity_df.loc[ensid, 'promoter_activity']))
        self._pe_feats = torch.from_numpy(
            np.concatenate([prm_feats, enh_feats], axis=1).astype(np.float32)
        )
        # Activity for aux head (slot 0 = promoter = 0, slots 1+ = activity)
        self._enhancer_activity = torch.from_numpy(
            np.concatenate([np.zeros((activity.shape[0], 1)), activity[:, 1:1+n_enh]], axis=1).astype(np.float32)
        )
        # Expression and RNA features from CSV
        expr_df = pd.read_csv(expr_csv, index_col='gene_id')
        _all_rna_cols = ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
                         'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        rna_cols = [c for c in _all_rna_cols if c in expr_df.columns]
        self.has_rna_feats = len(rna_cols) > 0
        if not self.has_rna_feats:
            print("  No Xpresso feature columns found — rna_feats will be zeros")
        n_rna_dim = len(rna_cols) if rna_cols else 0
        expr_col = _resolve_expr_col(expr_df, cell_type)
        rna_list, expr_list = [], []
        for ensid in self._ensid:
            row = expr_df.loc[ensid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if rna_cols:
                rna = row[rna_cols].values.astype(np.float32).flatten()
            else:
                rna = np.zeros(n_rna_dim, dtype=np.float32)
            if use_prm_signal:
                prm_act = 0.0
                if promoter_activity_df is not None and ensid in promoter_activity_df.index:
                    prm_act = float(np.log(1 + promoter_activity_df.loc[ensid, 'promoter_activity']))
                rna = np.concatenate([rna, [prm_act]])
            rna_list.append(rna)
            if expr_type == 'CAGE':
                expr_list.append(float(np.log10(row[cell_type + '_CAGE_128*3_sum'] + 1)))
            else:
                expr_list.append(float(row[expr_col]))
        self._rna_feats = torch.from_numpy(np.array(rna_list, dtype=np.float32))
        self._expr = torch.from_numpy(np.array(expr_list, dtype=np.float32))
        del expr_df

        print(f'Loaded: {len(self._ensid)} genes, {n_enh} enhancers/gene, '
              f'pe_ohe: {self._pe_ohe.nbytes/1e9:.1f} GB')

    def __len__(self):
        return len(self._ensid)

    def __getitem__(self, idx):
        pe_ohe = self._pe_ohe[idx]      # (1+n_enh, 2000, 4)
        rna_feats = self._rna_feats[idx]
        pe_feats = self._pe_feats[idx]   # (1+n_enh, n_enh_feats)
        expr = self._expr[idx]
        activity = self._enhancer_activity[idx]
        ensid = self._ensid[idx]
        return pe_ohe, rna_feats, pe_feats, expr, activity, ensid


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

def train(net, training_dataset, fold_i, saved_model_path='./models/', learning_rate=1e-4, model_logger=None, fixed_encoder=False, valid_dataset=None, model_name='', batch_size=64, device='cuda', stratify=None, class_weight=None, EPOCHS=100, valid_size=1000, hparams=None, early_stop_patience=5):
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
    _is_mps = (device == 'mps')
    _num_workers = 2 if _is_mps else 4
    trainloader = data_utils.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=_num_workers, pin_memory=(not _is_mps), drop_last=True, persistent_workers=True)
    es_patience = early_stop_patience if early_stop_patience > 0 else 10**9
    if early_stop_patience <= 0:
        print('Early stopping disabled (early_stop_patience=0); training for full EPOCHS.')
    early_stopping = EarlyStopping(patience=es_patience,
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
        aux_loss_e = 0
        for data in trainloader:
            optimizer.zero_grad(set_to_none=True)
            pe_seqs, rna_feats, enh_feats, y_expr, enh_activity, eid = data
            pe_seqs = pe_seqs.to(device, non_blocking=True)
            if net.useFeat or net.usePromoterSignal:
                rna_feats = rna_feats.to(device, non_blocking=True)
            else:
                rna_feats = None
            enh_feats = enh_feats.to(device, non_blocking=True)
            y_expr = y_expr.to(device, non_blocking=True)
            net._enh_activity = enh_activity.to(device, non_blocking=True)
            pred_expr, _ = net(pe_seqs, enh_feats=enh_feats, rna_feats=rna_feats)
            loss_expr = L_expr(pred_expr, y_expr)
            loss_e += loss_expr.item()
            loss = loss_expr
            # Add contrastive loss if model supports it
            if hasattr(net, '_contrastive_loss') and args.contrastive_lambda > 0:
                ctr_loss = net._contrastive_loss
                if not torch.isnan(ctr_loss):
                    loss = loss + args.contrastive_lambda * ctr_loss
            # Add auxiliary activity prediction loss if model supports it
            if hasattr(net, '_aux_activity_loss') and args.aux_activity_lambda > 0:
                aux_loss = net._aux_activity_loss
                if not torch.isnan(aux_loss):
                    loss = loss + args.aux_activity_lambda * aux_loss
                    aux_loss_e += aux_loss.item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        print('[Epoch %d] loss: %.9f' % (epoch + 1, train_loss))
        aux_str = ', aux_activity loss: {:.6f}'.format(aux_loss_e / len(trainloader)) if args.aux_activity_lambda > 0 else ''
        print('Training Loss: expression loss:', loss_e / len(trainloader), aux_str)
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
    _is_mps = (device == 'mps')
    _num_workers = 2 if _is_mps else 4
    validloader = data_utils.DataLoader(valid_ds, batch_size=batch_size, pin_memory=(not _is_mps), num_workers=_num_workers, persistent_workers=True)
    net.eval()
    L_expr = nn.SmoothL1Loss()
    with torch.no_grad():
        preds = []
        actual = []
        loss_e = 0
        for data in validloader:
            pe_seqs, rna_feats, enh_feats, y_expr, enh_activity, eid = data
            pe_seqs = pe_seqs.to(device, non_blocking=True)
            if net.useFeat or net.usePromoterSignal:
                rna_feats = rna_feats.to(device, non_blocking=True)
            else:
                rna_feats = None
            enh_feats = enh_feats.to(device, non_blocking=True)
            y_expr = y_expr.to(device, non_blocking=True)
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

def test(net, test_ds, fold_i, model_name=None, saved_model_path=None, batch_size=64, device='cuda', model_type='best', biotype_map=None):
    _is_mps = (device == 'mps')
    _num_workers = 2 if _is_mps else 4
    testloader = data_utils.DataLoader(test_ds, batch_size=batch_size, pin_memory=(not _is_mps), num_workers=_num_workers)
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
            pe_seqs, rna_feats, enh_feats, y_expr, enh_activity, eid = data
            pe_seqs = pe_seqs.to(device, non_blocking=True)
            if net.useFeat or net.usePromoterSignal:
                rna_feats = rna_feats.to(device, non_blocking=True)
            else:
                rna_feats = None
            enh_feats = enh_feats.to(device, non_blocking=True)
            y_expr = y_expr.to(device, non_blocking=True)
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

    summary = {'fold': fold_i, 'pearsonr': peasonr, 'r2': r_value**2, 'mse': mse, 'n_test': len(preds)}

    # Biotype-stratified evaluation
    if biotype_map:
        df['biotype'] = df.index.map(biotype_map)
        for bt, suffix in [('protein_coding', 'pc'), ('lincRNA', 'linc')]:
            sub = df[df['biotype'] == bt]
            if len(sub) > 10:
                r_bt, _ = stats.pearsonr(sub['Pred'], sub['actual'])
                mse_bt = mean_squared_error(sub['Pred'], sub['actual'])
                _, _, r2_bt, _, _ = stats.linregress(sub['Pred'], sub['actual'])
                print(f'  {bt}: n={len(sub)}, Pearson R={r_bt:.4f}, R2={r2_bt**2:.4f}, MSE={mse_bt:.4f}')
                summary[f'pearsonr_{suffix}'] = r_bt
                summary[f'r2_{suffix}'] = r2_bt**2
                summary[f'mse_{suffix}'] = mse_bt
                summary[f'n_test_{suffix}'] = len(sub)

    return df, summary


# ── CLI & main loop ──────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, help='cuda id', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default='EPInformer-v2', choices=['EPInformer-v2'])
    parser.add_argument('--expr_type', type=str, help='expression type', default='RNA', choices=['CAGE', 'RNA'])
    parser.add_argument('--n_enh_feats', type=int, help='number of enhancer features', default=3, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--cell', type=str, help='cell type', default='K562')
    parser.add_argument('--use_prm_signal', action='store_true', help='use promoter signal')
    parser.add_argument('--use_pretrained_encoder', action='store_true', help='use pretrained encoder')
    parser.add_argument('--pretrained_encoder_dir', type=str, default='./pretrained_seqencoder_h3k27ac',
                        help='directory containing pre-trained seqEncoder checkpoints')
    parser.add_argument('--legnet_ckpt_dir', type=str, default=None,
                        help='directory containing LegNet .ckpt files (Lightning format)')
    parser.add_argument('--rm_prm_seq', action='store_true', help='remove promoter sequence')
    parser.add_argument('--h5_path', type=str, default='./training_data/k562_run/samples.h5', help='path to preprocessed HDF5')
    parser.add_argument('--expr_csv', type=str, default='./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv', help='gene expression CSV')
    parser.add_argument('--split_csv', type=str, default='./data/leave_chrom_out_crossvalidation_split_18377genes.csv', help='leave-chrom-out fold assignments (index = gene_id / ENSID)')
    parser.add_argument('--use_h5_expr', action='store_true', help='legacy HDF5 only: use expr dataset in HDF5 as RNA target (still use expr_csv for Xpresso RNA features)')
    parser.add_argument('--output_dir', type=str, default='./EPInformer_models/', help='directory for saved models and predictions')
    parser.add_argument('--epochs', type=int, default=50, help='max training epochs (upper bound; may stop earlier via early stopping)')
    parser.add_argument('--early_stop_patience', type=int, default=8,
                        help='stop if validation R² does not improve for this many epochs; 0 disables early stopping')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--rm_self_promoter', action='store_true', help='remove self-promoter elements (distance < 1000bp) at training time')
    parser.add_argument('--gene_list', type=str, default=None,
                        help='GeneList.txt from ABC Neighborhoods (required for --use_prm_signal; provides DHS/H3K27ac promoter activity)')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='fold indices to run (default: 1..12)')
    parser.add_argument('--fold', type=int, default=None, help='if set (1–12), train/test only this fold (overrides --folds default)')
    parser.add_argument('--attn_res_layers', type=int, default=2, help='transformer layers in intra-sequence attention residual')
    parser.add_argument('--attn_res_heads', type=int, default=4, help='attention heads in intra-sequence attention residual')
    parser.add_argument('--intra_attn', action='store_true',
                        help='Add IntraSeqAttentionResidual between seq_encoder and conv_out')
    parser.add_argument('--rc_aware', action='store_true', help='wrap seq encoder to process both strands (fwd + RC, max pool)')
    parser.add_argument('--strand_aware', action='store_true', help='RC promoter sequences for minus-strand genes (uses strand column from expr CSV)')
    parser.add_argument('--contrastive_lambda', type=float, default=0.0,
                        help='Weight for enhancer contrastive loss (0 = disabled, try 0.1)')
    parser.add_argument('--aux_activity_lambda', type=float, default=0.0,
                        help='Weight for auxiliary enhancer activity prediction loss (0 = disabled, try 0.1-1.0)')
    parser.add_argument('--aux_activity_log', action='store_true',
                        help='Log-transform activity targets in aux prediction (log(1+x), reduces scale from 0-123 to 0-4.8)')
    parser.add_argument('--no_dist_in_pos_conv', action='store_true',
                        help='Ablation: remove distance from add_pos_conv (only activity+contact), rely on DistAwareMHA for position')
    parser.add_argument('--out_dim', type=int, default=None,
                        help='d_model for attention layers (default: 64 for LegNet, 128 for AttnRes)')
    parser.add_argument('--n_encoder', type=int, default=3, help='number of transformer encoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--no_enh_mask', action='store_true',
                        help='remove enhancer-enhancer attention mask (allow full attention)')
    parser.add_argument('--no_freeze_encoder', action='store_true',
                        help='do NOT freeze pre-trained encoder (fine-tune all parameters)')
    parser.add_argument('--separate_encoders', action='store_true',
                        help='use separate promoter and enhancer encoders (clone pre-trained encoder for promoter)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                        help='force device (default: auto-detect cuda > mps > cpu)')
    args = parser.parse_args()
    if args.early_stop_patience < 0:
        raise SystemExit('--early_stop_patience must be >= 0')
    if args.use_prm_signal and args.gene_list is None:
        raise SystemExit('--gene_list is required when --use_prm_signal is set')
    # Fail fast on flags inherited from the multi-variant trainer that this trimmed
    # EPInformer-v2 reproduction does NOT implement (they belong to models_abc) — so a
    # run can't be silently mislabeled or a no-op loss silently ignored.
    _unsupported = []
    if args.legnet_ckpt_dir is not None: _unsupported.append('--legnet_ckpt_dir')
    if args.rc_aware: _unsupported.append('--rc_aware')
    if args.intra_attn: _unsupported.append('--intra_attn')
    if args.separate_encoders: _unsupported.append('--separate_encoders')
    if args.no_enh_mask: _unsupported.append('--no_enh_mask')
    # --no_freeze_encoder IS supported (fine-tunes the pretrained encoder end-to-end); it
    # gates the freeze block below, so it is intentionally NOT added to _unsupported.
    if args.no_dist_in_pos_conv: _unsupported.append('--no_dist_in_pos_conv')
    if getattr(args, 'contrastive_lambda', 0) and args.contrastive_lambda > 0: _unsupported.append('--contrastive_lambda')
    if getattr(args, 'aux_activity_lambda', 0) and args.aux_activity_lambda > 0: _unsupported.append('--aux_activity_lambda')
    if _unsupported:
        parser.error('these flags are not supported by this EPInformer-v2 (models.py) '
                     'reproduction: ' + ', '.join(_unsupported))

    # Load promoter activity from GeneList.txt (for --use_prm_signal)
    promoter_activity_df = None
    if args.use_prm_signal and args.gene_list:
        promoter_activity_df = pd.read_csv(args.gene_list, sep='\t', index_col='name')
        dhs_col = 'DHS.RPKM.TSS1Kb' if 'DHS.RPKM.TSS1Kb' in promoter_activity_df.columns else 'DHS.RPM.TSS1Kb'
        h3k_col = 'H3K27ac.RPKM.TSS1Kb' if 'H3K27ac.RPKM.TSS1Kb' in promoter_activity_df.columns else 'H3K27ac.RPM.TSS1Kb'
        promoter_activity_df['promoter_activity'] = np.sqrt(
            promoter_activity_df[dhs_col] * promoter_activity_df[h3k_col]
        )
        print(f'Loaded promoter activity for {len(promoter_activity_df)} genes from {args.gene_list}')

    split_df = pd.read_csv(args.split_csv, index_col=0)

    # Build biotype map for stratified evaluation (optional — only if expr CSV has biotype)
    _expr_cols = pd.read_csv(args.expr_csv, nrows=0).columns
    if 'biotype' in _expr_cols:
        _bt_df = pd.read_csv(args.expr_csv, usecols=['ENSID', 'biotype'])
        biotype_map = dict(zip(_bt_df['ENSID'], _bt_df['biotype']))
        print(f'Biotype map loaded: {len(biotype_map)} genes')
    else:
        biotype_map = None

    if args.fold is not None:
        if not (1 <= args.fold <= 12):
            raise SystemExit('--fold must be between 1 and 12')
        fold_list = [args.fold]
    elif args.folds is not None:
        fold_list = args.folds
    else:
        fold_list = list(range(1, 13))
    for fi in fold_list:
        if not (1 <= fi <= 12):
            raise SystemExit('each fold index must be between 1 and 12')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':
        # Reduce MPS memory overhead by allowing medium-precision matmuls
        torch.set_float32_matmul_precision('medium')

    results = []
    fold_metrics = []
    expr_type = args.expr_type
    batch_size = args.batch_size
    max_n_enh = 60
    dist_thr = 100_000
    lr = args.lr
    model_type = args.model_type
    use_pretrained_encoder = args.use_pretrained_encoder
    cell_type = args.cell
    use_prm_signal = args.use_prm_signal
    saved_model_path = args.output_dir
    os.makedirs(saved_model_path, exist_ok=True)

    print('device:', device)
    print('use_prm_signal:', use_prm_signal)
    model_dist = {'EPInformer-v2': EPInformer_v2}

    # Auto-detect HDF5 format
    with h5py.File(args.h5_path, 'r') as _f:
        h5_keys = list(_f.keys())
        if 'enhancers_ohe' in _f:
            h5_format = 'legacy'
        elif 'pe_code' in _f:
            h5_format = 'pe_code'
        else:
            h5_format = 'factored'
    print(f'HDF5 format: {h5_format}')

    for fi in fold_list:
        fold_i = 'fold_{}'.format(fi)
        # Auto-detect RNA features from expression CSV
        _rna_cols_check = ['UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore',
                           'UTR3LEN_log10zscore', 'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
        _expr_df_check = pd.read_csv(args.expr_csv, nrows=1)
        _has_rna_feats = any(c in _expr_df_check.columns for c in _rna_cols_check)
        if not _has_rna_feats:
            print("No Xpresso feature columns in expression CSV — disabling rna_feats")
        for use_rna_feats, rm_prm_seq in [(_has_rna_feats, args.rm_prm_seq)]:
            for cell in [cell_type]:
                for n_enh_feats in [args.n_enh_feats]:
                    ds_kwargs = dict(
                        h5_path=args.h5_path, expr_csv=args.expr_csv,
                        cell_type=cell, expr_type=expr_type,
                        n_enh_feats=n_enh_feats, distance_thr=dist_thr,
                        max_n_enh=max_n_enh, use_prm_signal=use_prm_signal,
                        rm_prm_seq=rm_prm_seq,
                        promoter_activity_df=promoter_activity_df,
                        strand_aware=args.strand_aware,
                    )
                    if h5_format == 'legacy':
                        ds = promoter_enhancer_dataset_legacy(
                            **ds_kwargs, use_h5_expr=args.use_h5_expr)
                    elif h5_format == 'pe_code':
                        ds = promoter_enhancer_dataset_pecode(**ds_kwargs, rm_self_promoter=args.rm_self_promoter)
                    else:
                        ds = promoter_enhancer_dataset(**ds_kwargs, rm_self_promoter=args.rm_self_promoter)
                    train_ensid = split_df[split_df[fold_i] == 'train'].index
                    valid_ensid = split_df[split_df[fold_i] == 'valid'].index
                    test_ensid = split_df[split_df[fold_i] == 'test'].index
                    ensid_list = ds._ensid
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
                        pt_model_name = '{}/fold_{}_best_enhancer_predictor_H3K27ac_256bp_{}_checkpoint.pt'.format(args.pretrained_encoder_dir, fi, cell)
                        checkpoint = torch.load(pt_model_name, weights_only=False)
                        pretrained_convNet = enhancer_predictor_256bp()
                        pretrained_convNet.load_state_dict(checkpoint['model_state_dict'])
                        out_dim = 64 if args.out_dim is None else args.out_dim
                        model = model_dist[model_type](n_extraFeat=n_enh_feats, pre_trained_encoder=pretrained_convNet.encoder, useFeat=use_rna_feats, out_dim=out_dim, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal, n_encoder=args.n_encoder, head=args.n_head, device=device).to(device)
                        if not args.no_freeze_encoder:
                            print('freezing the encoder parameters')
                            for name, value in model.named_parameters():
                                if name.startswith('seq_encoder'):
                                    value.requires_grad = False
                        else:
                            print('NOT freezing encoder — fine-tuning all parameters end-to-end')
                    else:
                        out_dim = 64 if args.out_dim is None else args.out_dim
                        model = model_dist[model_type](n_extraFeat=n_enh_feats, pre_trained_encoder=None, useFeat=use_rna_feats, out_dim=out_dim, n_enhancer=max_n_enh, useBN=False, usePromoterSignal=use_prm_signal, n_encoder=args.n_encoder, head=args.n_head, device=device).to(device)
                    use_rna_feats_flag = 'rnafeats' if use_rna_feats else 'nornafeats'
                    use_prm_signal_flag = 'prmsig' if use_prm_signal else 'nonprmsig'
                    rm_prm_signal_flag = 'rmprmseq' if rm_prm_seq else 'nonrmprmseq'
                    rc_flag = '.rc' if args.rc_aware else ''
                    strand_flag = '.strand' if args.strand_aware else ''
                    intra_flag = '.intra' if args.intra_attn else ''
                    ctr_flag = '.ctr{}'.format(args.contrastive_lambda) if args.contrastive_lambda > 0 else ''
                    aux_flag = '.aux{}{}'.format(args.aux_activity_lambda, 'log' if args.aux_activity_log else '') if args.aux_activity_lambda > 0 else ''
                    # Architecture flags (only when non-default)
                    arch_flag = ''
                    if out_dim != 64: arch_flag += '.d{}'.format(out_dim)
                    if args.n_encoder != 3: arch_flag += '.L{}'.format(args.n_encoder)
                    if args.n_head != 4: arch_flag += '.h{}'.format(args.n_head)
                    if args.no_enh_mask: arch_flag += '.openmask'
                    if args.lr != 1e-4: arch_flag += '.lr{}'.format(args.lr)
                    if args.separate_encoders: arch_flag += '.dualenc'
                    if args.no_freeze_encoder: arch_flag += '.unfreeze'
                    model.name = model.name + '.{}.{}.{}enhs.{}feats.{}.{}.{}.{}kb2TSS{}{}{}{}{}{}'.format(cell, expr_type, max_n_enh, n_enh_feats, use_rna_feats_flag, use_prm_signal_flag, rm_prm_signal_flag, str(int(dist_thr / 1000)), rc_flag, strand_flag, intra_flag, ctr_flag, aux_flag, arch_flag)
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
                        'early_stop_patience': args.early_stop_patience,
                        'h5_path': args.h5_path,
                        'split_csv': args.split_csv,
                        'use_h5_expr': args.use_h5_expr,
                        'gene_list': args.gene_list,
                    }
                    train(model, train_ds, valid_dataset=valid_ds, learning_rate=lr, EPOCHS=args.epochs, model_name=model.name, fold_i=fi, batch_size=batch_size, device=device, saved_model_path=saved_model_path, hparams=hparams, early_stop_patience=args.early_stop_patience)
                    # Test
                    test_df, metrics = test(model, test_ds, model_name=model.name, saved_model_path=saved_model_path, fold_i=fi, batch_size=batch_size, device=device, biotype_map=biotype_map)
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
    # Fold-qualify so parallel one-fold-per-array-task runs do not overwrite/corrupt
    # a shared file. The reliable pooled source is the per-fold *_predictions.csv;
    # evaluate.py reads those. (fold_summary.csv below is flock-append-safe.)
    _folds_tok = "_".join(str(f) for f in fold_list)
    results_df.to_csv(os.path.join(saved_model_path, '{}.folds_{}_results.csv'.format(model.name, _folds_tok)), index=False)

    # Save fold summary — append-safe for parallel array jobs
    summary_df = pd.DataFrame(fold_metrics)
    summary_path = os.path.join(saved_model_path, 'fold_summary.csv')
    import fcntl
    with open(summary_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        write_header = f.tell() == 0
        summary_df.to_csv(f, index=False, header=write_header)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f'\nFold summary appended to {summary_path}')
    print(summary_df.to_string(index=False))
    print(f'\nFold(s) Pearson R: {summary_df["pearsonr"].mean():.4f} +/- {summary_df["pearsonr"].std():.4f}')
