"""
Predict enhancer-to-gene interactions using trained EPInformer models.

Loads trained checkpoints (from train_EPInformer_abc.py) and runs inference
on test genes per fold, extracting:
  - Predicted gene expression
  - Per-layer attention scores (promoter → enhancer)
  - Feature weights

Uses the same factored HDF5 format as the training script.

Usage:
    python predict_E2G_abc_feat.py \
        --h5_path ./training_data/k562_run/samples.h5 \
        --expr_csv ./data_EPInformer/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
        --model_dir ./EPInformer_models/ \
        --output_dir ./enhancer_attn_scores/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
from torch.utils.data import Subset, DataLoader

from train_EPInformer_abc import promoter_enhancer_dataset
from EPInformer.models_abc import (
    EPInformer_abc, EPInformer_v2,
    EPInformer_abc_dist, EPInformer_abc_dist_v2,
    enhancer_predictor_256bp,
)


def load_model(model, fold_i, model_dir):
    """Load the best checkpoint for a given fold."""
    model_path = os.path.join(
        model_dir, f"fold_{fold_i}_best_{model.name}_checkpoint.pt"
    )
    print(f'Loading model from {model_path}')
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'hparams' in checkpoint:
        print(f'Checkpoint hparams: {checkpoint["hparams"]}')
    print(f'{model.name} loaded!')
    return model


def predict_fold(net, test_ds, fold_i, h5_file, batch_size, device, output_dir, model_name):
    """Run inference on test set, extract attention scores and feature weights."""
    testloader = DataLoader(test_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    net.eval()

    preds, actuals, ensids = [], [], []
    attn_records = []

    # Preload enhancer names and gene_enh_idx for mapping
    enh_names = [n.decode() if isinstance(n, bytes) else n for n in h5_file['enhancer_name'][:]]
    gene_enh_idx = h5_file['gene_enh_idx'][:]  # (N, max_n_enh)
    all_ensids = [e.decode() if isinstance(e, bytes) else e for e in h5_file['ensid'][:]]
    ensid_to_row = {eid: i for i, eid in enumerate(all_ensids)}

    n_layers = None
    with torch.no_grad():
        for data in tqdm(testloader, desc=f'Fold {fold_i}'):
            pe_seqs, rna_feats, enh_feats, y_expr, eids = data
            pe_seqs = pe_seqs.float().to(device)
            rna_feats_t = rna_feats.float().to(device) if net.useFeat else None
            enh_feats_t = enh_feats.float().to(device)

            pred_expr, attn_out = net(pe_seqs, enh_feats=enh_feats_t, rna_feats=rna_feats_t)

            # dist models return (attn_tensor, feats_w); others return just attn_tensor
            if isinstance(attn_out, tuple):
                attn_tensor, feats_w = attn_out
                feats_w = feats_w.cpu().numpy()  # (batch, 1+n_enh)
            else:
                attn_tensor = attn_out
                feats_w = None

            # attn_tensor shape: (n_layers, batch, seq, seq) — heads already averaged
            attn_weights = attn_tensor.permute(1, 0, 2, 3).cpu().numpy()  # (batch, layers, seq, seq)
            if n_layers is None:
                n_layers = attn_weights.shape[1]

            batch_preds = pred_expr.flatten().cpu().numpy()
            batch_actual = y_expr.numpy()

            for i, eid in enumerate(eids):
                gene_row = ensid_to_row[eid]
                enh_idx = gene_enh_idx[gene_row]  # (max_n_enh,)
                n_enh = int((enh_feats[i].sum(dim=-1) != 0).sum())

                # One row per gene-enhancer pair
                for j in range(n_enh):
                    record = {'gene': eid}
                    record['element'] = enh_names[enh_idx[j]] if enh_idx[j] >= 0 else ''
                    for layer_i in range(n_layers):
                        record[f'attn_score_layer{layer_i}'] = float(attn_weights[i, layer_i, 0, j + 1])
                    # feat_score: from model's feature weighting module (dist models only)
                    record['feat_score'] = float(feats_w[i, j + 1]) if feats_w is not None else 0.0
                    attn_records.append(record)

                preds.append(float(batch_preds[i]))
                actuals.append(float(batch_actual[i]))
                ensids.append(eid)

    # Save predictions
    pred_df = pd.DataFrame({
        'ENSID': ensids, 'Actual': actuals, 'Pred': preds, 'fold_i': fold_i
    })
    pred_path = os.path.join(output_dir, f'{model_name}_fold{fold_i}_pred_expr.csv')
    pred_df.to_csv(pred_path, index=False)

    # Save attention scores
    attn_df = pd.DataFrame(attn_records)
    attn_path = os.path.join(output_dir, f'{model_name}_fold{fold_i}_attn_scores.csv')
    attn_df.to_csv(attn_path, index=False)

    pr = pearsonr(actuals, preds)[0]
    print(f'Fold {fold_i} Pearson R: {pr:.4f} ({len(preds)} genes)')
    return pred_df


# ── CLI & main loop ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EPInformer prediction & attention extraction')
    parser.add_argument('--h5_path', type=str, default='./training_data/k562_run/samples.h5')
    parser.add_argument('--expr_csv', type=str, default='./data_EPInformer/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv')
    parser.add_argument('--model_dir', type=str, default='./EPInformer_models/', help='directory with trained checkpoints')
    parser.add_argument('--output_dir', type=str, default='./enhancer_attn_scores/')
    parser.add_argument('--model_type', type=str, default='EPInformer-v2', choices=['EPInformer-v2', 'EPInformer-abc', 'EPInformer-abc-dist', 'EPInformer-abc-dist-v2'])
    parser.add_argument('--expr_type', type=str, default='RNA', choices=['CAGE', 'RNA'])
    parser.add_argument('--cell', type=str, default='K562')
    parser.add_argument('--n_enh_feats', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--use_prm_signal', action='store_true')
    parser.add_argument('--rm_prm_seq', action='store_true')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--folds', type=str, default='1-12', help='fold range, e.g. "1" or "1-12"')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

    os.makedirs(args.output_dir, exist_ok=True)

    split_df = pd.read_csv('./data/leave_chrom_out_crossvalidation_split_18377genes.csv', index_col=0)
    model_dist = {
        'EPInformer-abc': EPInformer_abc, 'EPInformer-v2': EPInformer_v2,
        'EPInformer-abc-dist': EPInformer_abc_dist, 'EPInformer-abc-dist-v2': EPInformer_abc_dist_v2,
    }

    max_n_enh = 60
    dist_thr = 100_000
    cell = args.cell
    n_enh_feats = args.n_enh_feats
    use_prm_signal = args.use_prm_signal
    rm_prm_seq = args.rm_prm_seq

    # Parse fold range
    if '-' in args.folds:
        f_start, f_end = args.folds.split('-')
        fold_range = range(int(f_start), int(f_end) + 1)
    else:
        fold_range = [int(args.folds)]

    all_preds = []
    for fi in fold_range:
        fold_i = f'fold_{fi}'
        print(f'\n=== Processing {fold_i} ===')

        ds = promoter_enhancer_dataset(
            h5_path=args.h5_path,
            expr_csv=args.expr_csv,
            cell_type=cell,
            expr_type=args.expr_type,
            n_enh_feats=n_enh_feats,
            distance_thr=dist_thr,
            max_n_enh=max_n_enh,
            use_prm_signal=use_prm_signal,
            rm_prm_seq=rm_prm_seq,
        )

        test_ensid = split_df[split_df[fold_i] == 'test'].index
        ensid_list = [eid.decode('utf-8') for eid in ds.data_h5['ensid'][:]]
        ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
        ensid_df['idx'] = np.arange(len(ensid_list))
        ensid_df = ensid_df.set_index('ensid')
        test_idx = ensid_df.loc[ensid_df.index.intersection(test_ensid)]['idx']
        test_ds = Subset(ds, test_idx)
        print(f'Test genes: {len(test_idx)}')

        # Build model
        use_rna_feats = True
        model = model_dist[args.model_type](
            n_extraFeat=n_enh_feats, pre_trained_encoder=None,
            useFeat=use_rna_feats, out_dim=64, n_enhancer=max_n_enh,
            useBN=False, usePromoterSignal=use_prm_signal,
        ).to(device)

        use_rna_feats_flag = 'rnafeats' if use_rna_feats else 'nornafeats'
        use_prm_signal_flag = 'prmsig' if use_prm_signal else 'nonprmsig'
        rm_prm_signal_flag = 'rmprmseq' if rm_prm_seq else 'nonrmprmseq'
        model.name = model.name + '.{}.{}.{}enhs.{}feats.{}.{}.{}.{}kb2TSS'.format(
            cell, args.expr_type, max_n_enh, n_enh_feats,
            use_rna_feats_flag, use_prm_signal_flag, rm_prm_signal_flag,
            str(int(dist_thr / 1000))
        )

        model = load_model(model, fi, args.model_dir)

        pred_df = predict_fold(
            model, test_ds, fi, ds.data_h5,
            batch_size=args.batch_size, device=device,
            output_dir=args.output_dir, model_name=model.name,
        )
        all_preds.append(pred_df)

    # Combined predictions
    all_preds_df = pd.concat(all_preds)
    combined_path = os.path.join(args.output_dir, f'{model.name}_all_folds_pred_expr.csv')
    all_preds_df.to_csv(combined_path, index=False)
    overall_pr = pearsonr(all_preds_df['Actual'], all_preds_df['Pred'])[0]
    print(f'\nOverall Pearson R (all folds): {overall_pr:.4f}')
    print(f'Combined predictions saved to {combined_path}')
