"""
Generate evaluation report from EPInformer training results.

Reads prediction CSVs and training logs, produces:
  - summary_metrics.csv — per-fold Pearson R, R², MSE
  - scatter_pred_vs_actual.png — predicted vs actual expression
  - barplot_pearsonr_per_fold.png — per-fold Pearson R bar chart
  - training_curves.png — train loss + val R² over epochs (if logs exist)

Usage:
    python report_EPInformer.py \
        --results_dir ./EPInformer_models/ \
        --output_dir ./EPInformer_report/
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_predictions(results_dir):
    """Find and load all fold prediction CSVs."""
    patterns = [
        os.path.join(results_dir, 'fold_*_predictions.csv'),
        os.path.join(results_dir, '*_fold*_pred_expr.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f'No prediction CSVs found in {results_dir}')

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        # Normalize column names
        if 'Pred' not in df.columns and 'pred' in df.columns:
            df = df.rename(columns={'pred': 'Pred'})
        if 'actual' not in df.columns and 'Actual' in df.columns:
            df = df.rename(columns={'Actual': 'actual'})
        if 'fold_idx' not in df.columns and 'fold_i' in df.columns:
            df = df.rename(columns={'fold_i': 'fold_idx'})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def collect_training_logs(results_dir):
    """Find and load per-fold training log CSVs."""
    files = sorted(glob.glob(os.path.join(results_dir, 'fold_*_training_log.csv')))
    if not files:
        return None
    logs = {}
    for f in files:
        fold = int(os.path.basename(f).split('_')[1])
        logs[fold] = pd.read_csv(f, sep='\t')
    return logs


def compute_metrics(pred_df):
    """Compute per-fold metrics."""
    rows = []
    for fold, group in pred_df.groupby('fold_idx'):
        pr, _ = stats.pearsonr(group['Pred'], group['actual'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(group['Pred'], group['actual'])
        mse = mean_squared_error(group['actual'], group['Pred'])
        rows.append({
            'fold': int(fold),
            'n_genes': len(group),
            'pearsonr': pr,
            'r2': r_value ** 2,
            'mse': mse,
        })
    return pd.DataFrame(rows).sort_values('fold')


def plot_scatter(pred_df, output_path):
    """Scatter plot of predicted vs actual expression, colored by fold."""
    fig, ax = plt.subplots(figsize=(8, 7))
    folds = sorted(pred_df['fold_idx'].unique())
    cmap = plt.cm.get_cmap('tab20', len(folds))
    for i, fold in enumerate(folds):
        group = pred_df[pred_df['fold_idx'] == fold]
        ax.scatter(group['actual'], group['Pred'], alpha=0.3, s=8, c=[cmap(i)], label=f'Fold {int(fold)}')

    pr_all, _ = stats.pearsonr(pred_df['Pred'], pred_df['actual'])
    lims = [min(pred_df['actual'].min(), pred_df['Pred'].min()),
            max(pred_df['actual'].max(), pred_df['Pred'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Actual Expression', fontsize=12)
    ax.set_ylabel('Predicted Expression', fontsize=12)
    ax.set_title(f'Predicted vs Actual (Pearson R = {pr_all:.4f}, N = {len(pred_df)})', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper left', markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved scatter plot: {output_path}')


def plot_barplot(metrics_df, output_path):
    """Bar chart of Pearson R per fold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    folds = metrics_df['fold'].values
    prs = metrics_df['pearsonr'].values
    colors = plt.cm.tab20(np.linspace(0, 1, len(folds)))
    bars = ax.bar(folds, prs, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(y=prs.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean = {prs.mean():.4f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Pearson R', fontsize=12)
    ax.set_title('Pearson R per Fold', fontsize=13)
    ax.set_xticks(folds)
    ax.legend(fontsize=10)
    for bar, pr in zip(bars, prs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{pr:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved barplot: {output_path}')


def plot_training_curves(logs, output_path):
    """Plot train loss and validation R² per epoch for each fold."""
    n_folds = len(logs)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    cmap = plt.cm.tab20(np.linspace(0, 1, n_folds))
    for i, (fold, df) in enumerate(sorted(logs.items())):
        axes[0].plot(df['epoch'], df['train_loss'], color=cmap[i], alpha=0.7, label=f'Fold {fold}')
        axes[1].plot(df['epoch'], df['val_r2'], color=cmap[i], alpha=0.7, label=f'Fold {fold}')

    axes[0].set_ylabel('Train Loss', fontsize=11)
    axes[0].set_title('Training Loss per Epoch', fontsize=12)
    axes[0].legend(fontsize=7, ncol=4)

    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation R²', fontsize=11)
    axes[1].set_title('Validation R² per Epoch', fontsize=12)
    axes[1].legend(fontsize=7, ncol=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved training curves: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='EPInformer evaluation report')
    parser.add_argument('--results_dir', type=str, default='./EPInformer_models/',
                        help='Directory containing prediction CSVs and training logs')
    parser.add_argument('--output_dir', type=str, default='./EPInformer_report/',
                        help='Directory for report outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions
    print(f'Looking for predictions in {args.results_dir}...')
    pred_df = collect_predictions(args.results_dir)
    print(f'Loaded {len(pred_df)} predictions across {pred_df["fold_idx"].nunique()} folds')

    # Compute metrics
    metrics_df = compute_metrics(pred_df)
    metrics_path = os.path.join(args.output_dir, 'summary_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f'\nSummary metrics saved to {metrics_path}')
    print(metrics_df.to_string(index=False))
    print(f'\nMean Pearson R: {metrics_df["pearsonr"].mean():.4f} +/- {metrics_df["pearsonr"].std():.4f}')
    print(f'Mean R²: {metrics_df["r2"].mean():.4f}')
    print(f'Mean MSE: {metrics_df["mse"].mean():.4f}')

    # Plots
    plot_scatter(pred_df, os.path.join(args.output_dir, 'scatter_pred_vs_actual.png'))
    plot_barplot(metrics_df, os.path.join(args.output_dir, 'barplot_pearsonr_per_fold.png'))

    # Training curves (optional)
    logs = collect_training_logs(args.results_dir)
    if logs:
        plot_training_curves(logs, os.path.join(args.output_dir, 'training_curves.png'))
    else:
        print('No training logs found; skipping training curves plot.')

    print(f'\nReport complete. Outputs in {args.output_dir}')


if __name__ == '__main__':
    main()
