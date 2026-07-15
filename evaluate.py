#!/usr/bin/env python
"""Evaluate EPInformer pipeline results (12-fold leave-chromosome-out).

Two modes:

  expression   Aggregate out-of-fold gene-expression predictions written by
               train_EPInformer.py and report Pearson R / R^2 / Spearman.
               Reads either  {output_dir}/*_results.csv  (all folds concatenated,
               columns: Pred, actual, fold_idx)  or, if absent, concatenates the
               per-fold  {output_dir}/fold_*_predictions.csv  files.

  encoder      Aggregate the sequence-encoder activity predictions written by
               train_seqEncoder.py (predictions/fold_*_predictions.csv, columns:
               preds, actual, ensid) and report Pearson R on log2 activity.

Usage:
  python evaluate.py expression --pred_dir ./EPInformer_models/K562
  python evaluate.py encoder    --pred_dir ./results/seqencoder/K562
  # options: --out <dir>  --label <name>  --no-plot

Headline targets (K562, EPInformer-v2 PE-Activity, 12-fold):
  expression  Pearson R  ~0.88 (CAGE) / ~0.86 (RNA)
  encoder     Pearson R  ~0.71 (H3K27ac log2 activity)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _pearson_r2_mse(pred, actual):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    if len(pred) < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return dict(pearsonr=np.nan, spearmanr=np.nan, r2=np.nan,
                    mse=float(np.mean((pred - actual) ** 2)) if len(pred) else np.nan,
                    n=int(len(pred)))
    r = stats.pearsonr(pred, actual)[0]
    rho = stats.spearmanr(pred, actual)[0]
    mse = float(np.mean((pred - actual) ** 2))
    return dict(pearsonr=float(r), spearmanr=float(rho), r2=float(r ** 2), mse=mse, n=int(len(pred)))


def _fold_from_name(path):
    m = re.search(r"fold_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _validate_fold_files(files, mode, allow_partial=False):
    """Guard against pooling multiple prediction sets for the same fold.

    Each held-out fold must be represented by exactly ONE prediction file.
    Reusing a directory for a second run — another model variant, or a
    different test strand for the encoder (``..._reverse_...`` vs
    ``..._forward_...``) — leaves >1 file per fold; concatenating them would
    count every held-out row twice and silently corrupt the pooled OOF metric.
    Fail loud instead. Also reports fold coverage (expects folds 1..12).
    """
    from collections import defaultdict
    by_fold = defaultdict(list)
    for f in files:
        by_fold[_fold_from_name(f)].append(os.path.basename(f))
    dups = {k: v for k, v in sorted(by_fold.items()) if len(v) > 1}
    if dups:
        detail = "\n".join(f"    fold {k}: {v}" for k, v in dups.items())
        sys.exit(
            f"[{mode}] {len(dups)} fold(s) have >1 prediction file in this dir — pooling them "
            f"would double-count held-out rows. Point --pred_dir at a clean/separate dir per run "
            f"(or delete stale files):\n{detail}"
        )
    folds = sorted(k for k in by_fold if k >= 0)
    if set(folds) != set(range(1, 13)):
        missing = sorted(set(range(1, 13)) - set(folds))
        extra = sorted(set(folds) - set(range(1, 13)))
        detail = (f"fold coverage {folds} (expected 1..12"
                  + (f"; missing {missing}" if missing else "")
                  + (f"; unexpected {extra}" if extra else "") + ")")
        if allow_partial:
            print(f"  [warn] {detail} — pooled R is over these folds only (--allow-partial)")
        else:
            sys.exit(
                f"[{mode}] {detail}. A 12-fold cross-validation is incomplete, so the "
                f"'OVERALL (out-of-fold pooled)' summary would be misleading. Wait for all "
                f"folds, or pass --allow-partial to evaluate the folds present so far."
            )


def _load_expression(pred_dir, allow_partial=False):
    """Return a DataFrame with columns pred, actual, fold from a training output dir.

    Prefers the per-fold ``fold_*_predictions.csv`` files — these are reliable
    under the one-fold-per-array-task SLURM workflow (each fold writes its own
    file). Falls back to an aggregated ``*_results.csv`` only if no per-fold
    files exist (a single-process run); note that file is overwritten per task
    in array mode, so it is not trusted for pooled metrics when folds exist.
    """
    files = sorted(glob.glob(os.path.join(pred_dir, "fold_*_predictions.csv")))
    if files:
        _validate_fold_files(files, "expression", allow_partial=allow_partial)
        frames = []
        for f in files:
            d = pd.read_csv(f)
            pred_col = "Pred" if "Pred" in d.columns else "pred"
            fold = d["fold_idx"].iloc[0] if "fold_idx" in d.columns else _fold_from_name(f)
            frames.append(pd.DataFrame({"pred": d[pred_col], "actual": d["actual"], "fold": fold}))
        return pd.concat(frames, ignore_index=True), f"{len(files)} per-fold prediction files"
    results = sorted(glob.glob(os.path.join(pred_dir, "*_results.csv")))
    if results:
        df = pd.read_csv(results[0])
        pred_col = "Pred" if "Pred" in df.columns else "pred"
        fold_col = "fold_idx" if "fold_idx" in df.columns else "fold"
        out = pd.DataFrame({"pred": df[pred_col], "actual": df["actual"],
                            "fold": df[fold_col] if fold_col in df.columns else -1})
        return out, os.path.basename(results[0]) + " (aggregate fallback)"
    sys.exit(f"No fold_*_predictions.csv or *_results.csv found in {pred_dir}")


def _load_encoder(pred_dir, allow_partial=False):
    """Return a DataFrame with columns pred, actual, fold from a seqEncoder output dir.

    train_seqEncoder.py writes predictions/ files named
    ``fold_{i}_enhancer_predictor_H3K27ac_256bp_{cell}_{strand}[_summit_only]_predictions.csv``
    each with a ``fold`` column. Uses the all-bins files (excludes summit_only).
    """
    search = os.path.join(pred_dir, "predictions")
    if not os.path.isdir(search):
        search = pred_dir
    files = [f for f in sorted(glob.glob(os.path.join(search, "*_predictions.csv")))
             if "summit_only" not in os.path.basename(f)]
    if not files:
        sys.exit(f"No *_predictions.csv found under {pred_dir} (or its predictions/ subdir)")
    _validate_fold_files(files, "encoder", allow_partial=allow_partial)
    frames = []
    for f in files:
        d = pd.read_csv(f)
        pred_col = "preds" if "preds" in d.columns else ("Pred" if "Pred" in d.columns else "pred")
        fold = d["fold"] if "fold" in d.columns else _fold_from_name(f)
        frames.append(pd.DataFrame({"pred": d[pred_col], "actual": d["actual"], "fold": fold}))
    return pd.concat(frames, ignore_index=True), f"{len(files)} prediction file(s)"


def _plot(df, overall, out_png, label, xlabel, ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (skipping plot: matplotlib unavailable — {e})")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["actual"], df["pred"], s=6, alpha=0.35, edgecolors="none")
    lo = float(min(df["actual"].min(), df["pred"].min()))
    hi = float(max(df["actual"].max(), df["pred"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label}\nPearson R = {overall['pearsonr']:.4f}  (n={overall['n']})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  scatter -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["expression", "encoder"])
    ap.add_argument("--pred_dir", required=True, help="training/seqencoder output directory")
    ap.add_argument("--out", default=None, help="directory for summary CSV + scatter (default: --pred_dir)")
    ap.add_argument("--label", default=None, help="title/label for the plot and summary")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Permit pooling an incomplete 12-fold CV (some folds missing). "
                         "Off by default so a single finished fold cannot masquerade as a "
                         "full out-of-fold result.")
    args = ap.parse_args()

    out_dir = args.out or args.pred_dir
    os.makedirs(out_dir, exist_ok=True)
    label = args.label or f"{os.path.basename(os.path.normpath(args.pred_dir))} ({args.mode})"

    if args.mode == "expression":
        df, src = _load_expression(args.pred_dir, allow_partial=args.allow_partial)
        xlabel, ylabel = "Observed expression", "Predicted expression"
    else:
        df, src = _load_encoder(args.pred_dir, allow_partial=args.allow_partial)
        xlabel, ylabel = "Observed log2 activity", "Predicted log2 activity"

    overall = _pearson_r2_mse(df["pred"].values, df["actual"].values)

    # per-fold
    rows = []
    for fold, g in df.groupby("fold"):
        m = _pearson_r2_mse(g["pred"].values, g["actual"].values)
        rows.append(dict(fold=fold, **m))
    per_fold = pd.DataFrame(rows).sort_values("fold")

    print(f"\n=== {label} ===")
    print(f"source: {src}   genes/bins: {overall['n']}")
    print(f"OVERALL (out-of-fold pooled):  Pearson R = {overall['pearsonr']:.4f}   "
          f"R^2 = {overall['r2']:.4f}   Spearman = {overall['spearmanr']:.4f}   MSE = {overall['mse']:.4f}")
    if len(per_fold) > 1:
        mean_r = per_fold["pearsonr"].mean()
        std_r = per_fold["pearsonr"].std()
        print(f"PER-FOLD Pearson R: mean {mean_r:.4f} +/- {std_r:.4f} over {len(per_fold)} folds")
        print(per_fold[["fold", "pearsonr", "r2", "mse", "n"]].to_string(index=False))

    summary_path = os.path.join(out_dir, f"{args.mode}_eval_summary.csv")
    overall_row = pd.DataFrame([dict(fold="ALL", **overall)])
    pd.concat([overall_row, per_fold], ignore_index=True).to_csv(summary_path, index=False)
    print(f"  summary -> {summary_path}")

    if not args.no_plot:
        _plot(df, overall, os.path.join(out_dir, f"{args.mode}_scatter.png"), label, xlabel, ylabel)


if __name__ == "__main__":
    main()
