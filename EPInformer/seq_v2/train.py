#!/usr/bin/env python3
"""Train one EPInformer-seq-v2 per-cell main model with a frozen bias model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .dataset import ProfileDSWide
from .model import BiasNet, PerCellProfileNetWide, count_mse, multinomial_nll


def load_splits(path: str, fold: int):
    table = pd.read_csv(path, index_col=0)
    table["chrom"] = "chr" + table["chrom"].astype(str)
    key = f"fold_{fold}"
    return tuple(set(table.loc[table[key] == label, "chrom"]) for label in ("train", "valid", "test"))


def run_epoch(main, bias, loader, device, in_window, out_window, optimizer=None, scheduler=None):
    training = optimizer is not None
    main.train(training)
    losses, predicted, observed = [], [], []
    pad = (in_window - out_window) // 2
    for one_hot, profile, counts in loader:
        one_hot, profile, counts = one_hot.to(device), profile.to(device), counts.to(device)
        if training:
            optimizer.zero_grad()
        with torch.no_grad():
            bias_logits, _ = bias(one_hot[:, :, pad:pad + out_window])
        logits, log_counts = main(one_hot)
        profile_loss = multinomial_nll(logits + bias_logits.detach(), profile)
        count_loss = count_mse(log_counts, counts)
        loss = profile_loss + count_loss
        if not torch.isfinite(loss):
            continue
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        losses.append(float(loss.detach()))
        predicted.append(log_counts.detach().cpu().numpy())
        observed.append(np.log10(counts.detach().cpu().numpy() + 1.0))
    return float(np.mean(losses)), np.concatenate(predicted), np.concatenate(observed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--bias-weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--in-window", type=int, default=2114)
    parser.add_argument("--out-window", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_chrom, valid_chrom, test_chrom = load_splits(args.split_csv, args.fold)
    train_ds = ProfileDSWide(args.h5, train_chrom, args.fasta, reverse_complement=True,
                             in_window=args.in_window, out_window=args.out_window)
    valid_ds = ProfileDSWide(args.h5, valid_chrom, args.fasta, reverse_complement=False,
                             in_window=args.in_window, out_window=args.out_window)
    test_ds = ProfileDSWide(args.h5, test_chrom, args.fasta, reverse_complement=False,
                            in_window=args.in_window, out_window=args.out_window)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, 192, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, 192, shuffle=False, num_workers=args.num_workers)
    bias = BiasNet(); bias.load_state_dict(torch.load(args.bias_weights, map_location="cpu", weights_only=False))
    bias.eval().to(device)
    for parameter in bias.parameters(): parameter.requires_grad_(False)
    model = PerCellProfileNetWide(in_window=args.in_window, out_window=args.out_window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=args.epochs,
                                                     steps_per_epoch=len(train_loader), pct_start=0.1,
                                                     div_factor=10, final_div_factor=100)
    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, bias, train_loader, device, args.in_window, args.out_window, optimizer, scheduler)
        valid_loss, pred, obs = run_epoch(model, bias, valid_loader, device, args.in_window, args.out_window)
        print(f"epoch={epoch + 1} train={train_loss:.5f} valid={valid_loss:.5f}", flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None: raise RuntimeError("no finite training epoch")
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state); torch.save(best_state, output / "main.pt")
    test_loss, pred, obs = run_epoch(model, bias, test_loader, device, args.in_window, args.out_window)
    summary = {"cell": args.cell, "model": "EPInformer-seq-v2", "fold": args.fold,
               "test_loss": test_loss, "test_r_dnase_count": float(stats.pearsonr(pred[:, 0], obs[:, 0]).statistic),
               "test_r_h3k27ac_count": float(stats.pearsonr(pred[:, 1], obs[:, 1]).statistic),
               "args": vars(args)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
