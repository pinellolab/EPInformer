#!/usr/bin/env python3
"""Train the frozen 1,024-bp sequence-bias model for one cell."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ProfileDSWide
from .model import BiasNet, count_mse, multinomial_nll


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="HDF5 containing a non-empty bias group")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-chroms", nargs="+", required=True)
    parser.add_argument("--valid-chroms", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    random.seed(777); np.random.seed(777); torch.manual_seed(777)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = ProfileDSWide(args.h5, args.train_chroms, args.fasta, group="bias",
                          reverse_complement=True, in_window=1024, out_window=1024)
    valid = ProfileDSWide(args.h5, args.valid_chroms, args.fasta, group="bias",
                          reverse_complement=False, in_window=1024, out_window=1024)
    if not len(train) or not len(valid):
        raise ValueError("bias group must contain training and validation windows")
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid, 256, shuffle=False, num_workers=args.num_workers)
    model = BiasNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                                     epochs=args.epochs,
                                                     steps_per_epoch=len(train_loader),
                                                     pct_start=0.1, div_factor=10,
                                                     final_div_factor=100)

    def epoch(loader, training):
        model.train(training); values = []
        for one_hot, profile, counts in loader:
            one_hot, profile, counts = one_hot.to(device), profile.to(device), counts.to(device)
            if training: optimizer.zero_grad()
            logits, log_counts = model(one_hot)
            loss = multinomial_nll(logits, profile) + count_mse(log_counts, counts)
            if not torch.isfinite(loss): continue
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
            values.append(float(loss.detach()))
        return float(np.mean(values)) if values else float("inf")

    best, state = float("inf"), None
    for index in range(args.epochs):
        train_loss, valid_loss = epoch(train_loader, True), epoch(valid_loader, False)
        print(f"epoch={index + 1} train={train_loss:.5f} valid={valid_loss:.5f}", flush=True)
        if valid_loss < best:
            best = valid_loss
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if state is None: raise RuntimeError("no finite bias training epoch")
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save(state, output / "bias.pt")


if __name__ == "__main__": main()
