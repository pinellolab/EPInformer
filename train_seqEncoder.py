"""Pre-train the 256bp sequence encoder on peak activity prediction.

Trains `enhancer_predictor_256bp` to predict log2(activity) from 256bp
one-hot-encoded DNA sequences using 12-fold leave-chromosome-out CV.
"""

import os
import random
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import Dataset, WeightedRandomSampler
from Bio.Seq import Seq

from EPInformer.models import enhancer_predictor_256bp
from preprocessing import one_hot_encode

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATTERN = (
    "/home/bingxing2/gpuuser926/project/EPInformer/"
    "data/enhancer_sequences/{}_peak_5bins_around_summit_activity_sequence.csv"
)


class PeakActivityDataset(Dataset):
    """256bp peak-bin sequences with activity labels."""

    def __init__(self, cell_name, chrom_list, strand="both",
                 csv_path=None, dataframe=None, summit_only=False):
        self.strand = strand
        if dataframe is not None:
            dnase_df = dataframe.copy()
        elif csv_path is not None:
            dnase_df = pd.read_csv(csv_path)
        else:
            dnase_df = pd.read_csv(DEFAULT_CSV_PATTERN.format(cell_name))
        dnase_df = dnase_df.rename(columns={"Offset_to_summit": "Pos"})
        dnase_df = dnase_df[dnase_df["Chromosome"].isin(chrom_list)]
        if summit_only:
            dnase_df = dnase_df[dnase_df["Pos"] == 0]
        self.dnase_df = dnase_df.reset_index(drop=True)

    def sample_weights(self, temperature=1.0):
        """Compute per-sample weights that upsample high-activity bins.

        Weight = softmax(log2_activity / temperature), so higher activity
        gets sampled more often.  temperature controls how aggressive the
        upsampling is (lower = more aggressive, 1.0 = moderate).
        """
        log2_act = np.log2(0.1 + self.dnase_df["Activity"].values)
        scaled = log2_act / temperature
        # shift for numerical stability before exp
        weights = np.exp(scaled - scaled.max())
        weights /= weights.sum()
        return torch.from_numpy(weights).double()

    def __len__(self):
        return len(self.dnase_df)

    def __getitem__(self, idx):
        row = self.dnase_df.iloc[idx]
        activity = np.log2(0.1 + row["Activity"])
        seq = row["Sequence"]
        seq_name = f"{row['Name']}_{row['Pos']}"
        if self.strand == "both":
            if random.random() > 0.5:
                seq = str(Seq(seq).reverse_complement())
        elif self.strand == "reverse":
            seq = str(Seq(seq).reverse_complement())
        ohe_seq = one_hot_encode(seq)[None, :, :]
        return ohe_seq, activity, seq_name


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class L1KLmixed(nn.Module):
    """Weighted combination of L1 loss and KL divergence."""

    def __init__(self, reduction="batchmean", alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss(reduction=reduction.replace("batch", ""))
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, preds, targets):
        preds_log = preds - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        l1_loss = self.l1(preds, targets)
        kl_loss = self.kl(preds_log, target_log)
        return (l1_loss * self.alpha + kl_loss * self.beta) / (self.alpha + self.beta)


# ---------------------------------------------------------------------------
# Logger & EarlyStopping
# ---------------------------------------------------------------------------

class Logger:
    """Simple tabular metric logger."""

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
            print("\t".join(
                str(round(x, 4) if isinstance(x, float) else x) for x in row
            ))

    def save(self, path):
        pd.DataFrame(self.data).to_csv(path, sep="\t", index=False)


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience=3, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, epoch_i):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, epoch_i)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience} "
                  f"(best={self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, epoch_i)
            self.counter = 0

    def _save(self, val_loss, model, epoch_i):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} "
                  f"--> {val_loss:.6f}).  Saving model ...")
        torch.save({"epoch": epoch_i, "model_state_dict": model.state_dict(),
                     "loss": val_loss}, self.path)
        print(f"Saving ckpt at {self.path}")
        self.val_loss_min = val_loss


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def _make_loader(dataset, batch_size, shuffle=False, pin_memory=False, num_workers=0):
    persistent = num_workers > 0
    return data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )


def train(net, train_ds, fold_i, saved_model_path="./models/",
          learning_rate=1e-4, model_logger=None, valid_dataset=None,
          model_name="", batch_size=64, device="cuda", EPOCHS=100,
          num_workers=0, upsample=False, upsample_temp=1.0):
    os.makedirs(saved_model_path, exist_ok=True)
    pin = device == "cuda"

    upsample_tag = ""
    if upsample:
        weights = train_ds.sample_weights(temperature=upsample_temp)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                        replacement=True)
        upsample_tag = f"  upsample=True (temp={upsample_temp})"
    else:
        sampler = None

    print(f"fold {fold_i}  train: {len(train_ds)}  "
          f"valid: {len(valid_dataset)}{upsample_tag}")

    persistent = num_workers > 0
    trainloader = data_utils.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
    )
    ckpt_path = f"{saved_model_path}/fold_{fold_i}_best_{model_name}_checkpoint.pt"
    early_stopping = EarlyStopping(patience=5, verbose=True, path=ckpt_path)

    criterion = L1KLmixed()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                  weight_decay=1e-6)

    for epoch in range(EPOCHS):
        net.train()
        lr = optimizer.param_groups[0]["lr"]
        print(f"learning rate: {lr}")
        running_loss = 0.0

        for ohe_seq, y_expr, _ in tqdm(trainloader):
            optimizer.zero_grad(set_to_none=True)
            ohe_seq = ohe_seq.float().to(device)
            y_expr = y_expr.float().to(device)
            loss = criterion(net(ohe_seq), y_expr)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"[Epoch {epoch + 1}] loss: {avg_loss:.9f}")

        val_mse, val_r2, val_pearson = validate(
            net, valid_dataset, device=device, num_workers=num_workers)
        print(f"Validation R²: {val_r2:.4f}")
        early_stopping(-val_r2, net, epoch)

        if model_logger is not None:
            label_type = net.name.split(".")[-1]
            model_logger.add([fold_i, epoch, avg_loss, val_mse,
                              val_pearson, val_r2, val_pearson, val_r2,
                              early_stopping.counter, label_type])
        if early_stopping.early_stop:
            print("Early stopping")
            break


def validate(net, valid_ds, batch_size=1024, device="cuda", num_workers=0):
    pin = device == "cuda"
    validloader = _make_loader(valid_ds, batch_size, pin_memory=pin,
                               num_workers=num_workers)
    net.eval()
    criterion = L1KLmixed()
    all_preds, all_actual = [], []
    total_loss = 0.0

    with torch.no_grad():
        for ohe_seq, y_expr, _ in tqdm(validloader):
            ohe_seq = ohe_seq.float().to(device)
            y_expr = y_expr.float().to(device)
            pred_expr = net(ohe_seq)
            total_loss += criterion(pred_expr, y_expr).item()
            all_preds.append(pred_expr.flatten().cpu())
            all_actual.append(y_expr.flatten().cpu())

    preds = torch.cat(all_preds).numpy()
    actual = torch.cat(all_actual).numpy()

    try:
        _, _, r_value, _, _ = stats.linregress(preds, actual)
        pearson_r, _ = stats.pearsonr(preds, actual)
    except Exception:
        pearson_r, r_value = 0.0, 0.0

    mse = float(np.mean((preds - actual) ** 2))
    print(f"Validation loss: {total_loss / len(validloader):.6f}")
    print(f"valid: mse {mse:.4f}  R² {r_value**2:.4f}  Pearson r {pearson_r:.4f}")
    return mse, r_value ** 2, pearson_r


def _rc_ohe(ohe_seq):
    """Reverse-complement one-hot tensor: flip sequence order and swap A↔T, C↔G.

    Input shape: (B, 1, L, 4).  Channel order is [A, C, G, T].
    RC = reverse along L and reverse channel order [T, G, C, A] → [A, C, G, T] swap.
    """
    return ohe_seq.flip(dims=[-2, -1])


def test(net, test_ds, fold_i, model_name=None, saved_model_path=None,
         batch_size=64, device="cuda", num_workers=0, rc_average=False):
    pin = device == "cuda"
    testloader = _make_loader(test_ds, batch_size, pin_memory=pin,
                              num_workers=num_workers)
    if saved_model_path is not None:
        ckpt_path = f"{saved_model_path}/fold_{fold_i}_best_{model_name}_checkpoint.pt"
        checkpoint = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"{model_name} fold {fold_i} loaded!")

    net.eval()
    all_preds, all_actual, all_names = [], [], []

    with torch.no_grad():
        for ohe_seq, y_expr, seq_name in tqdm(testloader):
            ohe_seq = ohe_seq.float().to(device)
            pred_fwd = net(ohe_seq)
            if rc_average:
                pred_rc = net(_rc_ohe(ohe_seq))
                pred_expr = (pred_fwd + pred_rc) / 2.0
            else:
                pred_expr = pred_fwd
            all_preds.append(pred_expr.flatten().cpu())
            all_actual.append(y_expr.flatten())
            all_names.extend(seq_name)

    preds = torch.cat(all_preds).numpy()
    actual = torch.cat(all_actual).numpy()
    pearson_r, _ = stats.pearsonr(preds, actual)

    preds_df = pd.DataFrame({
        "preds": preds, "actual": actual, "ensid": all_names
    })
    preds_df["fold"] = fold_i
    print(f"\nPearson R: {pearson_r:.4f}" +
          (" (RC-averaged)" if rc_average else ""))
    return preds_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-train 256bp sequence encoder on peak activity")
    parser.add_argument("--cell", type=str, default="HepG2", help="cell type")
    parser.add_argument("--data-csv", type=str, default=None,
                        help="Path to peak activity sequence CSV. "
                             "Overrides --cell for data loading.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="max training epochs")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="fold numbers to run (e.g. --folds 1 2). "
                             "Default: all 12 folds.")
    parser.add_argument("--num-workers", type=int,
                        default=min(4, os.cpu_count() or 1),
                        help="DataLoader workers (default: min(4, cpu_count))")
    parser.add_argument("--output-dir", type=str, default="./results/seqencoder",
                        help="Root output directory for checkpoints, predictions, "
                             "and summary (default: ./results/seqencoder)")
    parser.add_argument("--test-summit-only", action="store_true",
                        help="Additionally evaluate on summit-only (Pos==0) test set")
    parser.add_argument("--rc-average", action="store_true",
                        help="Average forward and reverse-complement predictions at test time")
    parser.add_argument("--upsample", action="store_true",
                        help="Upsample high-activity samples via weighted sampling")
    parser.add_argument("--upsample-temp", type=float, default=1.0,
                        help="Temperature for upsampling weights. "
                             "Lower = more aggressive (default: 1.0)")
    args = parser.parse_args()

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- CV splits ---
    split_df = pd.read_csv(
        "./data/leave_chrom_out_crossvalidation_split_18377genes.csv", index_col=0)
    split_df["chrom"] = "chr" + split_df["chrom"]

    # --- Cell list ---
    if args.cell == "all":
        cell_list = ["NHEK", "HUVEC", "HepG2", "H1", "GM12878", "K562v2"]
    else:
        cell_list = [args.cell]

    output_root = args.output_dir
    model_dir = os.path.join(output_root, "checkpoints")
    pred_dir = os.path.join(output_root, "predictions")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    test_strand = "forward" if args.rc_average else "reverse"
    fold_range = args.folds or list(range(1, 13))
    all_summary_rows = []

    for cell in cell_list:
        # Read CSV once, share across folds
        if args.data_csv:
            full_df = pd.read_csv(args.data_csv)
        else:
            full_df = pd.read_csv(DEFAULT_CSV_PATTERN.format(cell))

        cell_results = []
        for fi in fold_range:
            fold_key = f"fold_{fi}"
            train_chrom = list(split_df[split_df[fold_key] == "train"]["chrom"].unique())
            valid_chrom = list(split_df[split_df[fold_key] == "valid"]["chrom"].unique())
            test_chrom = list(split_df[split_df[fold_key] == "test"]["chrom"].unique())

            train_ds = PeakActivityDataset(cell, train_chrom, strand="both",
                                           dataframe=full_df)
            valid_ds = PeakActivityDataset(cell, valid_chrom, strand="forward",
                                           dataframe=full_df)
            test_ds = PeakActivityDataset(cell, test_chrom, strand=test_strand,
                                          dataframe=full_df)

            model = enhancer_predictor_256bp().to(device)
            model.name = f"enhancer_predictor_log2_H3K27ac_256bp_{cell}"
            print(f"model name: {model.name}")

            train(model, train_ds, valid_dataset=valid_ds,
                  learning_rate=0.0005, EPOCHS=args.epochs,
                  model_name=model.name, fold_i=fi,
                  batch_size=args.batch_size, device=device,
                  saved_model_path=model_dir,
                  num_workers=args.num_workers,
                  upsample=args.upsample,
                  upsample_temp=args.upsample_temp)

            # Save last checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model.name,
                "fold_i": fi, "cell": cell,
            }, f"{model_dir}/fold_{fi}_last_{model.name}_checkpoint.pt")

            preds_df = test(model, test_ds, model_name=model.name,
                            saved_model_path=model_dir, fold_i=fi,
                            batch_size=128, device=device,
                            num_workers=args.num_workers,
                            rc_average=args.rc_average)
            preds_df["cell"] = cell
            cell_results.append(("all_bins", preds_df))

            # Per-fold summary (all bins)
            fold_r, _ = stats.pearsonr(preds_df["preds"], preds_df["actual"])
            fold_mse = float(np.mean((preds_df["preds"] - preds_df["actual"]) ** 2))
            all_summary_rows.append({
                "cell": cell, "fold": fi, "test_set": "all_bins",
                "n_samples": len(preds_df),
                "pearson_r": fold_r, "mse": fold_mse,
                "upsample": args.upsample,
                "upsample_temp": args.upsample_temp if args.upsample else None,
            })

            # Summit-only test
            if args.test_summit_only:
                summit_ds = PeakActivityDataset(
                    cell, test_chrom, strand=test_strand,
                    dataframe=full_df, summit_only=True)
                print(f"  Summit-only test set: {len(summit_ds)} samples")
                summit_preds = test(model, summit_ds, model_name=model.name,
                                    saved_model_path=model_dir, fold_i=fi,
                                    batch_size=128, device=device,
                                    num_workers=args.num_workers,
                                    rc_average=args.rc_average)
                summit_preds["cell"] = cell
                cell_results.append(("summit_only", summit_preds))

                s_r, _ = stats.pearsonr(summit_preds["preds"], summit_preds["actual"])
                s_mse = float(np.mean((summit_preds["preds"] - summit_preds["actual"]) ** 2))
                all_summary_rows.append({
                    "cell": cell, "fold": fi, "test_set": "summit_only",
                    "n_samples": len(summit_preds),
                    "pearson_r": s_r, "mse": s_mse,
                    "upsample": args.upsample,
                    "upsample_temp": args.upsample_temp if args.upsample else None,
                })

        # Save predictions and compute overall metrics per test_set
        for test_set_name in dict.fromkeys(tag for tag, _ in cell_results):
            subset = pd.concat([df for tag, df in cell_results if tag == test_set_name])
            suffix = f"_{test_set_name}" if test_set_name != "all_bins" else ""
            out_path = os.path.join(
                pred_dir, f"enhancer_predictor_H3K27ac_256bp_{cell}_{test_strand}{suffix}_predictions.csv")
            subset.to_csv(out_path, index=False)

            overall_r, _ = stats.pearsonr(subset["preds"], subset["actual"])
            overall_mse = float(np.mean((subset["preds"] - subset["actual"]) ** 2))
            all_summary_rows.append({
                "cell": cell, "fold": "ALL", "test_set": test_set_name,
                "n_samples": len(subset),
                "pearson_r": overall_r, "mse": overall_mse,
                "upsample": args.upsample,
                "upsample_temp": args.upsample_temp if args.upsample else None,
            })
            print(f"{cell} [{test_set_name}]  n={len(subset)}  Pearson R: {overall_r:.4f}")
            print(f"  Predictions saved at: {out_path}")

    # --- Write aggregate summary ---
    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = os.path.join(output_root, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Aggregate summary saved at: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
