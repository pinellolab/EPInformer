# Legacy BSCC_GPU EPInformer vs this repository

## Canonical location of old scripts

The Pinello-style training stack and notebooks that originally produced the **200-CRE** legacy HDF5 live under:

**`/lustre/grp/zyjlab/linjc/BSCC_GPU/BSCC_GPU/EPInformer`**

Training scripts there expect pre-encoded HDF5 at:

`./data/{CELL}_200CREs-gene_RPM_4feats.hdf5`

(copied to `/dev/shm/data/` at runtime in `train_EPInformer_abc.py`).

Related modules: `epinformer_preprocessing/`, `seqs2expr_io.py`, `encoding_isoform_data.py` (isoform-specific encodes with the same schema but different `out_fn` / slot counts).

## Provenance of `*_200CREs-gene_RPM_4feats.hdf5`

The file is **written by the notebook** `process_EPInformer_data_v2.ipynb` in that tree (not by a standalone `.py` checked in under the same name). The encoding cell calls `encode_pe_pair(...)` with:

- `max_n_enh=200`
- `out_fn = 'CREs-gene_RPM_4feats'`
- `enh_features=['DHS.RPM', 'H3K27ac.RPM', 'activity_base_no_qnorm', 'hic_contact']`

Each row of `enhancers_feat` is built as:

`[distance] + [row[f] for f in enh_features]`

So **`enhancers_feat` has 5 columns**:

| Index | Field |
|------|--------|
| 0 | Signed distance (strand-aware; training uses `abs`) |
| 1 | DHS.RPM |
| 2 | H3K27ac.RPM |
| 3 | `activity_base_no_qnorm` |
| 4 | `hic_contact` |

The training code (old and new legacy dataset) selects **three** scalar channels for `EPInformer-abc` as:

`concatenate([abs(col 0), col 3, col -1])` → **distance magnitude, activity_base_no_qnorm, hic_contact**

So the middle two columns (DHS.RPM, H3K27ac.RPM) are **not** used when `n_enh_feats=3`, despite being present in the file.

## Factored HDF5 in this repo

New preprocessing writes shared `enhancer_seq`, `gene_enh_idx`, and separate `distance`, `activity`, `contact`, `dhs`, etc. (see `preprocessing/hdf5.py` and `train_EPInformer_abc.py` `promoter_enhancer_dataset`). The three-feature path stacks `[abs(distance), activity, contact]` from those arrays (with NaNs sanitized at load time).

**Interpretation:** Legacy “activity” for the 3-feat path is **`activity_base_no_qnorm`** (column 3), not the same tensor as factored `activity` from ABC preprocessing unless you align definitions explicitly.

**100 kb window:** Factored HDF5 from `run_pipeline` / `run_k562_preprocessing` uses `max_distance: 100000` (see `config/config.yaml`). The legacy file can list enhancers **beyond** 100 kb (up to 200 CREs). For a fair feature comparison, filter legacy rows with `abs(enhancers_feat[:,0]) <= 100000`. Use `python scripts/compare_h5_random_genes.py --max-distance-bp 100000 ...` (`--global-only` for dataset-wide in-window counts and contact NaN rate; `--include-global-stats` to prepend the same summary when also sampling random genes).

## Fair benchmark checklist (old vs new)

When comparing runs, align as many of these as possible:

| Item | This repo (`train_EPInformer_abc.py`) | Old BSCC_GPU `train_EPInformer_abc.py` |
|------|--------------------------------------|----------------------------------------|
| Expression CSV | `--expr_csv` (default `./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv`) | Hardcoded `./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv` |
| Cross-val splits | `--split_csv` (default `./data/leave_chrom_out_crossvalidation_split_18377genes.csv`) | Hardcoded `./data/leave_chrom_out_crossvalidation_split_18377genes_addBorzoi_enhancer_annot.csv` |
| Folds | `--fold` / `--folds` / default all 1–12 | Loops **all** folds `1..12` (no CLI to subset) |
| Epochs | `--epochs` (default 50) | `EPOCHS=50` fixed in script |
| Batch size | `--batch_size` (default 50) | `50` |
| `max_n_enh` | From HDF5 / model (factored) or legacy loader | **60** in training loop (subsample/truncate from 200-CRE file) |
| `dist_thr` | Dataset-specific | **100000** bp |
| Self-promoter | `--rm_self_promoter` on factored HDF5 | **No** equivalent flag in old script |
| HDF5 | `--h5_path` factored `samples.h5` | `./data/{cell}_200CREs-gene_RPM_4feats.hdf5` |

For a strict apples-to-apples replication of an **old** run, use the **same split CSV** as the legacy script if your goal is to match published fold splits (`addBorzoi` annot file).

## HDF5 quality stats (NaNs and ranges)

Use the read-only helper:

```bash
python scripts/compare_h5_legacy_factored_stats.py \
  --legacy-h5 /path/to/K562_200CREs-gene_RPM_4feats.hdf5 \
  --factored-h5 /path/to/samples.h5
```

Omit either flag to skip that side. The script reports NaN counts and finite min/max/mean for legacy `enhancers_feat` columns and for factored `activity` / `contact` (and mask statistics when a padding mask is available).

**Example (K562, paths on this cluster):** legacy `K562_200CREs-gene_RPM_4feats.hdf5` had **no NaNs** in any `enhancers_feat` column. Factored `batch_output/K562/encoding/K562_samples.h5` had **~0.17%** NaN values in `contact` overall, and **~0.75%** NaN among slots where `gene_enh_idx >= 0` (training loads apply `nan_to_num` on activity/contact). Re-run the script after regenerating HDF5 to refresh these figures.
