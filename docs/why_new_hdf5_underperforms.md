# Why validation scores can drop on newly generated HDF5

New training files from `[run_pipeline.py](../run_pipeline.py)` (Stage 1: `[run_abc_pipeline](../preprocessing/abc/__init__.py)`; Stage 2: `[obtain_PE_withSignals](../preprocessing/pipelines_legacy.py)`) using `[config/config.yaml](../config/config.yaml)` are **not bit-for-bit equivalent** to older artifacts (legacy Pinello-style HDF5, `[run_k562_preprocessing.py](../run_k562_preprocessing.py)`, Zenodo releases, or fixed ABC **EnhancerPredictionsAllPutative** tables). Until inputs and flags are aligned, **Pearson R and related metrics will not match** prior runs.

## 1. Supervision and feature tables (`expression_csv`)


| Source                                                          | Typical file                                                                                                           |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Batch pipeline (default)                                        | `reference.expression_csv` → often `[data/roadmap_expression/roadmap_expression_all.csv](../data/roadmap_expression/)` |
| `[train_EPInformer_abc.py](../train_EPInformer_abc.py)` default | `[data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv](../data/)`                                   |


If one run uses the **Roadmap-only** CSV (no Xpresso gene-structure columns) and another uses the **Xpresso-merged** CSV, the dataloader sets `has_rna_feats` from column presence: the eight columns in `_all_rna_cols` inside `promoter_enhancer_dataset` (`UTR5LEN_log10zscore`, `CDSLEN_log10zscore`, … `ORFEXONDENSITY`). Missing columns → **zero `rna_feats`** and `useFeat` behavior changes — metrics move a lot.

**Fair comparison:** use the **same** `--expr_csv` for both training jobs and confirm the same Xpresso columns exist.

## 2. Legacy format vs factored `samples.h5`

- **Legacy:** `promoter_ohe` / stacked PE tensors and `[promoter_enhancer_dataset_legacy](../train_EPInformer_abc.py)`.
- **New:** factored `promoter_seq`, `enhancer_seq`, `gene_enh_idx`, etc. and `promoter_enhancer_dataset`.

Padding, ordering, and distance handling differ. Expect a gap until the **encoding recipe** matches the reference file.

**Reference:** [legacy_epinformer_comparison.md](legacy_epinformer_comparison.md) (old HDF5 path, column semantics, benchmark table).

## 3. ABC / peaks source

- **New Stage 1** runs MACS2 (or supplied peaks) on **your** accessibility BAM and rebuilds links.
- **Published** EPInformer HDF5 often used a **fixed** ABC release (different peaks, distances, activity, Hi-C merge).

Different peaks → different sequences and ABC features → different learnability.

## 4. Hi-C `contact` quality and NaNs

Missing Hi-C for many pairs yields **NaNs** in `contact` in new encodes; training applies `nan_to_num` (often **0**). If an older HDF5 had **dense** or differently imputed contacts, the third enhancer channel carries more signal.

**Checks:** run `[scripts/compare_h5_legacy_factored_stats.py](../scripts/compare_h5_legacy_factored_stats.py)` on old vs new paths. **Ablation:** train with `--n_enh_feats 2` (distance + activity only); if metrics improve, contact noise or sparsity is a bottleneck.

**Upstream:** improve `[preprocessing/abc/contact.py](../preprocessing/abc/contact.py)` / resolution / imputation to match a reference strategy.

## 5. Self-promoter flags (Stage 1 vs Stage 2)

`[config/config.yaml](../config/config.yaml)` can set:

- `abc_params.include_self_promoter` — whether Stage 1 ABC linking includes self-promoter logic consistently with your expectations.
- `preprocessing_params.include_self_promoter` — whether Stage 2 pulls self-promoter rows from the predictions TSV into slot 0.

Example in the default config: Stage 1 has `include_self_promoter: false` while Stage 2 has `include_self_promoter: true`. Stage 2 only adds self-promoter rows **if they appear in predictions**; mismatch with [CLAUDE.md](../CLAUDE.md) “critical” self-promoter behavior can hurt benchmarks.

**Experiment:** try `[config/parity_k562_training.yaml](../config/parity_k562_training.yaml)` (`abc_params.include_self_promoter: true` + Xpresso `expression_csv`) and re-run links + encoding before re-training.

## 6. Training parity

Smoke tests (`--epochs 3`, one fold) are **not** comparable to fully trained checkpoints. Match `**--fold`**, `**--epochs**`, early stopping, `**batch_size**`, and seed where possible.

---

## Old HDF5 provenance (for “what was the previous file?”)


| Artifact                                  | Where it comes from                                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `*_200CREs-gene_RPM_4feats.hdf5`          | Encoded in BSCC_GPU tree notebook `**process_EPInformer_data_v2.ipynb**` (see [legacy_epinformer_comparison.md](legacy_epinformer_comparison.md)) |
| Factored `K562_samples.h5` from this repo | `[run_pipeline.py](../run_pipeline.py)` + config, or `[run_k562_preprocessing.py](../run_k562_preprocessing.py)`                                  |


---

## Suggested verification order

1. **Name the baseline HDF5** (path + how it was built).
2. **Align `--expr_csv`** (Roadmap vs Xpresso-merged).
3. **Compare distributions:** NaNs in `contact`/`activity`, gene count, enhancers per gene — use `scripts/compare_h5_legacy_factored_stats.py`.
4. **Optional ablation:** `--n_enh_feats 2` to test contact as bottleneck.
5. **Optional pipeline parity:** `python run_pipeline.py --config config/parity_k562_training.yaml` (adjust `samples_table` / paths as needed).

---

## Optional later code follow-ups

- CLI or write-time **contact imputation** (chrom mean, zero, nearest bin) in `[preprocessing/pipelines_legacy.py](../preprocessing/pipelines_legacy.py)`.
- Extend parity presets under `[config/](../config/)` for other cell types.

