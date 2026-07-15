# EPInformer pipeline (models.py)

A self-contained pipeline to run the **EPInformer** gene-expression model end to end,
using the model defined in [`EPInformer/models.py`](EPInformer/models.py)
(`EPInformer_v2`). Covers everything from raw ENCODE data to a trained, evaluated
model. The full expression model runs for **K562** and **GM12878**; the
enhancer-activity **encoder** additionally covers **H1, HepG2, HUVEC, and
NHEK** — 6 cell lines total (see [Other cell lines](#other-cell-lines-encoder)).

1. **Enhancer–gene nomination** (ABC) from ENCODE BAM/Hi-C → nominated links
2. **Enhancer-activity pretrain data** (256 bp bins) → pretrain the sequence encoder
3. **Enhancer–gene data** (per-gene promoter + up to 60 enhancers) → HDF5
4. **Full EPInformer training** (12-fold leave-chromosome-out CV)
5. **Evaluation + visualization** (Pearson R, scatter plots)

> **Model note.** This uses `EPInformer_v2` from **`EPInformer/models.py`**, which
> differs slightly (the `conv_out` block) from the `models_abc.py` version used by
> the upstream released checkpoints. So the model is **trained from scratch** here — the
> published checkpoints will not load. The 256 bp sequence encoder is identical, so
> a pretrained encoder is fully compatible.

**Headline numbers** (K562, `EPInformer_v2` PE-Activity, 12-fold leave-chromosome-out):

| Task | Metric | Target | Result (pooled OOF) |
|---|---|---|---|
| Gene expression (CAGE) | Pearson R | ~0.88 | **0.871** |
| Gene expression (RNA-seq) | Pearson R | ~0.86 | **0.854** |
| Enhancer-activity encoder (K562) | Pearson R | ~0.71 | **0.738** fwd+RC / 0.731 single |

GM12878 also runs (encoder **0.617**, expression RNA/CAGE **0.834 / 0.877** with the
consistent 2+2-replicate activity), and the encoder reaches these numbers across all six cell
lines — full numbers in [Results](#results--recipe-notes) below.

---

## Results & recipe notes

All metrics are **pooled out-of-fold Pearson R** on the **`log2(0.1 + Activity)`** scale
(concatenate all 12 held-out folds → one Pearson). Note: the encoder target is *log2* activity —
if a stored prediction file holds *raw* activity, a naive Pearson is misleading (0.70 vs the true
0.74).

**Encoder recipe (what reaches target).** 5 bins at `summit + 192·{−2..2}` (256 bp),
loss **L1KLmixed**, batch 256, LR 5e-4, `Activity = sqrt(H3K27ac_RPM · DNase_RPM)`,
target `log2(0.1 + Activity)`. K562 pooled OOF by recipe: 3-bins/MSE 0.659 · **5-bins/L1KL
0.738** (fwd+RC) · 2-bins/bs64/MSE 0.672. Loss matters: MSE mis-calibrates the per-fold
scale and drags pooled down. Using the K562 narrowPeak `ENCFF544LXB` (54,625 peaks) as the
summit source gives pooled OOF **0.734 (rev) / 0.740 (fwd+RC)**. Peak choice barely moves K562
(0.738 → 0.740): its H3K27ac is deep enough that the summit set doesn't matter. **K562 ships
single-rep by design**: unlike GM12878, pooling its H3K27ac replicates *hurts* — mean-pooling all
3 reps (0.745 → **0.729**) dilutes the deep rep3 (`ENCFF232RQF`, 70.7M reads) with the shallow
reps 1/2 (10.7M/4.27M), since mean weights each rep equally (the `K562v2` CSV has no per-rep
columns either). So `samples.tsv` leaves K562's `accessibility_bams`/`h3k27ac_bams` empty; the
2-rep-pool path is GM12878-only (its reps are similar depth, so pooling *adds* signal → 0.617).

**Expression (`EPInformer_v2`, frozen encoder).** Shipped headline: K562 RNA **0.856** /
CAGE **0.869**; GM12878 RNA **0.834** / CAGE **0.877** (on the consistent 2+2-replicate
activity). Four ablations settle the shipped config:

| Axis | K562 RNA | K562 CAGE | GM12878 RNA | GM12878 CAGE | Verdict |
|---|---|---|---|---|---|
| Feature (dist→+act→+HiC) | .753/.855/.856 | .811/.871/.869 | .761/.827/.828 | .827/.869/.872 | **activity dominates, Hi-C ~nothing** |
| + promoter signal (`prm`) | +0.000 | −0.002 | **+0.026** | **+0.014** | **cell-dependent → default on** |
| Fine-tune encoder (no-freeze) | −0.001 | +0.007 | −0.009 | +0.002 | **mixed/marginal → ship frozen** |
| Consistent 2+2 BAMs (GM12878) | — | — | +0.006 | +0.005 | pooled ABC activity helps |

So the headline model **freezes** the encoder, uses **3 enhancer features + the promoter
signal**, and (GM12878 only) quantifies activity from the same 2+2 replicate BAMs in both the
encoder-pretrain and the ABC-feature stages. Promoter signal helps GM12878 (+0.02) but is
neutral on the current K562 encoder; fine-tuning the encoder end-to-end doesn't reliably help
(RNA slightly hurts) so it stays a frozen, canonical design. The encoder *quality* barely
transfers to expression (5-bins-encoder K562 RNA 0.854 ≈ 3-bins-encoder 0.857): the model
fine-tunes the conv head and recovers regardless.

**GM12878 encoder (0.6146 fwd+RC / 0.617).** The gap between an early ~0.57 and the target was
**one missing DNase replicate** — not read depth, not filtering, not pooling arithmetic. The exact
inputs were pinned by matching the reference activity CSV's per-replicate RPM columns against
candidate ENCODE BAMs (`scripts/identify_encoder_reps.py`): every column matches a **filtered**
bio-rep at **corr = 1.0000, ratio 1.000** — `H3K27ac_0`=`ENCFF269GKF`(rep1),
`H3K27ac_1`=`ENCFF201OHW`(rep2), `DNase_0`=`ENCFF729UYK`(rep2), `DNase_1`=`ENCFF020WZB`(rep1).
The activity is `sqrt(mean(2 H3K27ac reps) · mean(2 DNase reps))`. Earlier runs pooled the two
H3K27ac reps but used only **one** DNase rep — that single omission was the entire gap:

| recipe | zero-activity | pooled OOF (rev / fwd+RC) |
|---|---|---|
| single H3K27ac + 1 DNase | 2.52% | 0.530 / 0.541 |
| 2 H3K27ac (sum-pool) + 1 DNase | 1.40% | 0.561 / 0.572 |
| 2 H3K27ac (mean-pool) + 1 DNase | 1.40% | 0.562 / 0.572 |
| **2 H3K27ac + 2 DNase, all filtered (4 BAMs)** | **0.74%** | **0.607 / 0.6146** |
| direct-train on the reference CSV | 0.74% | 0.613 / **0.621** |
| published target | 0.74% | — / **0.617** |

With the 2nd DNase rep, the extracted Activity is **byte-identical** to the reference CSV (aligned
on bin key: corr 1.00000, max|diff| 0), zeros 0.74%, medians identical (H3K27ac 1.197, DNase
0.478). Two hypotheses **falsified** along the way: (a) *mean-vs-sum pooling* — no effect (0.572
either way; a bin is zero iff both reps are zero regardless); (b) *unfiltered BAMs / ABC
read-extension* — read-extension actively hurt (inflates RPM, 0.534), and the corr=1.0
rep-identification confirmed the **filtered** reps. The extractor
`scripts/seq_activity_extract.py` (cells 35+38) pools reps via
`--h3k27ac-bam a.bam b.bam --dnase-bam c.bam d.bam --pool-method mean`. **To run this recipe,
see Steps 1–3** (download the 4 filtered reps via `config/gm12878_encoder_bams.json`;
`config/samples.tsv` already lists them in `accessibility_bams`/`h3k27ac_bams`).

**Target conventions** (all match upstream): encoder `log2(0.1+Activity)` · CAGE
`log10(cell_CAGE_128*3_sum + 1)` · RNA linear `Actual_{cell}`.

### Other cell lines (encoder)

The enhancer-activity encoder reaches these numbers across **all six** cell lines (single-reverse
pooled OOF; fwd+RC adds ~+0.007):

| Cell | R | Cell | R |
|---|---|---|---|
| K562 | 0.740 | HepG2 | 0.743 |
| GM12878 | 0.617 | HUVEC | 0.742 |
| H1 | 0.820 | NHEK | 0.677 |

These four extra cells train on pre-built activity CSVs (passed via
`train_seqEncoder.py --data-csv`), which validates the training recipe cell-by-cell. For
**from-raw** runs the exact ENCODE accessions are wired into the repo — DNase (1 rep) +
H3K27ac (2 reps, except HepG2) + H3K27ac narrowPeak:

| Cell (roadmap) | DNase | H3K27ac reps | narrowPeak |
|---|---|---|---|
| H1 (E003) | ENCFF761ZRE | ENCFF860ABR, ENCFF693IFG | ENCFF689CJG |
| HUVEC (E122) | ENCFF091KTX | ENCFF374DGO, ENCFF609TUB | ENCFF077LGZ |
| NHEK (E127) | ENCFF117RNM | ENCFF770JWP, ENCFF051NTC | ENCFF666UYC |
| HepG2 (E118) | ENCFF691HJY | *(3 downloaded; rep count unresolved)* | ENCFF392KDI *(inferred)* |

Rows are in `config/samples.tsv`; BAM/narrowPeak download manifests in
`config/extra_cells_bams.json` and `config/encoder_narrowpeaks.json`. **HepG2 is
provisional**: 3 filtered H3K27ac reps were downloaded but the CSV has a single H3K27ac column
and the build notebook globs one BAM — RPM-match against the CSV
(`scripts/identify_encoder_reps.py`) to settle the rep count and which rep before trusting, and
its narrowPeak is inferred (not cited).

**Expression (RNA).** The full `EPInformer_v2` expression model also runs for these four
cells — **RNA only** (the dataset carries CAGE labels for K562/GM12878 only; `Actual_{cell}` gives
RNA for all six). Gene HDF5s are built from precomputed ABC links
(`scripts/build_gene_h5_for_cell.py`; no raw BAMs needed; contact = ABC **average Hi-C**), then
trained 12-fold with the frozen per-cell encoder + promoter signal (pooled OOF Pearson):

| Cell | f1 (dist) | f2 (+activity) | f3 (+Hi-C) | **f3 + prm** |
|---|---|---|---|---|
| HepG2 | 0.634 | 0.838 | 0.835 | **0.845** |
| HUVEC | 0.630 | 0.827 | 0.829 | **0.839** |
| NHEK  | 0.605 | 0.818 | 0.817 | **0.828** |
| H1    | 0.661 | 0.780 | 0.772 | **0.781** |

Same signature as K562/GM12878: **activity dominates** (f1→f2 +0.12…+0.21), **Hi-C is inert**
(f2→f3 ≈ 0), **promoter signal** adds ~+0.01. Build + train per cell:
`CELL=HepG2 sbatch slurm/build_gene_h5.slurm` → `CELL=HepG2 EXPR_TYPE=RNA USE_PRM_SIGNAL=1
PRETRAINED_DIR=results/seqencoder/HepG2/checkpoints sbatch slurm/train_epinformer_12fold.slurm`.
(These are NEW numbers — upstream EPInformer reports expression for K562/GM12878 only.)

**Average Hi-C — exact provenance.** The contact feature above uses a precomputed avg-Hi-C. To
rebuild contact from scratch with the canonical hg38 ABC average Hi-C (ENCODE `ENCFF134PUN`, 5 kb,
58 GB, annotation `ENCSR382HAW`): download it, split once —
`python scripts/split_avg_hic.py --in ENCFF134PUN.bed.gz --out data/reference/abc_avg_hic/by_chrom`.
The pipeline selects this default directory automatically when a cell-specific `.hic` file is absent
or unavailable; a valid cell-specific map takes precedence. `predict_abc` auto-detects the
average-Hi-C loader (`AverageHiCContactMap`, alongside the juicebox `.hic` path). Hi-C is inert for
expression here, so this does not change the numbers; `scripts/test_avg_hic.py` self-tests the loader.
For H1/NHEK note there is **no cell-specific Hi-C in ENCODE** (H1: 5C/ChIA-PET only; NHEK: in-situ
Hi-C `ENCSR869CSI`→`ENCFF776JNR`); HepG2/HUVEC do have `.hic` (`ENCFF805ALH`/`ENCFF091YKP`).

---

## Contents

```
EPInformer/models.py        the model (EPInformer_v2 + 256bp encoder)   <- copied verbatim
preprocessing/              data pipeline (model-agnostic)
  abc/                        ABC enhancer-gene nomination (candidates -> neighborhoods
                              -> contact -> predictions = EnhancerPredictionsAllPutative.txt)
  abc/encoder_pretrain_data.py  256bp activity bins for encoder pretraining
  hdf5.py, pipelines_legacy.py  factored {cell}_samples.h5 builder
  data_prep/                  gene annotation / reference / Roadmap expression helpers
scripts/download_encode_data.py    ENCODE BAM/.hic downloader
scripts/download_abc_reference.sh  ABC reference files (gene bounds, chrom sizes, qnorm)
run_pipeline.py             config-driven Stage 1 (links) + Stage 2 (encoding)
run_preprocessing.py, run_abc_pipeline.py   standalone CLIs
config/config.yaml, config/samples.tsv       pipeline config (K562, GM12878, +4 encoder cells)
config/*_bams.json, encoder_narrowpeaks.json  pinned ENCODE download manifests
train_seqEncoder.py         Step 3: pretrain the 256bp enhancer-activity encoder
train_EPInformer.py         Step 5: train EPInformer_v2 (from EPInformer/models.py)
evaluate.py                 Step 6: 12-fold out-of-fold Pearson (expression + encoder)
slurm/                      HPC array-job templates (gpu33)
environment.yml             conda environment
```

---

## 0. Environment

```bash
conda env create -f environment.yml
conda activate epinformer_repro
# On GPU nodes, install the matching CUDA build of torch, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Key deps: `torch`, `h5py`, `pyfaidx`, `kipoiseq`, `pyranges`, `macs2`, `hicstraw`,
`pyBigWig`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `jupyter`.

---

## Reference & label files (once)

These are inputs you provide/download (not generated by the pipeline):

```bash
# ABC reference (gene bounds, chrom sizes, K562 quantile-norm reference)
bash scripts/download_abc_reference.sh data/reference/hg38

# hg38 genome FASTA — supply your own (not downloaded by any script)
#   -> data/reference/hg38/hg38.fa   (+ .fai will be created on first use)

# Expression labels + Xpresso mRNA features + 12-fold CV split (Zenodo 13232430)
#   expression_data.zip contains:
#     GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv
#     leave_chrom_out_crossvalidation_split_18377genes.csv
#   -> unzip into ./data/

# H3K27ac narrowPeaks = the encoder summit source (GM12878 ENCFF023LTU, K562 ENCFF544LXB).
# ENCODE serves them gzipped; fetch + gunzip to the exact names config/samples.tsv expects:
python scripts/download_encode_data.py --from-manifest config/encoder_narrowpeaks.json
gunzip -f reference/GM12878_H3K27ac.ENCFF023LTU.narrowPeak.gz \
          reference/K562_H3K27ac.ENCFF544LXB.narrowPeak.gz
```

Paths are set in [`config/config.yaml`](config/config.yaml). Defaults expect
`data/reference/hg38/hg38.fa` and `data/GM12878_K562_18377_gene_expr_fromXpresso.csv`.

---

## 1. Download ENCODE data (BAM + Hi-C)

BAMs + `.hic` contact maps for both cell lines (auto-queried from ENCODE):

```bash
python scripts/download_encode_data.py --cell-types K562,GM12878 --output-dir data
```

| Cell | DNase BAM | H3K27ac BAM | Hi-C |
|---|---|---|---|
| K562 | ENCFF257HEE | ENCFF232RQF | ENCFF621AIY |
| GM12878 | ENCFF729UYK | ENCFF269GKF | ENCFF318GOM |

Files land in `data/{cell}/{DNase,H3K27ac,HiC}/`. See `config/samples.tsv`.

**GM12878 encoder needs 2 filtered replicates per assay** (this is what reaches 0.617 —
see the results section; the activity is `sqrt(mean(2 H3K27ac rep RPMs) · mean(2 DNase rep
RPMs))`). The auto-query fetches only one "best" BAM per assay, so pull the exact 4 **filtered
bio-rep** BAMs from a pinned manifest, then index them:

```bash
python scripts/download_encode_data.py --from-manifest config/gm12878_encoder_bams.json
for f in data/GM12878/{DNase,H3K27ac}/ENCFF*.bam; do samtools index "$f"; done
```

| GM12878 assay | rep 1 | rep 2 |
|---|---|---|
| H3K27ac | ENCFF269GKF | ENCFF201OHW |
| DNase | ENCFF020WZB | ENCFF729UYK |

These are **filtered** `alignments` BAMs (the encoder reps, RPM-matched at corr 1.0);
distinct from the single unfiltered BAM the ABC/MACS2 peak-caller uses. K562 reaches 0.740
with its single reps, so no extra download is needed there.

---

## 2. ABC enhancer–gene nomination + encoder data (Stage 1)

Runs the ABC model: MACS2 peaks → per-element activity → Hi-C contact →
`ABC.Score` → **`EnhancerPredictionsAllPutative.txt`** (nominated links), and the
256 bp **encoder pretrain CSV**.

```bash
python run_pipeline.py --config config/config.yaml --samples K562,GM12878 --stages links
# (preview first:  python run_pipeline.py --config config/config.yaml --dry-run)
```

Produces per cell under `batch_output/{cell}/links/`:
- `EnhancerList.txt`, `Predictions/EnhancerPredictionsAllPutative.txt` → used by Stage 2
- `{cell}_peak_5bins_around_summit_activity_sequence.csv` → **encoder pretrain data (Step 3)**

HPC: `SAMPLES=K562,GM12878 STAGES=links sbatch slurm/run_pipeline_cpu.slurm`

The encoder activity **mean-pools replicate BAMs per assay** when `config/samples.tsv` lists
them in the `accessibility_bams` / `h3k27ac_bams` columns (space-separated). GM12878 ships with
its 2 reps/assay there, so `--stages links` produces the activity that reaches 0.617; K562
leaves those columns empty and uses its single reps (0.740). This is the
`sqrt(mean(H3K27ac rep RPMs) · mean(DNase rep RPMs))` recipe.

**Verified standalone alternative (no ABC run needed for the encoder CSV):** produce the exact
same GM12878 CSV directly from the 4 BAMs with `scripts/seq_activity_extract.py`:

```bash
python scripts/seq_activity_extract.py \
    --narrowpeak reference/GM12878_H3K27ac.ENCFF023LTU.narrowPeak \
    --dnase-bam data/GM12878/DNase/ENCFF729UYK.bam data/GM12878/DNase/ENCFF020WZB.bam \
    --h3k27ac-bam data/GM12878/H3K27ac/ENCFF269GKF.bam data/GM12878/H3K27ac/ENCFF201OHW.bam \
    --pool-method mean --fasta data/reference/hg38/hg38.fa --cell GM12878 \
    --out batch_output/GM12878/links/GM12878_peak_5bins_around_summit_activity_sequence.csv
# HPC: see the GM12878 example header in slurm/seq_activity_extract.slurm (POOL_METHOD=mean).
```

---

## 3. Pretrain the enhancer-activity encoder

12-fold leave-chromosome-out; target = `log2(0.1 + activity)`:

```bash
python train_seqEncoder.py --cell K562 \
    --data-csv batch_output/K562/links/K562_peak_5bins_around_summit_activity_sequence.csv \
    --loss l1kl --batch-size 256 \
    --output-dir ./results/seqencoder/K562 --epochs 50
```

**Recipe matters** (see results section): `--loss l1kl` (L1KLmixed, not the `mse`
default) + 5-bin data + batch 256 is what reaches ~0.74; MSE mis-calibrates the per-fold
scale. Test is single-reverse-strand by default; add `--rc-average` for the fwd+RC eval
(~+0.007), or re-score existing checkpoints with `slurm/eval_seqencoder_fwdrc.slurm`.

Checkpoints → `results/seqencoder/K562/checkpoints/fold_{i}_best_enhancer_predictor_H3K27ac_256bp_K562_checkpoint.pt`
(this exact name is what the trainer loads in Step 5). Predictions → `.../predictions/`.

HPC (all 12 folds): `CELL=K562 sbatch slurm/train_seqencoder_12fold.slurm`

---

## 4. Build the enhancer–gene HDF5 (Stage 2)

```bash
python run_pipeline.py --config config/config.yaml --samples K562,GM12878 --stages encoding
```

Produces `batch_output/{cell}/encoding/{cell}_samples.h5` (factored: `promoter_seq`,
shared `enhancer_seq` pool, `gene_enh_idx`, `activity/dhs/distance/contact`, `ensid`).
The expression **label is not in the HDF5** — it is joined at train time from the
expression CSV on ENSID. `include_self_promoter=true` in the config is important
(~0.81 vs ~0.63 Pearson without it).

---

## 5. Train EPInformer_v2

12-fold leave-chromosome-out. The **shipped/headline config** = **3 enhancer features**
(`--n_enh_feats 3` = `[distance, activity, Hi-C contact]`) **+ the promoter signal**
(`--use_prm_signal`) **with the pretrained encoder frozen** (the default — matches the
canonical EPInformer design). Two optional knobs, off for the headline:
`--no_freeze_encoder` (slurm `NO_FREEZE=1`) fine-tunes the encoder end-to-end (doesn't reliably
help — see results); dropping `--use_prm_signal` and setting `--n_enh_feats 1|2` runs the
feature ablation.

```bash
python train_EPInformer.py \
    --model_type EPInformer-v2 --cell K562 --expr_type RNA --n_enh_feats 3 \
    --h5_path   batch_output/K562/encoding/K562_samples.h5 \
    --expr_csv  data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
    --split_csv data/leave_chrom_out_crossvalidation_split_18377genes.csv \
    --use_pretrained_encoder --pretrained_encoder_dir results/seqencoder/K562/checkpoints \
    --use_prm_signal --gene_list batch_output/K562/links/GeneList.txt \
    --output_dir ./EPInformer_models/K562 --epochs 50
# one fold only: add --fold 1 ;  specific folds: --folds 1 2 3
# pure feature ablation: drop --use_prm_signal and set --n_enh_feats 1|2|3
```

Per cell/fold it writes to `EPInformer_models/{cell}/`: best checkpoint,
`fold_{i}_..._predictions.csv`, `fold_summary.csv`, and `{model.name}_results.csv`
(all folds pooled). Defaults: `out_dim=64, useBN=False, n_enhancer=60, SmoothL1 + AdamW,
lr=1e-4, batch_size=50, early-stop patience 8`.

HPC (all 12 folds, full config — prm is the slurm default): `CELL=K562 sbatch
slurm/train_epinformer_12fold.slurm` (GM12878: `CELL=GM12878 ...`). Feature-ablation runs pass
`USE_PRM_SIGNAL=0 N_ENH_FEATS=1|2 CELL=K562 sbatch ...`; encoder fine-tuning adds `NO_FREEZE=1`.

---

## 6. Evaluate & visualize

```bash
# pooled out-of-fold Pearson R / R^2 / Spearman (+ scatter PNG)
python evaluate.py expression --pred_dir ./EPInformer_models/K562
python evaluate.py encoder    --pred_dir ./results/seqencoder/K562
```

Each evaluation command writes a pooled summary and scatter plot to the prediction directory.

---

## End-to-end (K562, from repo root)

```bash
conda activate epinformer_repro
bash scripts/download_abc_reference.sh data/reference/hg38          # + supply hg38.fa, expr CSV, split CSV
python scripts/download_encode_data.py --cell-types K562 --output-dir data
python run_pipeline.py --config config/config.yaml --samples K562 --stages links
python train_seqEncoder.py --cell K562 \
    --data-csv batch_output/K562/links/K562_peak_5bins_around_summit_activity_sequence.csv \
    --loss l1kl --batch-size 256 --output-dir ./results/seqencoder/K562 --epochs 50
python run_pipeline.py --config config/config.yaml --samples K562 --stages encoding
python train_EPInformer.py --model_type EPInformer-v2 --cell K562 --n_enh_feats 3 \
    --h5_path batch_output/K562/encoding/K562_samples.h5 \
    --expr_csv data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
    --split_csv data/leave_chrom_out_crossvalidation_split_18377genes.csv \
    --use_pretrained_encoder --pretrained_encoder_dir results/seqencoder/K562/checkpoints \
    --use_prm_signal --gene_list batch_output/K562/links/GeneList.txt \
    --output_dir ./EPInformer_models/K562 --epochs 50
python evaluate.py expression --pred_dir ./EPInformer_models/K562
```

**For GM12878**, replace `K562` throughout, **plus** first fetch the 4 filtered encoder reps
(Step 1: `--from-manifest config/gm12878_encoder_bams.json` + `samtools index`). The
`accessibility_bams`/`h3k27ac_bams` columns for GM12878 are already in `config/samples.tsv`, so
`--stages links` mean-pools the 2 reps/assay and the encoder reaches **0.6146 fwd+RC / 0.617**
(single-rep gets only ~0.572). The headline pooled-OOF numbers come from the fwd+RC
re-eval (`slurm/eval_seqencoder_fwdrc.slurm --skip-train --rc-average`).

---

## Optional quick start (skip the heavy BAM→ABC step)

Prebuilt data is on Zenodo. Record **13233337** has prebuilt ABC links
(`{cell}_ABC_EGLinks.zip`) and a legacy training HDF5; record **13232430** has the
expression CSV + CV split. With the prebuilt ABC links you can jump straight to
**Step 4** (`run_pipeline.py --stages encoding`) using the downloaded
`EnhancerPredictionsAllPutative.txt` / `EnhancerList.txt`, then **Step 5**. Note the
encoder-pretrain CSV (Step 2/3) still requires the BAMs, so for a fully prebuilt run
use a released pretrained encoder or train `EPInformer_v2` from scratch (drop
`--use_pretrained_encoder`).

---

## Notes / gotchas

- **Sequence length is 2000 bp** for promoter/enhancer inputs to `EPInformer_v2`
  (256 bp is only the encoder-pretraining bin size). The HDF5 `max_seq_len` is 2000.
- The trimmed `train_EPInformer.py` keeps a few inert CLI flags inherited from the
  original multi-variant trainer (e.g. `--rc_aware`, `--intra_attn`, `--legnet_ckpt_dir`);
  they are not used by the `EPInformer-v2` path — use the flags shown above.
- Device is auto-detected (`cuda > mps > cpu`); force with `--device`.
- On the HPC, submit to `gpu33` first (already set in the `slurm/` templates). The conda
  env there is **`EPInformer_env`** (torch 2.10+cu128); pass `CONDA_ENV=EPInformer_env` to the
  slurm jobs (the `epinformer_repro` default in `environment.yml` is the local name).
- **Encoder recipe knobs**: bin geometry is env-overridable in `encoder_pretrain_data.py`
  (`ENCODER_OFFSETS`, `ENCODER_OVERLAP`; default 5-bins/stride-192); `train_seqEncoder.py`
  adds `--loss {mse,l1kl}`, `--test-strand {auto,forward,reverse}`, and an eval-only mode
  (`--skip-train --checkpoint-dir <dir>`) used by `slurm/eval_seqencoder_fwdrc.slurm`.
- **Sequence+activity extraction**: `scripts/seq_activity_extract.py`
  (+`slurm/seq_activity_extract.slurm`) does the exact sequence+activity extraction
  (5-bin regions, `pysam.count`, `Activity=sqrt(DHS·H3K27ac)`, no read-extension, no negatives).
  It **pools multiple replicate BAMs per assay** — `--dnase-bam A.bam B.bam --h3k27ac-bam C.bam
  D.bam --pool-method mean` (default `mean` = per-rep-RPM average). GM12878's 0.617 requires the
  4 filtered reps (H3K27ac `ENCFF269GKF`+`ENCFF201OHW`, DNase `ENCFF729UYK`+`ENCFF020WZB`); the
  config-driven path does the same pooling via the `accessibility_bams`/`h3k27ac_bams` columns
  in `config/samples.tsv`.
- **How the exact BAMs were found**: `scripts/identify_encoder_reps.py` RPM-matches the reference
  CSV's per-replicate columns against candidate ENCODE BAMs (corr 1.0000 pins each rep) — this
  identified the 4 *filtered* bio-reps (not unfiltered) and that the gap was a missing 2nd
  DNase rep.
- **Promoter-activity feature**: `train_EPInformer.py --use_prm_signal --gene_list <ABC GeneList.txt>`
  adds `log(1+sqrt(DHS·H3K27ac at TSS1kb))` to the promoter slot (slurm knob `USE_PRM_SIGNAL=1`).
