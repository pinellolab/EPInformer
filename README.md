# EPInformer — reproduction (enhancer activity + gene expression from DNA sequence)

A clean, self-contained reproduction of **EPInformer**, built around the model in
[`EPInformer/models.py`](EPInformer/models.py) (`EPInformer_v2` + the 256 bp
`enhancer_predictor_256bp`). It trains **two models, in order**, end-to-end from raw ENCODE data
across **6 cell lines** (K562, GM12878, H1, HepG2, HUVEC, NHEK):

1. **Enhancer-activity encoder** — predicts 256 bp enhancer activity (H3K27ac·DNase) from sequence.
2. **Gene-expression model** (`EPInformer_v2`) — predicts RNA / CAGE from a gene's promoter plus
   its ABC-nominated enhancers, reusing the pretrained encoder (frozen) as the sequence backbone.

Do **Part 1 (encoder) first**, then **Part 2 (expression)** — the expression model loads the
encoder.

## Results (12-fold leave-chromosome-out, pooled out-of-fold Pearson R)

**Part 1 — enhancer encoder** (log2 activity), matches the BSCC reference within ±0.003:

| | H1 | HepG2 | K562 | HUVEC | NHEK | GM12878 |
|---|---|---|---|---|---|---|
| **ours** | 0.820 | 0.743 | 0.740 | 0.742 | 0.677 | 0.617 |

**Part 2 — gene expression**, shipped **`f3`** config (frozen encoder + 3 enhancer features + promoter signal):

| | K562 | GM12878 | HepG2 | HUVEC | NHEK | H1 |
|---|---|---|---|---|---|---|
| **RNA** | 0.856 | 0.860 | 0.845 | 0.839 | 0.828 | 0.781 |
| **CAGE** | 0.867 | 0.890 | — | — | — | — |

CAGE labels exist only for K562/GM12878, so the other four are **RNA-only**. The
H1/HepG2/HUVEC/NHEK expression numbers are **new** (upstream EPInformer reports expression for
K562/GM12878 only).

<p align="center">
  <img height="330" src="images/rna_expression_scatter.png">
</p>

> **What `f3` means** (used throughout): the shipped expression model = `--n_enh_feats 3`
> (distance + activity + Hi-C contact) **+ `--use_prm_signal`** (promoter activity), with the
> pretrained encoder **frozen**. In the feature ablation, activity dominates, Hi-C is ~inert, and
> the promoter signal adds up to +0.03 (cell-dependent) — so it is bundled into `f3` by default.

---

## Datasets

All inputs are public: chromatin & Hi-C from **ENCODE**, expression labels from Xpresso/FANTOM
(Zenodo), genome **hg38**.

### ENCODE accessions (per cell)

DNase (accessibility) + H3K27ac (activity; 2 filtered bio-reps where available) + H3K27ac
**narrowPeak** (the encoder's summit source) + cell-specific **Hi-C**:

| Cell (Roadmap) | DNase | H3K27ac rep(s) | narrowPeak | Hi-C (`.hic`) |
|---|---|---|---|---|
| K562 (E123) | ENCFF257HEE | ENCFF232RQF | ENCFF544LXB | ENCFF621AIY |
| GM12878 (E116) | ENCFF729UYK, ENCFF020WZB | ENCFF269GKF, ENCFF201OHW | ENCFF023LTU | ENCFF318GOM |
| H1 (E003) | ENCFF761ZRE | ENCFF860ABR, ENCFF693IFG | ENCFF689CJG | — *(none in ENCODE — 5C/ChIA-PET only)* |
| HepG2 (E118) | ENCFF691HJY | ENCFF862NDZ, ENCFF926NHE, ENCFF745JCH | ENCFF392KDI\* | ENCFF805ALH |
| HUVEC (E122) | ENCFF091KTX | ENCFF374DGO, ENCFF609TUB | ENCFF077LGZ | ENCFF091YKP |
| NHEK (E127) | ENCFF117RNM | ENCFF770JWP, ENCFF051NTC | ENCFF666UYC | ENCFF776JNR |

- **K562** ships **single-rep** — its H3K27ac reps differ hugely in depth, so pooling dilutes.
- **GM12878** uses **both filtered reps per assay** — this is what reproduces 0.617.
- **HepG2** has 3 filtered H3K27ac reps (mean-pooled); its narrowPeak (`*`) is inferred, not cited.
- Wired in [`config/samples.tsv`](config/samples.tsv), `config/encoder_narrowpeaks.json`,
  `config/bscc_extra_cells_bams.json`.

### Other inputs

- **Genome:** supply `hg38.fa` → `data/reference/hg38/hg38.fa` (not downloaded by any script).
- **ABC reference** (gene bounds, chrom sizes, K562 quantile-norm): `bash scripts/download_abc_reference.sh data/reference/hg38`.
- **Expression labels + Xpresso features + 12-fold CV split** — Zenodo **13232430**
  (`expression_data.zip`) → unzip into `data/`:
  - `GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv` — `Actual_{cell}` RNA
    for all 6 cells + `{cell}_CAGE_128*3_sum` for K562/GM12878.
  - `leave_chrom_out_crossvalidation_split_18377genes.csv` — the fold assignment.
- **ABC average Hi-C** (hg38, optional — for from-raw contact on cells without cell-specific
  `.hic`): ENCODE **`ENCFF134PUN`** (annotation `ENCSR382HAW`, 5 kb). Split into a per-chromosome
  directory with `scripts/split_avg_hic.py`, then point `hic_file` at it.

## Setup

```bash
conda env create -f environment.yml && conda activate epinformer_repro
# key deps: torch, h5py, pyfaidx, kipoiseq, pyranges, macs2, hicstraw, pyBigWig, scipy, scikit-learn, pandas
bash scripts/download_abc_reference.sh data/reference/hg38          # ABC reference
# then: place hg38.fa at data/reference/hg38/hg38.fa, and unzip Zenodo expression_data.zip into data/
```

---

# Part 1 — Enhancer-activity encoder (do this first)

### 1a. Data preprocessing — download ENCODE → ABC → 256 bp activity CSV

```bash
# DNase/H3K27ac BAMs + Hi-C (K562/GM12878)
python scripts/download_encode_data.py --cell-types K562,GM12878

# GM12878 needs 2 filtered reps per assay (reproduces 0.617):
python scripts/download_encode_data.py --from-manifest config/gm12878_bscc_encoder_bams.json
for f in data/GM12878/{DNase,H3K27ac}/ENCFF*.bam; do samtools index "$f"; done

# H3K27ac narrowPeaks = the encoder's summit source:
python scripts/download_encode_data.py --from-manifest config/encoder_narrowpeaks.json
gunzip -f reference/*_H3K27ac.*.narrowPeak.gz

# ABC nomination + encoder CSV
python run_pipeline.py --config config/config.yaml --samples K562,GM12878 --stages links
#   -> batch_output/{cell}/links/{cell}_peak_5bins_around_summit_activity_sequence.csv   (encoder data)
#   -> batch_output/{cell}/links/{EnhancerList,GeneList}.txt + Predictions/...            (for Part 2)
```

### 1b. Train the encoder (12-fold, 5-bin / L1KL recipe)

```bash
python train_seqEncoder.py --cell K562 \
    --data-csv batch_output/K562/links/K562_peak_5bins_around_summit_activity_sequence.csv \
    --loss l1kl --batch-size 256 --output-dir results/seqencoder/K562 --epochs 50
# HPC, all 12 folds:   CELL=K562 sbatch slurm/train_seqencoder_12fold.slurm
```

### 1c. Evaluate (target ~0.71–0.74 log2-activity Pearson)

```bash
python evaluate.py encoder --pred_dir results/seqencoder/K562
```

---

# Part 2 — Gene-expression model (do this second)

Reuses the Part 1 encoder (frozen) + the ABC enhancer–gene links from step 1a.

### 2a. Data preprocessing — ABC links → factored gene HDF5

```bash
python run_pipeline.py --config config/config.yaml --samples K562,GM12878 --stages encoding
#   -> batch_output/{cell}/encoding/{cell}_samples.h5
#      (promoter + enhancer one-hot sequence, and activity / dhs / distance / contact features)
```

### 2b. Train `EPInformer_v2` — the shipped `f3` config

```bash
python train_EPInformer.py --model_type EPInformer-v2 --cell K562 --expr_type RNA \
    --n_enh_feats 3 --use_prm_signal \
    --h5_path   batch_output/K562/encoding/K562_samples.h5 \
    --expr_csv  data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
    --split_csv data/leave_chrom_out_crossvalidation_split_18377genes.csv \
    --use_pretrained_encoder --pretrained_encoder_dir results/seqencoder/K562/checkpoints \
    --gene_list batch_output/K562/links/GeneList.txt \
    --output_dir EPInformer_models/K562 --epochs 50
# HPC, all 12 folds (f3 is the slurm default):   CELL=K562 sbatch slurm/train_epinformer_12fold.slurm
#   CAGE instead of RNA:   EXPR_TYPE=CAGE ...
#   feature ablation:      USE_PRM_SIGNAL=0 N_ENH_FEATS=1|2|3 ...   (f1=dist, f2=+activity, f3=+Hi-C)
```

`--n_enh_feats 3 --use_prm_signal` **is `f3`** (the shipped config). The encoder is frozen by
default; `--no_freeze_encoder` (slurm `NO_FREEZE=1`) fine-tunes it end-to-end (optional, doesn't
reliably help).

### 2c. Evaluate (target RNA ~0.86 / CAGE ~0.88)

```bash
python evaluate.py expression --pred_dir EPInformer_models/K562
# visual report: open visualize_results.ipynb  (density scatter + per-fold R + cross-cell table)
```

---

## Other cell lines (H1 / HepG2 / HUVEC / NHEK) — RNA only

No CAGE labels exist for these. Their encoders train on BSCC's pre-built activity CSVs, and the
gene HDF5 is built from BSCC's precomputed ABC links (`scripts/build_gene_h5_for_cell.py` — no
raw BAMs needed; contact = ABC average Hi-C). Per cell (e.g. HepG2):

```bash
CELL=HepG2 sbatch slurm/train_seqencoder_12fold.slurm      # encoder -> results/seqencoder/HepG2_bsccData
CELL=HepG2 sbatch slurm/build_gene_h5.slurm                # gene H5 -> batch_output/HepG2/encoding/HepG2_samples.h5
CELL=HepG2 EXPR_TYPE=RNA USE_PRM_SIGNAL=1 \
  PRETRAINED_DIR=results/seqencoder/HepG2_bsccData/checkpoints \
  OUTPUT_DIR=EPInformer_models/HepG2_repro_RNA sbatch slurm/train_epinformer_12fold.slurm
python evaluate.py expression --pred_dir EPInformer_models/HepG2_repro_RNA
```

---

## Repository layout

```
EPInformer/models.py       EPInformer_v2 + 256 bp encoder  (the model)
preprocessing/             ABC nomination + HDF5 encoding
  abc/                       candidates -> neighborhoods -> contact -> predictions
run_pipeline.py            Stage 1 (links) + Stage 2 (encoding) orchestrator
train_seqEncoder.py        Part 1: enhancer-activity encoder
train_EPInformer.py        Part 2: EPInformer_v2 expression model
evaluate.py                pooled out-of-fold Pearson (encoder + expression)
scripts/                   download_encode_data, download_abc_reference,
                           build_gene_h5_for_cell, split_avg_hic (+ test_avg_hic), ...
config/                    config.yaml, samples.tsv, *_bams.json, encoder_narrowpeaks.json
slurm/                     12-fold array jobs (encoder / expression / build-H5) — gpu33 first
REPRODUCTION_NOTES.md      detailed findings, recipe provenance, per-cell BAM/Hi-C notes
```

## Notes

- **CV:** 12-fold **leave-chromosome-out**; report the **pooled** out-of-fold Pearson (concatenate
  all 12 folds → one R), not the per-fold mean.
- **Encoder recipe** (reproduces BSCC): 5 bins at summit ±192·{−2..2} (256 bp windows),
  Activity = `sqrt(H3K27ac_RPM · DNase_RPM)`, target `log2(0.1 + Activity)`, **L1KL** loss, batch 256.
- **Expression:** `EPInformer_v2` (SmoothL1 + AdamW, lr 1e-4, batch 50, 60 enhancers, frozen encoder).
- On our HPC the conda env is `EPInformer_env` (torch 2.10); override with `CONDA_ENV=` in slurm.
- Full provenance, ablations, and gotchas: **[`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md)**.
