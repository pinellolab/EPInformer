# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPInformer is a deep learning framework for predicting gene expression by integrating promoter-enhancer sequences with epigenomic signals. It models enhancer-promoter interactions using convolutional sequence encoders and multi-head attention layers. From the Pinello Lab.

Key applications:
- Gene expression prediction from promoter-enhancer sequences + epigenomic features
- Cell-type-specific enhancer-gene interaction identification and in-silico perturbation
- Enhancer activity prediction and TF binding motif discovery

## Environment Setup

```bash
conda create --name EPInformer_env python=3.8 pandas scipy scikit-learn jupyter seaborn
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyranges pyfaidx kipoiseq openpyxl tangermeme h5py pyBigWig
```

## Common Commands

```bash
# Download training data from Zenodo
sh ./download_data.sh

# Preprocess K562 data with multiple signal tracks (no BigWig)
python run_k562_preprocessing.py with-signals --no-bigwig \
    --output-dir ./training_data/k562_run \
    --include-self-promoter

# Preprocess with BigWig signals
python run_k562_preprocessing.py with-signals \
    --output-dir ./training_data/k562_run \
    --signal-bigwigs dnase.bigWig h3k27ac.bigWig h3k4me1.bigWig h3k4me3.bigWig ctcf.bigWig \
    --include-self-promoter

# Preprocess with configurable distance range
python run_k562_preprocessing.py with-signals --no-bigwig \
    --min-distance 0 --max-distance 100000 --n-enhancer 60 \
    --include-self-promoter \
    --abc-all-putative ./data_EPInformer/.../EnhancerPredictionsAllPutative.avghic.txt

# Train model (new factored HDF5)
python -u train_EPInformer_abc.py \
    --model_type EPInformer-abc --n_enh_feats 3 \
    --h5_path ./training_data/k562_run/samples.h5 \
    --epochs 2 --output_dir ./EPInformer_models/

# Train with self-promoter removed at training time (same HDF5)
python -u train_EPInformer_abc.py \
    --model_type EPInformer-abc --n_enh_feats 3 \
    --h5_path ./training_data/k562_run/samples.h5 \
    --rm_self_promoter --epochs 2 --output_dir ./EPInformer_models_noselfprm/
```

No formal test suite exists. Validation is done through the Jupyter notebooks and benchmark datasets (CRISPR, eQTL).

## Architecture

### Pipeline Flow

```
Raw Data (FASTA, BigWig signals, CSV expression, ABC links)
  → Preprocessing (epinformer_preprocessing/)
  → HDF5 arrays (sequences, activity, distance, contact, signals)
  → Model Training (PyTorch, src/train_EPInformer_abc.py)
  → Prediction & Evaluation (notebooks)
```

### Key Modules

**`src/EPInformer/models_abc.py`** — Core model definitions:
- `seq_256bp_encoder`: Conv encoder for 256bp DNA sequences → 128-dim embeddings
- `MHAttention_encoderLayer`: Multi-head attention (d_model=128, nhead=8) for PE interactions
- `EPInformer_abc`: Main model integrating promoter/enhancer sequences with epigenomic features (activity, distance, Hi-C contact, DNase). Supports up to 50 enhancers per gene, 3 transformer layers.
- `EPInformer_abc_dist`, `EPInformer_abc_dist_v2`, `EPInformer_v2`: Variant architectures

**`epinformer_preprocessing/`** — Data preprocessing module:
- `extract.py`: Extracts promoter-enhancer sequences and signals from FASTA/BigWig files
- `hdf5.py`: HDF5 I/O. Structure: `seq_code` (N, 1+n_enh, L, 4), `activity`, `dhs`, `distance`, `contact`, `seq_signal`
- `links.py`: Encodes enhancer-gene links with ABC scores
- `pipelines_legacy.py`: Legacy cell-type-specific preprocessing pipelines (K562, GM12878, HepG2, etc.)

**`src/scripts/`** — Training utilities:
- `utils.py`: Data processing and normalization
- `utils_forTraining.py`: Custom datasets, metrics (MSE, Pearson correlation), device management

**`run_k562_preprocessing.py`** — CLI with subcommands: `with-signals`, `h3k27ac`, `pe`
- Key flags: `--min-distance` (default 0), `--max-distance` (default 100000), `--n-enhancer` (default 60), `--include-self-promoter`, `--abc-all-putative`
- `--include-self-promoter` injects near-TSS self-promoter elements (from ABC `isSelfPromoter` flag) at slot 0 of each gene's enhancer list with real ABC features (activity, contact, DHS, distance). This is critical for matching legacy performance (~0.81 vs ~0.63 Pearson R without).

**`train_EPInformer_abc.py`** — Training script with two dataset classes:
- `promoter_enhancer_dataset`: For new factored HDF5 (`gene_enh_idx` referencing shared `enhancer_seq` pool). Supports `--rm_self_promoter` to filter out self-promoter elements (distance < 1kb) at training time.
- `promoter_enhancer_dataset_legacy`: For legacy HDF5 with pre-computed `promoter_ohe` and `pe_ohe` arrays. Loads promoter from HDF5 with zero-padding (fast path).

### Data

- Related dataset directory: `../data_EPInformer/` (hg38.fa, HDF5 files, ABC links, pre-trained models)
- In-repo `data/`: gene expression CSVs, ABC enhancer-gene links, cross-validation splits, CRISPR/eQTL benchmarks
- Pre-trained models in `trained_models/` (163 enhancer encoder checkpoints, expression model checkpoints as `.pt` files)

### Notebooks

- `predict_gene_expression.ipynb`: Load trained models, predict expression, evaluate on benchmarks
- `predict_enhancer_activity.ipynb`: Predict enhancer activity, TF motif discovery via saturation mutagenesis (Tangermeme)
