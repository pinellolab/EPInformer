# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPInformer is a deep learning framework for predicting gene expression by integrating promoter-enhancer sequences with epigenomic signals. It models enhancer-promoter interactions using convolutional sequence encoders and multi-head attention layers. From the Pinello Lab.

Key applications:
- Gene expression prediction from promoter-enhancer sequences + epigenomic features
- Cell-type-specific enhancer-gene interaction identification and in-silico perturbation
- Enhancer activity prediction and TF binding motif discovery

## Environment Setup

### EPInformer_env (recommended, matches `dna_composer` PyTorch stack)

Use the helper script so torch/torchvision/torchaudio and CUDA wheels match the `dna_composer` conda env (torch 2.10 + cu128-style wheels, plus pytorch-lightning / torchmetrics / triton as in that env):

```bash
# Fast path: clone dna_composer, then install EPInformer-only pip deps
bash scripts/setup_EPInformer_env.sh --clone-dna-composer

# Or: fresh Python 3.10 + pinned torch stack from scripts/requirements_torch_dna_composer.txt
bash scripts/setup_EPInformer_env.sh
```

If a broken partial env is left on disk, remove it first: `conda env remove -n EPInformer_env -y` and delete `…/miniconda3/envs/EPInformer_env` if it remains.

### Legacy (manual conda + pytorch 12.1)

```bash
conda create --name EPInformer_env python=3.8 pandas scipy scikit-learn jupyter seaborn
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyranges pyfaidx kipoiseq openpyxl tangermeme h5py pyBigWig
```

## Common Commands

```bash
# Download training data from Zenodo
sh ./download_data.sh

# --- ABC Pipeline (from BAM/HiC → enhancer-gene predictions) ---
# Run full ABC pipeline for a cell type
python -c "
from epinformer_preprocessing.abc import run_abc_pipeline
run_abc_pipeline(
    cell_type='K562',
    dnase_bam='./data/K562/DNase/ENCFF257HEE.bam',
    h3k27ac_bam='./data/K562/H3K27ac/ENCFF232RQF.bam',
    hic_file='./data/K562/HiC/ENCFF621AIY.hic',
    fasta='./data_EPInformer/hg38.fa',
    output_dir='./abc_output/K562',
    max_encoder_peaks=100000,
)
"

# --- Build Roadmap expression for all 57 epigenomes ---
python scripts/build_roadmap_expression.py \
    --xpresso-csv data/GM12878_K562_18377_gene_expr_fromXpresso.csv \
    --output-dir data/roadmap_expression

# --- Preprocessing (ABC outputs → HDF5 for training) ---
# Generic cell type (uses run_preprocessing.py)
python run_preprocessing.py with-signals --no-bigwig \
    --cell-type K562 \
    --gene-expr-csv data/roadmap_expression/roadmap_expression_all.csv \
    --predictions ./abc_output/K562/Predictions/EnhancerPredictionsAllPutative.txt \
    --enhancer-list ./abc_output/K562/EnhancerList.txt \
    --output-dir ./training_data/K562_run \
    --include-self-promoter

# K562-specific legacy preprocessing
python run_k562_preprocessing.py with-signals --no-bigwig \
    --output-dir ./training_data/k562_run \
    --include-self-promoter

# With BigWig signals
python run_k562_preprocessing.py with-signals \
    --output-dir ./training_data/k562_run \
    --signal-bigwigs dnase.bigWig h3k27ac.bigWig h3k4me1.bigWig h3k4me3.bigWig ctcf.bigWig \
    --include-self-promoter

# --- Train model ---
python -u train_EPInformer_abc.py \
    --model_type EPInformer-abc --n_enh_feats 3 \
    --h5_path ./training_data/k562_run/samples.h5 \
    --epochs 2 --output_dir ./EPInformer_models/

# Train with self-promoter removed at training time
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
- `pipelines_legacy.py`: Legacy cell-type-specific preprocessing pipelines. Gene matching uses `_map_symbol_to_ensid()` to convert ABC gene symbols (TargetGene) → ENSID before merging with expression CSV.
- `abc/`: Streamlined ABC pipeline (candidates → neighborhoods → contact → predictions → encoder data). Entry point: `run_abc_pipeline()` in `__init__.py`. Supports cell-type presets (K562, GM12878, etc.) and `max_encoder_peaks` filtering.

**`src/scripts/`** — Training utilities:
- `utils.py`: Data processing and normalization
- `utils_forTraining.py`: Custom datasets, metrics (MSE, Pearson correlation), device management

**`run_preprocessing.py`** — General preprocessing CLI for any cell type. Wraps `obtain_PE_withSignals()` to produce factored HDF5 from ABC outputs + expression CSV. Accepts `--cell-type`, `--gene-expr-csv`, `--predictions`, `--enhancer-list`.

**`run_k562_preprocessing.py`** — K562-specific CLI with subcommands: `with-signals`, `h3k27ac`, `pe`
- Key flags: `--min-distance` (default 0), `--max-distance` (default 100000), `--n-enhancer` (default 60), `--include-self-promoter`, `--abc-all-putative`
- `--include-self-promoter` injects near-TSS self-promoter elements (from ABC `isSelfPromoter` flag) at slot 0 of each gene's enhancer list with real ABC features (activity, contact, DHS, distance). This is critical for matching legacy performance (~0.81 vs ~0.63 Pearson R without).

**`scripts/build_roadmap_expression.py`** — Downloads Roadmap Epigenomics RNA-seq RPKM for all 57 epigenomes and builds expression CSVs. Applies `log10(RPKM + 0.1) → z-score per cell type` (Xpresso convention). Outputs `roadmap_expression_all.csv` with columns: `gene_id`, `ENSID`, `Gene name`, `Actual_{cell_type}`, plus Xpresso sequence features. Key Roadmap ID mappings: E123=K562, E116=GM12878, E118=HepG2, E003=H1, E122=HUVEC, E127=NHEK.

**`train_EPInformer_abc.py`** — Training script with two dataset classes:
- `promoter_enhancer_dataset`: For new factored HDF5 (`gene_enh_idx` referencing shared `enhancer_seq` pool). Supports `--rm_self_promoter` to filter out self-promoter elements (distance < 1kb) at training time.
- `promoter_enhancer_dataset_legacy`: For legacy HDF5 with pre-computed `promoter_ohe` and `pe_ohe` arrays. Loads promoter from HDF5 with zero-padding (fast path).

### Gene ID Matching

ABC pipeline outputs use **gene symbols** (e.g., `BET1L`, `K562`) in `TargetGene`, while the expression CSV uses **ENSID** (e.g., `ENSG00000000003`). The pipeline maps symbols → ENSID via `_map_symbol_to_ensid()` using the `Gene name` column in the expression CSV, then merges on `ENSID`. Both `ENSID` and `Gene name` are kept in the enhancer-gene table through to HDF5.

### Data

- Related dataset directory: `../data_EPInformer/` (hg38.fa, HDF5 files, ABC links, pre-trained models)
- In-repo `data/`: gene expression CSVs, ABC enhancer-gene links, cross-validation splits, CRISPR/eQTL benchmarks
- `data/roadmap_expression/`: Roadmap-derived expression for 57 epigenomes (generated by `scripts/build_roadmap_expression.py`)
- `abc_output/`: ABC pipeline outputs per cell type (predictions, enhancer lists, encoder data)
- Pre-trained models in `trained_models/` (163 enhancer encoder checkpoints, expression model checkpoints as `.pt` files)

### Notebooks

- `predict_gene_expression.ipynb`: Load trained models, predict expression, evaluate on benchmarks
- `predict_enhancer_activity.ipynb`: Predict enhancer activity, TF motif discovery via saturation mutagenesis (Tangermeme)
