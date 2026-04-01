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

# --- Download ENCODE BAM/HiC files ---
# Dry run: show what would be downloaded for all 11 cell lines
python scripts/download_encode_data.py --dry-run
# Download specific cell types
python scripts/download_encode_data.py --cell-types HepG2,H1,NHEK
# All replicates (one BAM per bio rep, marks *best*)
python scripts/download_encode_data.py --all-replicates --dry-run
# Roadmap 57 epigenomes (report only)
python scripts/download_encode_data.py \
    --roadmap data/roadmap_expression/.cache/EG.name.txt \
    --dry-run --report data/roadmap_encode_report.html
# Save manifest (query API once, reuse later without API)
python scripts/download_encode_data.py --dry-run --save-manifest data/encode_manifest
# Download from saved manifest (no API queries)
python scripts/download_encode_data.py --from-manifest data/encode_manifest.json
# Build Roadmap per-replicate manifest (38 epigenomes × DNase/H3K27ac/HiC)
python scripts/build_roadmap_manifest.py
# Download from Roadmap manifest
python scripts/download_encode_data.py --from-manifest data/roadmap_download_manifest.json

# --- ABC Pipeline (from BAM/HiC → enhancer-gene predictions) ---
# Run full ABC pipeline for a cell type (with Hi-C)
python -c "
from preprocessing.abc import run_abc_pipeline
run_abc_pipeline(
    accessibility_bam='./data/K562/DNase/ENCFF257HEE.bam',
    h3k27ac_bam='./data/K562/H3K27ac/ENCFF232RQF.bam',
    hic_file='./data/K562/HiC/ENCFF621AIY.hic',
    output_dir='./abc_output/K562',
    cell_type='K562',
    preset='K562',
    max_distance=1_000_000,
    include_promoter_region=False,
    max_encoder_peaks=100000,
)
"

# --- Batch pipeline (config-driven, recommended) ---
# Runs Stage 1 (ABC links) + Stage 2 (HDF5 encoding) from YAML config
python run_pipeline.py --config config/config.yaml --samples K562 --stages both
# Dry run to validate config without running
python run_pipeline.py --config config/config.yaml --dry-run

# Re-run only the encoder data step (Step 4) for a cell type
python scripts/rerun_encoder_step.py --cell K562

# --- Build gene annotation BED from Roadmap Ensembl v65 (hg19 → hg38 liftover) ---
python preprocessing/data_prep/build_gene_annotation.py \
    --gene-set pc --output-dir data/reference/hg38        # protein-coding (~20K)
python preprocessing/data_prep/build_gene_annotation.py \
    --gene-set pc_linc --output-dir data/reference/hg38   # + lincRNA (~25K)

# --- Build Roadmap expression for all 57 epigenomes ---
# Pure Roadmap (no Xpresso features — rna_feats auto-disabled at training time)
python preprocessing/data_prep/build_roadmap_expression.py \
    --gene-set pc --output-dir data/roadmap_expression

# With Xpresso features (18,377 genes, for compatibility with existing models)
python preprocessing/data_prep/build_roadmap_expression.py \
    --xpresso-csv data/GM12878_K562_18377_gene_expr_fromXpresso.csv \
    --output-dir data/roadmap_expression

# --- Preprocessing (ABC outputs → HDF5 for training) ---
# With Xpresso expression (18,377 genes, recommended for apple-to-apple comparison)
python run_preprocessing.py --no-bigwig \
    --cell-type K562 \
    --gene-expr-csv data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv \
    --predictions ./abc_output/K562/Predictions/EnhancerPredictionsAllPutative.txt \
    --enhancer-list ./abc_output/K562/EnhancerList.txt \
    --output-dir ./training_data/K562_run \
    --include-self-promoter \
    --fasta /path/to/hg38.fa

# With Roadmap expression (pure Roadmap, no Xpresso features)
python run_preprocessing.py --no-bigwig \
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
    --h5_path ./training_data/K562_run/K562_samples.h5 \
    --epochs 2 --output_dir ./EPInformer_models/K562_test

# Train with self-promoter removed at training time
python -u train_EPInformer_abc.py \
    --model_type EPInformer-abc --n_enh_feats 3 \
    --h5_path ./training_data/K562_run/K562_samples.h5 \
    --rm_self_promoter --epochs 2 --output_dir ./EPInformer_models_noselfprm/
```

No formal test suite exists. Validation is done through the Jupyter notebooks and benchmark datasets (CRISPR, eQTL).

## Architecture

### Pipeline Flow

```
Raw Data (FASTA, BigWig signals, CSV expression, ABC links)
  → Preprocessing (preprocessing/)
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

**`preprocessing/`** — Data preprocessing module:
- `extract.py`: Extracts promoter-enhancer sequences and signals from FASTA/BigWig files
- `hdf5.py`: HDF5 I/O. Structure: `seq_code` (N, 1+n_enh, L, 4), `activity`, `dhs`, `distance`, `contact`, `seq_signal`
- `links.py`: Encodes enhancer-gene links with ABC scores
- `pipelines_legacy.py`: Legacy cell-type-specific preprocessing pipelines. Gene matching uses `_map_symbol_to_ensid()` which auto-detects whether `TargetGene` contains ENSIDs or gene symbols — if ENSIDs (>50% match), uses them directly; otherwise maps symbols → ENSID via expression CSV `Gene name` column. Enhancers are sorted by absolute distance to TSS (nearest first) and capped at 60 per gene. No minimum distance filter by default (all enhancers including ≤1kb are kept).
- `abc/`: Streamlined ABC pipeline (candidates → neighborhoods → contact → predictions → encoder data). Entry point: `run_abc_pipeline()` in `__init__.py`. Supports cell-type presets (K562, GM12878, etc.) and `max_encoder_peaks` filtering. MACS2 settings match original Broad ABC pipeline (`-p 0.1`, `--nomodel`, `--shift -75`, `--extsize 150`). Presets use `XpressoGeneBounds.hg38.bed` (18,377 genes). `TargetGene` outputs ENSID with `TargetGeneSymbol` for gene symbols. Default `max_distance=1MB`. Hi-C processing matches the original Broad ABC pipeline: SCALE normalization via `hicstraw.HiCFile`, doubly-stochastic correction, diagonal bin correction (`tss_hic_contribution=100`), power-law formula `exp(scale - gamma*log(dist+1))` with 5kb min clip, power-law scaling (`hic * reference/observed`), pseudocount (`min(powerlaw, powerlaw_at_5kb)`), and QC (replace genes with insufficient Hi-C coverage). Self-promoter ABC scores are NOT overridden to 1.0.
- `abc/encoder_pretrain_data.py`: Generates 256bp sequences + activity labels for pre-training the `enhancer_predictor_256bp` encoder. Extracts 5 bins per summit at offsets [-2,-1,0,1,2] × 156bp stride. Activity is computed **per 256bp bin** from BAM read counts (DNase RPM, or √(H3K27ac·DNase) when both BAMs provided), not over the full MACS peak interval. Optional negative samples from random genomic positions ≥1kb from any peak.

**`run_pipeline.py`** — Config-driven batch orchestrator. Reads a YAML config (`config/config.yaml`) and TSV sample table (`config/samples.tsv`), runs Stage 1 (ABC links via `run_abc_pipeline()`) and Stage 2 (HDF5 encoding via `obtain_PE_withSignals()`). Supports `--stages links|encoding|both`, `--samples` filter, `--dry-run`. Hi-C parameters (`hic_gamma`, `hic_scale`, etc.) are read from `abc_params` in the YAML config.

**`src/scripts/`** — Training utilities:
- `utils.py`: Data processing and normalization
- `utils_forTraining.py`: Custom datasets, metrics (MSE, Pearson correlation), device management

**`run_preprocessing.py`** — General preprocessing CLI for any cell type. Wraps `obtain_PE_withSignals()` to produce factored HDF5 from ABC outputs + expression CSV. Accepts `--cell-type`, `--gene-expr-csv`, `--predictions`, `--enhancer-list`.

**`run_k562_preprocessing.py`** — K562-specific CLI with subcommands: `with-signals`, `h3k27ac`, `pe`
- Key flags: `--min-distance` (default 0), `--max-distance` (default 100000), `--n-enhancer` (default 60), `--include-self-promoter`, `--abc-all-putative`
- `--include-self-promoter` injects near-TSS self-promoter elements (from ABC `isSelfPromoter` flag) at slot 0 of each gene's enhancer list with real ABC features (activity, contact, DHS, distance). This is critical for matching legacy performance (~0.81 vs ~0.63 Pearson R without).

**`scripts/rerun_encoder_step.py`** — Re-runs only Step 4 (encoder data generation) of the ABC pipeline for a given cell type, reading config/samples from the same YAML/TSV used by `run_pipeline.py`. Usage: `python scripts/rerun_encoder_step.py --cell K562`.

**`scripts/download_encode_data.py`** — Queries ENCODE REST API to find and download unfiltered-alignment BAM files (DNase, H3K27ac, ATAC) and Hi-C `.hic` files. Reads cell types from `data/cell_line_list.txt` (11 cell lines) by default; `--roadmap` mode supports all 57 Roadmap epigenomes via `data/roadmap_expression/.cache/EG.name.txt`. Key features: biosample ontology name mapping (e.g., NHEK→`keratinocyte`, HUVEC→`endothelial cell of umbilical vein`), 4DN Hi-C fallback, `--all-replicates` mode (one BAM per bio rep with `*best*` marker), ENCODE phase detection (ENCODE2/3/4 via `award.rfa`), `--report` HTML generation, per-file metadata JSON. Uses `frame=embedded` ENCODE search to get award info in a single query. BAM files must be **unfiltered alignments** (not filtered "alignments") because the ABC pipeline uses `MACS2 --keep-dup all`. Supports `--save-manifest PATH` to export TSV+JSON manifests and `--from-manifest PATH.json` to download from a saved manifest without re-querying the API. Pre-built manifests: `data/encode_manifest.{tsv,json}` (11 cell lines), `data/roadmap_encode_manifest.{tsv,json}` (57 epigenomes).

**`scripts/build_roadmap_manifest.py`** — Builds a per-replicate download manifest for 38 verified Roadmap epigenomes (exact/close ENCODE matches) across DNase, H3K27ac, and Hi-C. Selects the best experiment per assay (newest pipeline, most replicates), includes all biological replicates, marks the best (largest file). Outputs `data/roadmap_download_manifest.{tsv,json,html}`. The mapping was verified against Roadmap metadata (`Roadmap.metadata.qc.jul2013.xlsx`) — E024 was corrected (was wrongly mapped to CD4+ Treg instead of ESC), E016/HUES64 confirmed no DNase. See `docs/encode_data_guide.md` for the full search→match→review→download pipeline.

**`preprocessing/data_prep/build_gene_annotation.py`** — Builds hg38 gene annotation BED from Roadmap's Ensembl v65 gene_info by lifting over hg19 coordinates. Supports `--gene-set pc` (protein-coding, ~20K) or `--gene-set pc_linc` (+ lincRNA, ~25K). Requires `liftOver` binary on PATH.

**`preprocessing/data_prep/build_roadmap_expression.py`** — Downloads Roadmap Epigenomics RNA-seq RPKM for all 57 epigenomes and builds expression CSVs. Applies `log10(RPKM + 0.1) → z-score per cell type`. Supports `--gene-set pc` (default, protein-coding) or `--gene-set pc_linc` (+ lincRNA, downloads `57epigenomes.RPKM.nc.gz`). When `--xpresso-csv` is omitted, outputs pure Roadmap expression without Xpresso gene-structural features — the training script auto-detects this and disables `rna_feats`. Key Roadmap ID mappings: E123=K562, E116=GM12878, E118=HepG2, E003=H1, E122=HUVEC, E127=NHEK.

**`train_EPInformer_abc.py`** — Training script with two dataset classes:
- `promoter_enhancer_dataset`: For new factored HDF5 (`gene_enh_idx` referencing shared `enhancer_seq` pool). Supports `--rm_self_promoter` to filter out self-promoter elements (distance < 1kb) at training time.
- `promoter_enhancer_dataset_legacy`: For legacy HDF5 with pre-computed `promoter_ohe` and `pe_ohe` arrays. Loads promoter from HDF5 with zero-padding (fast path).
- Auto-detects Xpresso feature columns in expression CSV; if absent, sets `useFeat=False` (model skips `rna_feats` branch). Works with both Xpresso-merged and pure Roadmap CSVs.

### Gene ID Matching

The new ABC pipeline outputs **ENSID** (e.g., `ENSG00000000003`) in `TargetGene` and gene symbols in `TargetGeneSymbol`. Legacy ABC outputs use gene symbols in `TargetGene`. The encoding pipeline (`_map_symbol_to_ensid()`) auto-detects the format: if `TargetGene` values match known ENSIDs, they are used directly; otherwise symbols are mapped via the expression CSV `Gene name` column. The merge with expression data always joins on `ENSID`.

### Enhancer Slot Ordering

Enhancers are sorted by **absolute distance** to TSS (nearest first) and capped at 60 per gene. Slot 0 is reserved for the self-promoter element (if `--include-self-promoter`), which is the gene's own promoter region (`isSelfPromoter=True`, typically `class=promoter`, distance 0–2kb). Remaining slots fill with the next closest enhancers. With `--rm_self_promoter` at training time, elements with `abs(distance) < 1kb` are skipped and remaining enhancers shift up.

### Data

- **Legacy BSCC_GPU EPInformer (Pinello-style scripts + 200-CRE HDF5):** canonical checkout at `/lustre/grp/zyjlab/linjc/BSCC_GPU/BSCC_GPU/EPInformer` — expects `./data/{CELL}_200CREs-gene_RPM_4feats.hdf5`. How that file is produced, how it maps to factored `samples.h5`, and a benchmark parity checklist are documented in [docs/legacy_epinformer_comparison.md](docs/legacy_epinformer_comparison.md). For quick NaN/range checks on disk, run `python scripts/compare_h5_legacy_factored_stats.py --legacy-h5 ... --factored-h5 ...`.
- **Why new `run_pipeline` HDF5 can score lower than older files:** expression CSV choice (Roadmap vs Xpresso), ABC/peaks provenance, Hi-C NaNs, and Stage 1 vs Stage 2 self-promoter flags — see [docs/why_new_hdf5_underperforms.md](docs/why_new_hdf5_underperforms.md). Optional parity config: `config/parity_k562_training.yaml`.
- Related dataset directory: `../data_EPInformer/` (hg38.fa, HDF5 files, ABC links, pre-trained models)
- In-repo `data/`: gene expression CSVs, ABC enhancer-gene links, cross-validation splits, CRISPR/eQTL benchmarks
- `data/roadmap_expression/`: Roadmap-derived expression for 57 epigenomes (generated by `preprocessing/data_prep/build_roadmap_expression.py`)
- `abc_output/`: ABC pipeline outputs per cell type (predictions, enhancer lists, encoder data)
- Pre-trained models in `trained_models/` (163 enhancer encoder checkpoints, expression model checkpoints as `.pt` files)

### Notebooks

- `predict_gene_expression.ipynb`: Load trained models, predict expression, evaluate on benchmarks
- `predict_enhancer_activity.ipynb`: Predict enhancer activity, TF motif discovery via saturation mutagenesis (Tangermeme)
