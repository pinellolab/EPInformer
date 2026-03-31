# EPInformer Data Processing Guide

This guide walks through preparing input data, configuring the batch pipeline, and generating HDF5 training files for EPInformer.

## Pipeline Overview

The data processing pipeline has two stages:

1. **Element-gene links** (Stage 1) -- Runs the ABC pipeline on BAM files to produce enhancer-gene predictions and enhancer lists.
2. **HDF5 encoding** (Stage 2) -- Converts predictions + FASTA sequences into HDF5 arrays ready for model training.

Both stages are orchestrated by `run_pipeline.py`, which reads a YAML config file and a TSV sample table.

```
BAM files + reference genome
  → Stage 1: element-gene links (ABC pipeline)
  → Stage 2: HDF5 encoding (sequences + features)
  → train_EPInformer_abc.py
```

---

## Step 1: Set Up the Environment

```bash
bash scripts/setup_EPInformer_env.sh
```

Activate the environment:

```bash
conda activate EPInformer_env
```

Key dependencies (installed by the setup script):

- PyTorch, pybedtools, pysam, pyBigWig
- pandas, h5py, pyranges, pyfaidx, kipoiseq
- pyyaml (for config parsing)

If you see `ModuleNotFoundError: No module named 'pybedtools'`, install it via conda:

```bash
conda install -c bioconda pybedtools
```

---

## Step 2: Prepare Reference Data

### 2.1 Download reference files

The pipeline needs these reference files (hg38):

| File | Description | Default path |
|------|-------------|-------------|
| hg38 FASTA | Reference genome | `data/reference/hg38/hg38.fa` |
| Gene bounds BED | Collapsed gene boundaries | `data/reference/hg38/CollapsedGeneBounds.Ensembl_v65.hg38.pc.bed` |
| Chromosome sizes | Chromosome lengths | `data/reference/hg38/GRCh38_EBV.chrom.sizes.tsv` |
| Expression CSV | Gene expression values | `data/roadmap_expression/roadmap_expression_all.csv` |

To download ABC reference files (chrom sizes, quantile normalization reference):

```bash
bash preprocessing/data_prep/download_abc_reference.sh
```

### 2.2 Build gene annotation BED

Build a hg38 gene annotation BED from Roadmap's Ensembl v65 / Gencode v10 gene_info (lifts over hg19 coordinates to hg38). Requires the `liftOver` binary on PATH.

```bash
# Protein-coding genes only (~20K)
python preprocessing/data_prep/build_gene_annotation.py \
    --gene-set pc \
    --output-dir data/reference/hg38

# Protein-coding + lincRNA (~25K)
python preprocessing/data_prep/build_gene_annotation.py \
    --gene-set pc_linc \
    --output-dir data/reference/hg38
```

This produces BED files like `CollapsedGeneBounds.Ensembl_v65.hg38.pc.bed` matching the format used by the ABC pipeline.

### 2.3 Build expression CSV

The expression CSV contains normalized RNA-seq RPKM for 57 Roadmap epigenomes. Normalization: `log10(RPKM + 0.1) -> z-score` per cell type.

```bash
# Protein-coding only (~20K genes, no Xpresso features)
python preprocessing/data_prep/build_roadmap_expression.py \
    --gene-set pc \
    --output-dir data/roadmap_expression

# Protein-coding + lincRNA (~25K genes)
python preprocessing/data_prep/build_roadmap_expression.py \
    --gene-set pc_linc \
    --output-dir data/roadmap_expression

# With Xpresso features (18,377 genes — for compatibility with existing models)
python preprocessing/data_prep/build_roadmap_expression.py \
    --xpresso-csv data/GM12878_K562_18377_gene_expr_fromXpresso.csv \
    --output-dir data/roadmap_expression
```

When `--xpresso-csv` is omitted, the output CSV will not have Xpresso gene-structural features (UTR length, GC content, etc.). The training script auto-detects this and disables `rna_feats` accordingly.

This produces `data/roadmap_expression/roadmap_expression_all.csv` with expression columns for each cell type.

**Expression column names** (use these in the `expression_column` field of your sample table):

| Cell type | Roadmap CSV | Xpresso CSV |
|-----------|-------------|-------------|
| K562 | `K562_RPKM` | `Actual_K562` |
| GM12878 | `GM12878_RPKM` | `Actual_GM12878` |
| HepG2 | `HepG2_RPKM` | `Actual_HepG2` |
| H1 | `H1_RPKM` | `Actual_H1` |
| HUVEC | `HUVEC_RPKM` | `Actual_HUVEC` |
| NHEK | `NHEK_RPKM` | `Actual_NHEK` |

> **Important:** The column name must match the expression CSV you are using. Roadmap expression CSVs use `{cell}_RPKM`; Xpresso CSVs (`GM12878_K562_18377_gene_expr_fromXpresso.csv`) use `Actual_{cell}`.

---

## Step 3: Prepare Input Data

For each cell type you want to process, you need:

- **Accessibility BAM** (required): DNase-seq or ATAC-seq alignment file (`.bam`)
- **H3K27ac BAM** (optional): H3K27ac ChIP-seq BAM. If omitted, activity is estimated from the accessibility signal.
- **Hi-C file** (optional): `.hic` contact matrix. If omitted, a power-law distance fallback is used.

> **Note:** The pipeline automatically sorts and indexes BAM files if needed (via `samtools sort` and `samtools index`). You do not need to prepare them manually.

---

## Step 4: Configure the Pipeline

The pipeline is controlled by two files in the `config/` directory.

### 4.1 YAML config (`config/config.yaml`)

```yaml
# Path to sample table (relative to this config file's directory)
samples_table: samples.tsv

# Shared reference files (relative to project root)
reference:
  fasta: ./data_EPInformer/hg38.fa
  gene_bed: ./data/reference/hg38/CollapsedGeneBounds.Ensembl_v65.Gencode_v10.hg38.bed
  chrom_sizes: ./data/reference/hg38/GRCh38_EBV.chrom.sizes.tsv
  expression_csv: ./data/roadmap_expression/roadmap_expression_all.csv
  blacklist: null

# Stage 1: Element-gene link parameters
abc_params:
  n_top_peaks: 150000
  peak_extend: 250
  max_distance: 2500000
  gamma: 0.87
  tss_slop: 500
  hic_resolution: 5000
  neg_fraction: 0.05           # fraction of negative samples for encoder pre-training
  max_encoder_peaks: 100000   # top peaks (by signalValue) for encoder pre-training data
  include_self_promoter: false
  include_promoter_region: false
  n_threads: 4

# Stage 2: HDF5 encoding parameters
preprocessing_params:
  min_distance: 0
  max_distance: 100000
  n_enhancer: 60
  max_seq_len: 2000
  include_self_promoter: true
  tss_column: TSS_xpresso
  no_bigwig: true   # sequence-only mode; no BigWig signal tracks

# Output directory
output:
  base_dir: ./batch_output
```

**Path resolution rules:**
- `samples_table` is resolved relative to the **config file directory** (e.g., `config/`)
- All other paths (reference, output) are resolved relative to the **project root** (where you run the command)

### 4.2 Sample table (`config/samples.tsv`)

A tab-separated file with one row per cell type:

| Column | Required | Description |
|--------|----------|-------------|
| `cell_type` | yes | Sample identifier (e.g., `K562`) |
| `accessibility_bam` | yes | Path to DNase-seq or ATAC-seq BAM |
| `assay` | yes | `dnase` or `atac` |
| `h3k27ac_bam` | no | H3K27ac ChIP-seq BAM |
| `hic_file` | no | `.hic` contact matrix |
| `expression_column` | no | Column name in expression CSV (e.g., `K562_RPKM` for Roadmap, `Actual_K562` for Xpresso) |
| `qnorm_ref` | no | Quantile normalization reference |
| `peaks_file` | no | Pre-called narrowPeak file (skips MACS2 peak calling) |
| `preset` | no | Cell-type preset: `K562`, `GM12878`, `H1`, `HUVEC`, `NHEK` |
| `skip_links` | no | `true` to skip Stage 1 |
| `skip_encoding` | no | `true` to skip Stage 2 |

Example:

```tsv
cell_type	accessibility_bam	assay	h3k27ac_bam	hic_file	expression_column	preset
K562	./data/K562/DNase/ENCFF257HEE.bam	dnase	./data/K562/H3K27ac/ENCFF232RQF.bam	./data/K562/HiC/ENCFF621AIY.hic	Actual_K562	K562
GM12878	./data/GM12878/DNase/sample.bam	dnase	./data/GM12878/H3K27ac/sample.bam		Actual_GM12878	GM12878
```

Empty cells in optional columns are treated as unset.

---

## Step 5: Validate with Dry Run

Before running the full pipeline, validate your config:

```bash
python run_pipeline.py --config config/config.yaml --dry-run
```

This checks that:
- The YAML config and sample table parse correctly
- All required columns are present
- File paths resolve properly

It prints the parameters that would be passed to each stage without executing anything (no heavy dependencies needed).

---

## Step 6: Run the Pipeline

### Full pipeline (both stages, all samples)

```bash
python run_pipeline.py --config config/config.yaml
```

### Run a single sample

```bash
python run_pipeline.py --config config/config.yaml --samples K562
```

### Run multiple specific samples

```bash
python run_pipeline.py --config config/config.yaml --samples K562,GM12878
```

### Run only Stage 1 (element-gene links)

```bash
python run_pipeline.py --config config/config.yaml --stages links
```

### Run only Stage 2 (HDF5 encoding)

Use this when you already have link predictions and just want to regenerate the HDF5:

```bash
python run_pipeline.py --config config/config.yaml --stages encoding
```

### Stop on first failure

```bash
python run_pipeline.py --config config/config.yaml --fail-fast
```

By default, the pipeline continues processing remaining samples if one fails, printing a summary at the end.

---

## Step 7: Check Outputs

After the pipeline completes, outputs are organized per cell type:

```
batch_output/
  K562/
    links/                                          # Stage 1 outputs
      macs2/
        peaks_peaks.narrowPeak                      # MACS2 peak calls
      Neighborhoods/
        EnhancerList.txt                            # Candidate enhancer elements
      Predictions/
        EnhancerPredictionsAllPutative.txt          # All enhancer-gene predictions
      K562_peak_5bins_around_summit_activity_sequence.csv  # Encoder pre-training data
    encoding/                                       # Stage 2 outputs
      K562_samples.h5                               # HDF5 training file
  GM12878/
    links/
      ...
    encoding/
      GM12878_samples.h5
```

The HDF5 file contains arrays for each gene: one-hot encoded sequences, activity scores, distances, Hi-C contact, and DHS signals -- ready for model training.

---

## Step 8: Encoder Pre-training Data

Stage 1 also generates encoder pre-training data (Step 4 of the ABC pipeline). This produces 256bp sequences with per-bin activity labels for pre-training the `enhancer_predictor_256bp` sequence encoder.

For each of the top peaks (controlled by `max_encoder_peaks`), 5 bins of 256bp are extracted at offsets [-2, -1, 0, 1, 2] relative to the peak summit (stride = 156bp). Activity is computed **per 256bp bin** by counting BAM reads in the same window used for sequence extraction (DNase RPM, or sqrt(H3K27ac * DNase) when both BAMs are provided). Optional negative samples from random genomic positions >= 1kb from any peak are included (controlled by `neg_fraction`).

Output: `{output_dir}/{cell_type}/links/{cell_type}_peak_5bins_around_summit_activity_sequence.csv`

### Re-running encoder data only

To regenerate encoder pre-training data without re-running the full ABC pipeline, use the helper script:

```bash
python scripts/rerun_encoder_step.py --cell K562
```

This reads the same `config/config.yaml` and `config/samples.tsv` as `run_pipeline.py`, locates the existing narrowPeak file, and re-runs only the encoder data generation step.

---

## Step 9: Train the Model

Point the training script at the HDF5 file:

```bash
python -u train_EPInformer_abc.py \
    --model_type EPInformer-abc \
    --n_enh_feats 3 \
    --h5_path ./batch_output/K562/encoding/K562_samples.h5 \
    --epochs 20 \
    --output_dir ./EPInformer_models/
```

### Key training flags

| Flag | Description |
|------|-------------|
| `--h5_path` | Path to the HDF5 file from Stage 2 |
| `--model_type` | Model architecture (`EPInformer-abc`, `EPInformer-v2`, etc.) |
| `--n_enh_feats` | Number of enhancer features (default: 3) |
| `--rm_self_promoter` | Remove self-promoter elements at training time |
| `--epochs` | Number of training epochs |
| `--output_dir` | Directory for model checkpoints |

---

## Troubleshooting

### Missing pybedtools

```
ModuleNotFoundError: No module named 'pybedtools'
```

Solution: `conda install -c bioconda pybedtools`

### Expression column not found

```
KeyError: No expression column found
```

Check that your `expression_column` in the sample table matches a column in the expression CSV. Roadmap CSVs use `K562_RPKM`; Xpresso CSVs use `Actual_K562`. See the expression column table in Step 2.

### MACS2 peak calling fails or is slow

Provide a pre-called peaks file in the sample table (`peaks_file` column) to skip MACS2:

```tsv
peaks_file
./data/K562/peaks/ENCFF621SXE.narrowPeak
```

### Stage 2 fails with "predictions file not found"

Make sure Stage 1 completed successfully first, or that the predictions file exists at the expected path (`batch_output/{cell_type}/links/Predictions/EnhancerPredictionsAllPutative.txt`).
