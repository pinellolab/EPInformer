# EPInformer Batch Pipeline Configuration

## Files

- **config.yaml** -- Main configuration (reference files, algorithm parameters, output settings)
- **parity_k562_training.yaml** -- Optional preset: Xpresso `expression_csv` + `abc_params.include_self_promoter: true` for fairer comparison with [`train_EPInformer_abc.py`](../train_EPInformer_abc.py) defaults and [`run_k562_preprocessing.py`](../run_k562_preprocessing.py)-style self-promoter handling (see [docs/why_new_hdf5_underperforms.md](../docs/why_new_hdf5_underperforms.md))
- **samples.tsv** -- Sample table with one row per cell type

## Sample Table Columns

| Column | Required | Description |
|--------|----------|-------------|
| `cell_type` | yes | Sample identifier (e.g., K562) |
| `accessibility_bam` | yes | DNase-seq or ATAC-seq BAM file |
| `assay` | yes | `dnase` or `atac` |
| `h3k27ac_bam` | no | H3K27ac ChIP BAM (activity from accessibility if omitted) |
| `hic_file` | no | .hic file (power-law fallback if omitted) |
| `expression_column` | no | Column in expression CSV (e.g., RPKM_E123) |
| `qnorm_ref` | no | Quantile normalization reference file |
| `peaks_file` | no | Pre-called narrowPeak file (skips MACS2) |
| `preset` | no | Cell-type preset (K562, GM12878, H1, HUVEC, NHEK) |
| `skip_links` | no | `true` to skip element-gene link stage |
| `skip_encoding` | no | `true` to skip HDF5 encoding stage |

Empty cells in optional columns are treated as unset (uses defaults).

## Usage

```bash
# Run full pipeline for all samples
python run_pipeline.py --config config/config.yaml

# Training-parity-oriented preset (Xpresso CSV + self-promoter in Stage 1)
python run_pipeline.py --config config/parity_k562_training.yaml

# Element-gene links only, single sample
python run_pipeline.py --config config/config.yaml --stages links --samples K562

# HDF5 encoding only (links already exist)
python run_pipeline.py --config config/config.yaml --stages encoding

# Validate config without running
python run_pipeline.py --config config/config.yaml --dry-run
```
