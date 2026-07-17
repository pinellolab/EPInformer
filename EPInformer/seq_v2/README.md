# EPInformer-seq-v2 training pipeline

This directory contains the canonical training and data-processing workflow for
EPInformer-seq-v2, the post-publication extension designed for Chorus. The final
recipe is the Roadmap retrain used for the published per-cell checkpoints:

1. Call or obtain Roadmap DNase-summit peaks.
2. Convert DNase cut-site and optional H3K27ac BigWigs into `profile_data.h5` with
   [`build_profile_h5.py`](build_profile_h5.py).
3. Build a background `bias` group, train a frozen-bias model with `train_bias.py`,
   and then train a per-cell `PerCellProfileNetWide` model.
4. Evaluate held-out chromosomes and publish only `main.pt`, `bias.pt`, and JSON summaries.

The HDF5 schema is:

```text
peak/chrom, start, summit       coordinates
peak/profile                    (N, 2, 1024) int16 profiles
peak/counts                     (N, 2) int32 totals
bias/                           optional background group for bias training
```

Channel 0 is DNase 5′ cut-site signal and channel 1 is H3K27ac signal. The model
receives 2,114 bp and predicts the central 1,024 bp. Training uses fold-10
leave-chromosome-out splits, reverse-complement augmentation, multinomial profile
NLL plus log-count MSE, AdamW, and OneCycleLR.

Example data preparation:

```bash
python -m EPInformer.seq_v2.build_profile_h5 \
  --peaks data/roadmap/DNase-peaks.bed.gz \
  --bigwig data/K562_dnase_cutsites.bw \
  --h3k27ac-bigwig data/K562_h3k27ac.bw \
  --chrom-sizes data/reference/hg38.chrom.sizes \
  --cell K562 --out-h5 training/K562/profile_data.h5
```

Example training:

```bash
python -m EPInformer.seq_v2.train \
  --cell K562 --h5 training/K562/profile_data.h5 \
  --fasta data/reference/hg38/hg38.fa \
  --split-csv data/leave_chrom_out_crossvalidation_split_18377genes.csv \
  --bias-weights results/bias/K562/bias.pt \
  --out-dir results/epinformer_seq_v2/K562
```

Large FASTA, BAM/BigWig, HDF5, checkpoint, and prediction files belong on HPC
storage and are intentionally not included in Git.

`build_profile_h5.py` intentionally creates an empty bias group because bias
windows are a separate background sampling problem. Supply a populated bias HDF5
from the Roadmap background-generation step before running `train_bias.py`.
