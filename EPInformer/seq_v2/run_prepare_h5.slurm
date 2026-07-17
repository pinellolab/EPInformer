#!/bin/bash
#SBATCH --job-name=episeq_h5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
set -euo pipefail

: "${PEAKS:?set PEAKS to a Roadmap DNase narrowPeak file}"
: "${DNASE_BW:?set DNASE_BW to a DNase cut-site BigWig}"
: "${CHROM_SIZES:?set CHROM_SIZES to hg38.chrom.sizes}"
: "${CELL:?set CELL}"
: "${OUT_H5:?set OUT_H5}"
H3_ARGS=()
if [[ -n "${H3K27AC_BW:-}" ]]; then H3_ARGS+=(--h3k27ac-bigwig "$H3K27AC_BW"); fi
python -m EPInformer.seq_v2.build_profile_h5 \
  --peaks "$PEAKS" --bigwig "$DNASE_BW" --chrom-sizes "$CHROM_SIZES" \
  --cell "$CELL" --out-h5 "$OUT_H5" "${H3_ARGS[@]}"
