#!/bin/bash
#SBATCH --job-name=episeq_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail

: "${CELL:?set CELL}"
: "${H5:?set H5 profile_data.h5}"
: "${FASTA:?set FASTA}"
: "${SPLIT_CSV:?set SPLIT_CSV}"
: "${BIAS_WEIGHTS:?set BIAS_WEIGHTS bias.pt}"
: "${OUT_DIR:?set OUT_DIR}"
python -m EPInformer.seq_v2.train --cell "$CELL" --h5 "$H5" --fasta "$FASTA" \
  --split-csv "$SPLIT_CSV" --bias-weights "$BIAS_WEIGHTS" --out-dir "$OUT_DIR"
