#!/bin/bash
# Eval-only: re-evaluate already-trained encoder checkpoints under the fwd+RC test
# protocol (forward + reverse-complement averaged), so different recipes are compared
# on ONE consistent test-time strand. No training — loads fold_*_best checkpoints.
#
# Usage (all 12 folds in one job):
#   CELL=K562 \
#   DATA_CSV=./batch_output/K562/links/K562_peak_5bins_around_summit_activity_sequence.csv \
#   CKPT_DIR=./results/seqencoder/K562_5bins/checkpoints \
#   OUTPUT_DIR=./results/seqencoder/K562_5bins_fwdRC \
#   sbatch --export=ALL,CONDA_ENV=EPInformer_env,CELL=K562,DATA_CSV=...,CKPT_DIR=...,OUTPUT_DIR=... \
#          slurm/eval_seqencoder_fwdrc.slurm
#
#SBATCH --job-name=epi_seqenc_fwdrc
#SBATCH --output=log_gpu/%x_%j.out
#SBATCH --error=log_gpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu33
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/pipeline}"
cd "$REPO_ROOT"
mkdir -p log_gpu

CELL="${CELL:-K562}"
DATA_CSV="${DATA_CSV:?set DATA_CSV to the recipe activity CSV}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the trained checkpoints dir}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a fresh dir for fwd+RC predictions}"

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-EPInformer_env}"; set -u
py="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) CELL=${CELL} DATA_CSV=${DATA_CSV}"
echo "CKPT_DIR=${CKPT_DIR}  OUTPUT_DIR=${OUTPUT_DIR}  (fwd+RC)"
[[ -f "$DATA_CSV" ]] || { echo "ERROR: missing ${DATA_CSV}"; exit 1; }
[[ -d "$CKPT_DIR" ]] || { echo "ERROR: missing ${CKPT_DIR}"; exit 1; }

# --skip-train + --rc-average => load best checkpoints, test with forward+RC averaging (all 12 folds).
exec "$py" train_seqEncoder.py \
  --cell "$CELL" \
  --data-csv "$DATA_CSV" \
  --skip-train --rc-average \
  --checkpoint-dir "$CKPT_DIR" \
  --output-dir "$OUTPUT_DIR"
