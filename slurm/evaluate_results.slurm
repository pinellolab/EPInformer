#!/bin/bash
# Evaluate a completed encoder or expression array after all 12 folds finish.
# Required exports: MODE=encoder|expression, PRED_DIR. Optional: OUT_DIR, LABEL.
# Submit from the repository root or export REPO_ROOT explicitly.
#
#SBATCH --job-name=epi_eval
#SBATCH --output=log_cpu/%x_%j.out
#SBATCH --error=log_cpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p log_cpu

MODE="${MODE:?set MODE=encoder|expression}"
PRED_DIR="${PRED_DIR:?set PRED_DIR}"
OUT_DIR="${OUT_DIR:-${PRED_DIR}/evaluation}"
LABEL="${LABEL:-$(basename "$PRED_DIR")}"

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-epinformer_repro}"; set -u
export PYTHONUNBUFFERED=1

exec "${CONDA_PREFIX}/bin/python" evaluate.py "$MODE" \
  --pred_dir "$PRED_DIR" --out "$OUT_DIR" --label "$LABEL" --no-plot
