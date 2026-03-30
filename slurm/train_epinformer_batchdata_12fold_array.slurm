#!/bin/bash
# 12-fold leave-chrom-out CV on pipeline-encoded factored HDF5 (run_pipeline Stage 2).
# One array task per fold; each task uses a distinct --output_dir to avoid overwriting
# fold_summary.csv and *results.csv (see train_EPInformer_abc.py).
#
# Prerequisites:
#   ./batch_output/<CELL>/encoding/<CELL>_samples.h5
#
# Dated Slurm logs (stdout/stderr under log_gpu/YYYY-MM-DD/) — use the wrapper:
#   ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#
# Examples (from repo root):
#   K562 — logs in log_gpu/$(date +%Y-%m-%d)/
#     CELL=K562 ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#
#   GM12878
#     CELL=GM12878 ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#
#   Fixed date folder:
#     LOG_DATE=2026-03-30 CELL=K562 ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#
# EPInformer-v2 (use a distinct OUTPUT_DIR_BASE so checkpoints do not mix with EPInformer-abc):
#   K562:
#     CELL=K562 MODEL_TYPE=EPInformer-v2 OUTPUT_DIR_BASE=./EPInformer_models/batch_K562_cv12_v2/ \
#       ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#   GM12878:
#     CELL=GM12878 MODEL_TYPE=EPInformer-v2 OUTPUT_DIR_BASE=./EPInformer_models/batch_GM12878_cv12_v2/ \
#       ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#
# Plain sbatch (no date subfolder; writes log_gpu/epinformer_batch_cv12_*_*.out):
#   CELL=K562 sbatch slurm/train_epinformer_batchdata_12fold_array.slurm
#
# Overrides (export before sbatch):
#   CELL=K562|GM12878          (default: K562)
#   MODEL_TYPE=...           (default: EPInformer-abc; use EPInformer-v2 for v2 architecture)
#   H5_PATH=...              (default: ./batch_output/${CELL}/encoding/${CELL}_samples.h5)
#   EXPR_CSV=...             (default: ./data/GM12878_K562_18377_gene_expr_fromXpresso.csv)
#   SPLIT_CSV=...            (default: ./data/leave_chrom_out_crossvalidation_split_18377genes.csv)
#   OUTPUT_DIR_BASE=...      (default: ./EPInformer_models/batch_${CELL}_cv12/ — set explicitly when using MODEL_TYPE=EPInformer-v2)
#   EPOCHS BATCH_SIZE CUDA_ID EARLY_STOP_PATIENCE
#
# Limit concurrent GPUs: edit #SBATCH --array=1-12%4 below (%4 = max 4 tasks at once).
#
#SBATCH --job-name=epinformer_batch_cv12
#SBATCH --array=1-12%4
#SBATCH --output=log_gpu/%x_%A_%a.out
#SBATCH --error=log_gpu/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu33
#SBATCH --time=24:00:00
#SBATCH --mem=256G

set -eo pipefail
cd /lustre/grp/zyjlab/linjc/epinformer
mkdir -p "log_gpu/${EPINFORMER_LOG_DATE:-$(date +%Y-%m-%d)}"
mkdir -p log_gpu

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

FOLD="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset — submit with sbatch array (not bash)}"

CELL="${CELL:-K562}"
H5_PATH="${H5_PATH:-./batch_output/${CELL}/encoding/${CELL}_samples.h5}"
EXPR_CSV="${EXPR_CSV:-./data/GM12878_K562_18377_gene_expr_fromXpresso.csv}"
SPLIT_CSV="${SPLIT_CSV:-./data/leave_chrom_out_crossvalidation_split_18377genes.csv}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-./EPInformer_models/batch_${CELL}_cv12/}"
OUTPUT_DIR="${OUTPUT_DIR_BASE%/}/fold_${FOLD}/"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CUDA_ID="${CUDA_ID:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
MODEL_TYPE="${MODEL_TYPE:-EPInformer-abc}"

echo "=== EPInformer batch pipeline 12-fold CV (Slurm array) ==="
echo "host=$(hostname)  date=$(date -Is)  job=${SLURM_JOB_ID:-?}  array_job=${SLURM_ARRAY_JOB_ID:-?}  task=${SLURM_ARRAY_TASK_ID}  cwd=$(pwd)"
echo "CELL=${CELL}  MODEL_TYPE=${MODEL_TYPE}  FOLD=${FOLD}  OUTPUT_DIR=${OUTPUT_DIR}"

MINICONDA_ROOT="${MINICONDA_ROOT:-/lustre/grp/zyjlab/linjc/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"

if [[ ! -f "${MINICONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found under MINICONDA_ROOT=${MINICONDA_ROOT}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"

PYTHON_EXE=""
_TRAIN_IMPORTS='import numpy, pandas, scipy, tqdm, torch, h5py, sklearn'
for env in EPInformer_env dna_composer epinformer_env; do
  conda activate "$env" 2>/dev/null || continue
  py="${CONDA_PREFIX}/bin/python"
  if [[ -x "$py" ]] && "$py" -c "${_TRAIN_IMPORTS}" 2>/dev/null; then
    PYTHON_EXE="$py"
    echo "Using conda env: ${env} (${py})" >&2
    break
  fi
done
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "ERROR: No usable conda env (EPInformer_env, dna_composer, epinformer_env)." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

if [[ ! -f "$H5_PATH" ]]; then
  echo "ERROR: HDF5 not found: ${H5_PATH}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

exec "${PYTHON_EXE}" -u train_EPInformer_abc.py \
  --h5_path "$H5_PATH" \
  --expr_csv "$EXPR_CSV" \
  --split_csv "$SPLIT_CSV" \
  --model_type "$MODEL_TYPE" \
  --cell "$CELL" \
  --expr_type RNA \
  --n_enh_feats 3 \
  --epochs "$EPOCHS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --fold "$FOLD" \
  --batch_size "$BATCH_SIZE" \
  --cuda_id "$CUDA_ID" \
  --output_dir "$OUTPUT_DIR"
