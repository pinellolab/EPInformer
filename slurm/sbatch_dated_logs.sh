#!/usr/bin/env bash
# Submit Slurm with stdout/stderr under log_gpu/YYYY-MM-DD/ so all jobs submitted
# the same calendar day (same DATE) share one folder.
#
# Usage (from repo root):
#   ./slurm/sbatch_dated_logs.sh slurm/train_epinformer_test_fold1.slurm
#   ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_test_folds.slurm
#   ./slurm/sbatch_dated_logs.sh --array slurm/train_epinformer_batchdata_12fold_array.slurm
#     (12-fold CV on batch_output/*/encoding/*_samples.h5; set CELL=K562 or CELL=GM12878;
#      MODEL_TYPE=EPInformer-v2 OUTPUT_DIR_BASE=./EPInformer_models/batch_${CELL}_cv12_v2/ for v2)
# Override folder: LOG_DATE=2026-03-28 ./slurm/sbatch_dated_logs.sh ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARRAY=0
if [[ "${1:-}" == --array ]]; then
  ARRAY=1
  shift
fi

DATE="${LOG_DATE:-$(date +%Y-%m-%d)}"
mkdir -p "log_gpu/$DATE"

# Propagated into the job so mkdir below matches this folder (same YYYY-MM-DD for all tasks that day).
export EPINFORMER_LOG_DATE="$DATE"

if [[ "$ARRAY" -eq 1 ]]; then
  OUTPAT='%x_%A_%a'
else
  OUTPAT='%x_%j'
fi

exec sbatch \
  --output="log_gpu/$DATE/${OUTPAT}.out" \
  --error="log_gpu/$DATE/${OUTPAT}.err" \
  "$@"
