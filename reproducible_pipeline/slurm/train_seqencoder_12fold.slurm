#!/bin/bash
# Pre-train the 256bp enhancer-activity sequence encoder, 12-fold leave-chromosome-out.
# One fold per array task.  Input CSV comes from Stage 1 (run_pipeline --stages links) OR,
# for GM12878, from the 2-rep mean-pool extractor (slurm/bscc_extract.slurm POOL_METHOD=mean;
# the single-rep path only reaches ~0.572, the 2-rep recipe reaches 0.617). The headline
# POOLED-OOF numbers (GM12878 0.617, K562 0.740) come from the fwd+RC re-eval afterwards
# (slurm/eval_seqencoder_fwdrc.slurm --skip-train --rc-average); this job's default test is
# single-reverse-strand.
#
# Usage (from repo root on HPC):
#   mkdir -p log_gpu
#   CELL=K562    sbatch slurm/train_seqencoder_12fold.slurm
#   CELL=GM12878 sbatch slurm/train_seqencoder_12fold.slurm
#
# Env overrides:
#   CELL (K562)  EPOCHS (50)  DATA_CSV (auto: batch_output/{CELL}/links/{CELL}_peak_5bins...csv)
#   OUTPUT_DIR (./results/seqencoder/{CELL})  CONDA_ENV (epinformer_repro)
#
#SBATCH --job-name=epi_repro_seqenc
#SBATCH --array=1-12
#SBATCH --output=log_gpu/%x_%A_%a.out
#SBATCH --error=log_gpu/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu33
#SBATCH --mem=32G
#SBATCH --time=6:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/reproducible_pipeline}"
cd "$REPO_ROOT"
mkdir -p log_gpu

FOLD="${SLURM_ARRAY_TASK_ID:?submit with sbatch}"
CELL="${CELL:-K562}"
EPOCHS="${EPOCHS:-50}"
DATA_CSV="${DATA_CSV:-./batch_output/${CELL}/links/${CELL}_peak_5bins_around_summit_activity_sequence.csv}"
LOSS="${LOSS:-l1kl}"
BATCH="${BATCH:-256}"
TEST_STRAND="${TEST_STRAND:-reverse}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/seqencoder/${CELL}}"

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-epinformer_repro}"; set -u
py="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) fold=${FOLD} CELL=${CELL} DATA_CSV=${DATA_CSV}"
echo "torch: $(${py} -c 'import torch;print(torch.__version__,"CUDA",torch.cuda.is_available())')"
[[ -f "$DATA_CSV" ]] || { echo "ERROR: missing ${DATA_CSV} (run Stage 1 first)"; exit 1; }

# Original BSCC_GPU 0.71 recipe: l1kl loss, batch 256, reverse-strand all-bins test
# (no --rc-average -> test_strand=reverse; no --test-summit-only -> pool over all 5 bins).
exec "$py" train_seqEncoder.py \
  --cell "$CELL" \
  --data-csv "$DATA_CSV" \
  --folds "$FOLD" \
  --loss "$LOSS" --batch-size "$BATCH" --test-strand "$TEST_STRAND" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS"
