#!/bin/bash
# Stage 1 (ABC enhancer-gene nomination) + Stage 2 (HDF5 encoding), config-driven, CPU-only.
#
# Usage (from repo root on HPC):
#   mkdir -p log_cpu
#   sbatch slurm/run_pipeline_cpu.slurm                    # both cells, both stages
#   SAMPLES=K562 STAGES=links sbatch slurm/run_pipeline_cpu.slurm
#   PIPELINE_DRY_RUN=1 sbatch slurm/run_pipeline_cpu.slurm
#
# Env overrides: CONFIG (config/config.yaml), STAGES (both|links|encoding),
#   SAMPLES (comma-separated cells; unset = all), CONDA_ENV (epinformer_repro).
# Keep abc_params.n_threads in config.yaml in sync with --cpus-per-task below.
#
#SBATCH --job-name=epi_repro_pipeline
#SBATCH --output=log_cpu/%x_%j.out
#SBATCH --error=log_cpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -p cpu1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/reproducible_pipeline}"
cd "$REPO_ROOT"
mkdir -p log_cpu

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-epinformer_repro}"; set -u
py="${CONDA_PREFIX}/bin/python"

CONFIG="${CONFIG:-config/config.yaml}"
STAGES="${STAGES:-both}"
echo "host=$(hostname) date=$(date -Is) CONFIG=${CONFIG} STAGES=${STAGES} SAMPLES=${SAMPLES:-<all>}"

ARGS=(-u run_pipeline.py --config "$CONFIG" --stages "$STAGES")
[[ -n "${SAMPLES:-}" ]] && ARGS+=(--samples "$SAMPLES")
[[ "${PIPELINE_DRY_RUN:-}" == "1" ]] && ARGS+=(--dry-run)
exec "$py" "${ARGS[@]}"
