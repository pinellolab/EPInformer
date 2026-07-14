#!/bin/bash
# Regenerate ONLY the encoder-pretrain data (ABC Step 4) from the ENCODE H3K27ac
# narrowPeak (encoder_peaks_file in samples.tsv), 5 windows / 192bp stride (64bp
# overlap) — matches the original ~0.70 pooled-OOF recipe.
# Avoids a full Stage-1 redo.
#   CELL=K562 sbatch --export=ALL,CONDA_ENV=EPInformer_env,CELL=K562 slurm/regen_encoder.slurm
#
#SBATCH --job-name=epi_regen_enc
#SBATCH --output=log_cpu/%x_%j.out
#SBATCH --error=log_cpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -p cpu1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/pipeline}"
cd "$REPO_ROOT"
mkdir -p log_cpu
CELL="${CELL:-K562}"
MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-EPInformer_env}"; set -u
export PYTHONUNBUFFERED=1
echo "host=$(hostname) CELL=${CELL} date=$(date -Is)"
exec "${CONDA_PREFIX}/bin/python" scripts/regen_encoder_data.py --cell "$CELL" --n-threads "${SLURM_CPUS_PER_TASK:-12}"
