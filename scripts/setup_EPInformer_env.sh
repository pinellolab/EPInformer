#!/usr/bin/env bash
# Create conda env EPInformer_env with the same PyTorch stack as dna_composer.
#
# Recommended (exact clone of dna_composer, then EPInformer-only pip deps):
#   bash scripts/setup_EPInformer_env.sh --clone-dna-composer
#
# Alternative (fresh python 3.10 + pip install torch 2.10 cu128 wheels + extras):
#   bash scripts/setup_EPInformer_env.sh
#
# If a broken half-installed env is left on disk, remove it first:
#   conda env remove -n EPInformer_env -y
#   rm -rf "${MINICONDA_ROOT}/envs/EPInformer_env"
#
set -euo pipefail

MINICONDA_ROOT="${MINICONDA_ROOT:-/lustre/grp/zyjlab/linjc/miniconda3}"
CONDA="${MINICONDA_ROOT}/bin/conda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$CONDA" ]]; then
  echo "conda not found at $CONDA (set MINICONDA_ROOT)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"

CLONE_MODE=0
if [[ "${1:-}" == "--clone-dna-composer" ]]; then
  CLONE_MODE=1
fi

if conda env list | grep -qE '^EPInformer_env[[:space:]]'; then
  echo "EPInformer_env already exists; installing/upgrading pip packages only."
  conda activate EPInformer_env
elif [[ "$CLONE_MODE" -eq 1 ]]; then
  echo "Cloning conda env dna_composer -> EPInformer_env (same PyTorch/CUDA pip stack)..."
  conda create -n EPInformer_env --clone dna_composer -y
  conda activate EPInformer_env
else
  echo "Creating conda env EPInformer_env (python 3.10)..."
  conda create -n EPInformer_env python=3.10 pip -y
  conda activate EPInformer_env
  python -m pip install --upgrade pip
  python -m pip install -r "${SCRIPT_DIR}/requirements_torch_dna_composer.txt"
fi

python -m pip install -r "${SCRIPT_DIR}/requirements_epinformer_extras.txt"

echo ""
echo "Verifying torch (CUDA may be false on CPU-only login nodes):"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Activate with: conda activate EPInformer_env"
