#!/bin/bash
# Execute the Hugging Face checkpoint inference notebook end-to-end.
#
# Submit from the repository root:
#   sbatch slurm/test_hf_checkpoint_notebook.slurm
#
# Overrides: CONDA_ENV, HF_CHECKPOINT_ROOT, OUTPUT_DIR, NOTEBOOK

#SBATCH --job-name=epi_hf_notebook
#SBATCH --output=log_gpu/%x_%j.out
#SBATCH --error=log_gpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu33
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOTEBOOK="${NOTEBOOK:-predict_enhancer_and_expression.ipynb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/hpc_test_outputs/hf_checkpoint_notebook}"
HF_CHECKPOINT_ROOT="${HF_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/EPInformer-hf}"

cd "$REPO_ROOT"
mkdir -p log_gpu "$OUTPUT_DIR"

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV:-EPInformer_env}"
set -u

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export EPINFORMER_HF_CHECKPOINT_ROOT="$HF_CHECKPOINT_ROOT"

[[ -f "$NOTEBOOK" ]] || { echo "ERROR: missing notebook $NOTEBOOK"; exit 1; }
[[ -f "$HF_CHECKPOINT_ROOT/enhancer_encoders/K562/fold_1.pt" ]] || {
  echo "ERROR: missing Hugging Face checkpoint tree at $HF_CHECKPOINT_ROOT"
  exit 1
}

# Install a job-local kernelspec so nbconvert necessarily launches this conda env.
JUPYTER_PREFIX="$OUTPUT_DIR/jupyter"
"${CONDA_PREFIX}/bin/python" -m ipykernel install \
  --prefix "$JUPYTER_PREFIX" --name epinformer_hf --display-name "EPInformer HF test"
export JUPYTER_PATH="$JUPYTER_PREFIX/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}"

echo "host=$(hostname)"
echo "python=${CONDA_PREFIX}/bin/python"
echo "checkpoint_root=$EPINFORMER_HF_CHECKPOINT_ROOT"
"${CONDA_PREFIX}/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.cuda.is_available())'

"${CONDA_PREFIX}/bin/python" -m jupyter nbconvert \
  --to notebook --execute "$NOTEBOOK" \
  --ExecutePreprocessor.kernel_name=epinformer_hf \
  --ExecutePreprocessor.timeout=14400 \
  --output-dir "$OUTPUT_DIR" \
  --output predict_enhancer_and_expression.hf_executed.ipynb

echo "Executed notebook: $OUTPUT_DIR/predict_enhancer_and_expression.hf_executed.ipynb"
