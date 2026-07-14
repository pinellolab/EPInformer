#!/bin/bash
# Train the full EPInformer_v2 expression model (models.py), 12-fold leave-chromosome-out.
# One fold per array task. Uses the pretrained encoder from train_seqencoder_12fold.slurm.
#
# Usage (from repo root on HPC), after Stage 1+2 and encoder pretraining:
#   mkdir -p log_gpu
#   # FULL config (default headline = 3 enh feats + promoter signal):
#   USE_PRM_SIGNAL=1 CELL=K562    sbatch slurm/train_epinformer_12fold.slurm
#   USE_PRM_SIGNAL=1 CELL=GM12878 sbatch slurm/train_epinformer_12fold.slurm
#   # Feature ablation (no promoter signal, to isolate enh-feature contribution):
#   N_ENH_FEATS=2 CELL=K562 sbatch slurm/train_epinformer_12fold.slurm   # activity-only, no Hi-C
#   N_ENH_FEATS=1 CELL=K562 sbatch slurm/train_epinformer_12fold.slurm   # distance-only
#
# Env overrides:
#   CELL (K562)  EPOCHS (50)  N_ENH_FEATS (3 = distance,activity,HiC; 2 = distance,activity)
#   EXPR_TYPE (RNA|CAGE)  BATCH_SIZE (50)  LR (1e-4)
#   H5_PATH   (./batch_output/{CELL}/encoding/{CELL}_samples.h5)
#   EXPR_CSV  (./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv)
#   SPLIT_CSV (./data/leave_chrom_out_crossvalidation_split_18377genes.csv)
#   PRETRAINED_DIR (./results/seqencoder/{CELL}/checkpoints)  set empty to train encoder from scratch
#   OUTPUT_DIR (./EPInformer_models/{CELL})  CONDA_ENV (epinformer_repro)
#
#SBATCH --job-name=epi_repro_train
#SBATCH --array=1-12
#SBATCH --output=log_gpu/%x_%A_%a.out
#SBATCH --error=log_gpu/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu33
#SBATCH --mem=96G
#SBATCH --time=6:00:00
# --mem=96G: the dataset preloads all pe_ohe (~18k genes x 61 x 2000 x 4 f32 ~= 36 GB) into RAM.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/pipeline}"
cd "$REPO_ROOT"
mkdir -p log_gpu

FOLD="${SLURM_ARRAY_TASK_ID:?submit with sbatch}"
CELL="${CELL:-K562}"
EPOCHS="${EPOCHS:-50}"
N_ENH_FEATS="${N_ENH_FEATS:-3}"
EXPR_TYPE="${EXPR_TYPE:-RNA}"
BATCH_SIZE="${BATCH_SIZE:-50}"
LR="${LR:-1e-4}"
H5_PATH="${H5_PATH:-./batch_output/${CELL}/encoding/${CELL}_samples.h5}"
EXPR_CSV="${EXPR_CSV:-./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv}"
SPLIT_CSV="${SPLIT_CSV:-./data/leave_chrom_out_crossvalidation_split_18377genes.csv}"
PRETRAINED_DIR="${PRETRAINED_DIR-./results/seqencoder/${CELL}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./EPInformer_models/${CELL}}"
USE_PRM_SIGNAL="${USE_PRM_SIGNAL:-1}"   # SHIPPED DEFAULT = 1 => add promoter activity (sqrt(DHS*H3K27ac at
                                        # TSS1kb)) as input. Justified: helps GM12878 (+0.026 RNA / +0.014 CAGE),
                                        # neutral for K562 (+0.000/-0.002). Set USE_PRM_SIGNAL=0 for feature-ablation
                                        # runs. Requires GeneList.txt from Stage 1 (hard-errors if missing).
GENE_LIST="${GENE_LIST:-./batch_output/${CELL}/links/GeneList.txt}"
NO_FREEZE="${NO_FREEZE:-0}"              # SHIPPED DEFAULT = 0 => freeze the pretrained seq encoder
                                        # (headline EPInformer freezes; matches canonical design).
                                        # 1 => fine-tune the encoder end-to-end (optional experiment only).

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-epinformer_repro}"; set -u
py="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) fold=${FOLD} CELL=${CELL} N_ENH_FEATS=${N_ENH_FEATS} EXPR_TYPE=${EXPR_TYPE}"
echo "H5_PATH=${H5_PATH}"
echo "torch: $(${py} -c 'import torch;print(torch.__version__,"CUDA",torch.cuda.is_available())')"
[[ -f "$H5_PATH"  ]] || { echo "ERROR: missing ${H5_PATH} (run Stage 2 encoding first)"; exit 1; }
[[ -f "$EXPR_CSV" ]] || { echo "ERROR: missing ${EXPR_CSV}"; exit 1; }

ARGS=(train_EPInformer.py
  --model_type EPInformer-v2
  --fold "$FOLD" --cell "$CELL" --expr_type "$EXPR_TYPE"
  --n_enh_feats "$N_ENH_FEATS"
  --h5_path "$H5_PATH" --expr_csv "$EXPR_CSV" --split_csv "$SPLIT_CSV"
  --output_dir "$OUTPUT_DIR" --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE" --lr "$LR" --early_stop_patience 8)
if [[ -n "$PRETRAINED_DIR" ]]; then
  ARGS+=(--use_pretrained_encoder --pretrained_encoder_dir "$PRETRAINED_DIR")
fi
if [[ "$NO_FREEZE" == "1" ]]; then
  echo "encoder fine-tuning ON (--no_freeze_encoder)"
  ARGS+=(--no_freeze_encoder)
fi
if [[ "$USE_PRM_SIGNAL" == "1" ]]; then
  [[ -f "$GENE_LIST" ]] || { echo "ERROR: missing GeneList ${GENE_LIST} (needed for USE_PRM_SIGNAL=1)"; exit 1; }
  echo "promoter signal ON: GENE_LIST=${GENE_LIST}"
  ARGS+=(--use_prm_signal --gene_list "$GENE_LIST")
fi

exec "$py" "${ARGS[@]}"
