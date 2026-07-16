#!/bin/bash
# Submit the full K562 (or GM12878) pipeline as a dependency-chained SLURM pipeline:
#   Stage 1 ABC links (cpu1) -> [Stage 2 encoding (cpu1)  +  encoder pretrain (gpu33 array)]
#   -> train EPInformer_v2 (gpu33 array, aftercorr per-fold on pretrain + afterok encode)
#
# Assumes data/ is present (symlink or real) and the conda env has the deps.
# Usage (from repo root on HPC):
#   bash slurm/submit_pipeline.sh                 # CELL=K562, CONDA_ENV=epinformer_repro
#   CELL=GM12878 CONDA_ENV=EPInformer_env bash slurm/submit_pipeline.sh
#   N_ENH_FEATS=2 bash slurm/submit_pipeline.sh   # activity-only (no Hi-C)
#   GPU_PARTITION=gpu2 bash slurm/submit_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
mkdir -p log_cpu log_gpu

CELL="${CELL:-K562}"
CONDA_ENV="${CONDA_ENV:-epinformer_repro}"
N_ENH_FEATS="${N_ENH_FEATS:-3}"
CPU_PARTITION="${CPU_PARTITION:-cpu1}"
GPU_PARTITION="${GPU_PARTITION:-gpu33}"
REPO_ROOT="$PWD"
EXP="--export=ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},CELL=${CELL},N_ENH_FEATS=${N_ENH_FEATS}"

JOB_LINKS=$(sbatch --parsable --partition="$CPU_PARTITION" --export=ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},SAMPLES=${CELL},STAGES=links slurm/run_pipeline_cpu.slurm)
echo "links    = $JOB_LINKS  ($CPU_PARTITION, Stage 1 ABC nomination)"

JOB_ENCODE=$(sbatch --parsable --partition="$CPU_PARTITION" --dependency=afterok:$JOB_LINKS --export=ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},SAMPLES=${CELL},STAGES=encoding slurm/run_pipeline_cpu.slurm)
echo "encode   = $JOB_ENCODE  ($CPU_PARTITION, Stage 2 HDF5, after links)"

JOB_PRE=$(sbatch --parsable --partition="$GPU_PARTITION" --dependency=afterok:$JOB_LINKS $EXP slurm/train_seqencoder_12fold.slurm)
echo "pretrain = $JOB_PRE  ($GPU_PARTITION array 1-12, after links)"

JOB_TRAIN=$(sbatch --parsable --partition="$GPU_PARTITION" --dependency=aftercorr:$JOB_PRE,afterok:$JOB_ENCODE $EXP slurm/train_epinformer_12fold.slurm)
echo "train    = $JOB_TRAIN  ($GPU_PARTITION array 1-12, per-fold aftercorr pretrain + afterok encode)"

echo
echo "Submitted chain for ${CELL}. Watch: squeue -u \$USER | grep epi_repro"
echo "When training finishes: python evaluate.py expression --pred_dir ./EPInformer_models/${CELL}"
