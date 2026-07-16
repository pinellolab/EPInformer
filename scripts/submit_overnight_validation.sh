#!/bin/bash
# Submit the complete EPInformer overnight validation matrix.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p log_cpu log_gpu hpc_test_outputs

SBATCH="${SBATCH:-sbatch}"
CPU_PARTITION="${CPU_PARTITION:-cpu1}"
# The caller should probe gpu33 first and set this to gpu31 when gpu33 has no
# near-term capacity. A comma-separated partition list is intentionally avoided:
# this cluster preserves list order and can wait on gpu33 instead of falling back.
GPU_PARTITION="${GPU_PARTITION:-gpu33}"
CONDA_ENV="${CONDA_ENV:-EPInformer_env}"
SOURCE_ROOT="${SOURCE_ROOT:-/lustre/grp/zyjlab/linjc/epinformer_reproduce}"
ABC_ROOT="${ABC_ROOT:-/lustre/grp/zyjlab/linjc/BSCC_GPU/BSCC_GPU/EPInformer/epinformer_data_20250503}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-$REPO_ROOT/hpc_test_outputs/overnight_jobs_${STAMP}.tsv}"

printf 'cell\tstage\tjob_id\tdependency\toutput_dir\n' > "$MANIFEST"

record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$MANIFEST"
  printf '%-9s %-18s job=%s dep=%s\n' "$1" "$2" "$3" "${4:--}"
}

submit_eval() {
  local cell="$1" stage="$2" dependency="$3" mode="$4" pred_dir="$5" out_dir="$6"
  local job
  job=$($SBATCH --parsable --partition="$CPU_PARTITION" \
    --dependency="afterok:${dependency}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",MODE="$mode",PRED_DIR="$pred_dir",OUT_DIR="$out_dir",LABEL="${cell}_${stage}" \
    slurm/evaluate_results.slurm)
  record "$cell" "eval_${stage}" "$job" "$dependency" "$out_dir"
}

submit_primary_cell() {
  local cell="$1"
  local links encode pre rna cage
  local pre_out="$REPO_ROOT/results/seqencoder/$cell"
  local h5="$REPO_ROOT/batch_output/$cell/encoding/${cell}_samples.h5"
  local rna_out="$REPO_ROOT/EPInformer_models/${cell}_overnight_RNA"
  local cage_out="$REPO_ROOT/EPInformer_models/${cell}_overnight_CAGE"

  links=$($SBATCH --parsable --partition="$CPU_PARTITION" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",SAMPLES="$cell",STAGES=links \
    slurm/run_pipeline_cpu.slurm)
  record "$cell" links "$links" - "$REPO_ROOT/batch_output/$cell/links"

  encode=$($SBATCH --parsable --partition="$CPU_PARTITION" --dependency="afterok:${links}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",SAMPLES="$cell",STAGES=encoding \
    slurm/run_pipeline_cpu.slurm)
  record "$cell" encoding "$encode" "$links" "$(dirname "$h5")"

  pre=$($SBATCH --parsable --partition="$GPU_PARTITION" --dependency="afterok:${links}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",OUTPUT_DIR="$pre_out" \
    slurm/train_seqencoder_12fold.slurm)
  record "$cell" encoder_12fold "$pre" "$links" "$pre_out"

  rna=$($SBATCH --parsable --partition="$GPU_PARTITION" \
    --dependency="aftercorr:${pre},afterok:${encode}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",EXPR_TYPE=RNA,H5_PATH="$h5",PRETRAINED_DIR="$pre_out/checkpoints",OUTPUT_DIR="$rna_out" \
    slurm/train_epinformer_12fold.slurm)
  record "$cell" expression_RNA "$rna" "$pre,$encode" "$rna_out"

  cage=$($SBATCH --parsable --partition="$GPU_PARTITION" \
    --dependency="aftercorr:${pre},afterok:${encode}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",EXPR_TYPE=CAGE,H5_PATH="$h5",PRETRAINED_DIR="$pre_out/checkpoints",OUTPUT_DIR="$cage_out" \
    slurm/train_epinformer_12fold.slurm)
  record "$cell" expression_CAGE "$cage" "$pre,$encode" "$cage_out"

  submit_eval "$cell" encoder "$pre" encoder "$pre_out" "$pre_out/evaluation"
  submit_eval "$cell" RNA "$rna" expression "$rna_out" "$rna_out/evaluation"
  submit_eval "$cell" CAGE "$cage" expression "$cage_out" "$cage_out/evaluation"
}

submit_secondary_cell() {
  local cell="$1"
  local build pre rna
  local data_csv="$SOURCE_ROOT/data/enhancer_sequences_old/${cell}_peak_5bins_around_summit_activity_sequence.csv"
  local pre_out="$REPO_ROOT/results/seqencoder/$cell"
  local h5="$REPO_ROOT/batch_output/$cell/encoding/${cell}_samples.h5"
  local rna_out="$REPO_ROOT/EPInformer_models/${cell}_overnight_RNA"

  [[ -s "$data_csv" ]] || { echo "Missing encoder CSV: $data_csv" >&2; return 1; }

  build=$($SBATCH --parsable --partition="$CPU_PARTITION" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",ABC_ROOT="$ABC_ROOT" \
    slurm/build_gene_h5.slurm)
  record "$cell" encoding "$build" - "$(dirname "$h5")"

  pre=$($SBATCH --parsable --partition="$GPU_PARTITION" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",DATA_CSV="$data_csv",OUTPUT_DIR="$pre_out" \
    slurm/train_seqencoder_12fold.slurm)
  record "$cell" encoder_12fold "$pre" - "$pre_out"

  rna=$($SBATCH --parsable --partition="$GPU_PARTITION" \
    --dependency="aftercorr:${pre},afterok:${build}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",CELL="$cell",EXPR_TYPE=RNA,H5_PATH="$h5",PRETRAINED_DIR="$pre_out/checkpoints",OUTPUT_DIR="$rna_out" \
    slurm/train_epinformer_12fold.slurm)
  record "$cell" expression_RNA "$rna" "$pre,$build" "$rna_out"

  submit_eval "$cell" encoder "$pre" encoder "$pre_out" "$pre_out/evaluation"
  submit_eval "$cell" RNA "$rna" expression "$rna_out" "$rna_out/evaluation"
}

for cell in K562 GM12878; do
  submit_primary_cell "$cell"
done
for cell in H1 HepG2 HUVEC NHEK; do
  submit_secondary_cell "$cell"
done

echo "Job manifest: $MANIFEST"
