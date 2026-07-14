#!/bin/bash
# Build the gene-level HDF5 ({cell}_samples.h5) for a cell line from PRECOMPUTED
# ABC nomination outputs (no raw BAMs required). Extends expression prediction to
# H1/HepG2/HUVEC/NHEK using precomputed ABC links (avg Hi-C).
#
# Usage (from repo root on HPC):
#   mkdir -p log_cpu
#   CELL=H1    sbatch slurm/build_gene_h5.slurm
#   CELL=HepG2 sbatch slurm/build_gene_h5.slurm
#   CELL=HUVEC sbatch slurm/build_gene_h5.slurm
#   CELL=NHEK  sbatch slurm/build_gene_h5.slurm
#
# Writes:  batch_output/{CELL}/encoding/{CELL}_samples.h5
#          batch_output/{CELL}/links/GeneList.txt        (staged for --use_prm_signal)
#
# Env overrides: ABC_ROOT, EXPR_CSV, FASTA, CONDA_ENV
#
#SBATCH --job-name=epi_build_h5
#SBATCH --output=log_cpu/%x_%j.out
#SBATCH --error=log_cpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p cpu1
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/pipeline}"
cd "$REPO_ROOT"
mkdir -p log_cpu

CELL="${CELL:?set CELL=H1|HepG2|HUVEC|NHEK}"
ABC_ROOT="${ABC_ROOT:-/path/to/abc_links}"

# Per-cell ABC directories. Neighborhoods (EnhancerList/GeneList) and the matching
# avg-Hi-C predictions dir. Folder names are the reference (note H1's predictions dir is
# truncated to ENCFF761ZR, and NHEK's dir is mis-named ENCFF862NDZ but the DATA
# inside uses the correct NHEK BAM DHS.ENCFF117RNM — verified by column headers).
case "$CELL" in
  H1)    NB=DNase_ENCFF761ZRE_Neighborhoods; PR=DNase_ENCFF761ZR_avghic_noCutOff_1MB_predictions;;
  HepG2) NB=DNase_ENCFF691HJY_Neighborhoods; PR=DNase_ENCFF691HJY_avghic_noCutOff_1MB_predictions;;
  HUVEC) NB=DNase_ENCFF091KTX_Neighborhoods; PR=DNase_ENCFF091KTX_avghic_noCutOff_1MB_predictions;;
  NHEK)  NB=DNase_ENCFF862NDZ_Neighborhoods; PR=DNase_ENCFF862NDZ_avghic_noCutOff_1MB_predictions;;
  *) echo "ERROR: unknown CELL=$CELL"; exit 1;;
esac

EL="$ABC_ROOT/$CELL/$NB/EnhancerList.txt"
GL="$ABC_ROOT/$CELL/$NB/GeneList.txt"
PF="$ABC_ROOT/$CELL/$PR/EnhancerPredictionsAllPutative.txt.gz"
EXPR_CSV="${EXPR_CSV:-./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv}"
FASTA="${FASTA:-./data/reference/hg38/hg38.fa}"

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
export PATH="${PATH:-}:${MINICONDA_ROOT}/bin:${MINICONDA_ROOT}/condabin"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-EPInformer_env}"; set -u
py="${CONDA_PREFIX}/bin/python"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE

echo "host=$(hostname) CELL=$CELL"
echo "EL=$EL"
echo "PF=$PF"
echo "GL=$GL"
for f in "$EL" "$PF" "$GL" "$FASTA" "$EXPR_CSV"; do
  [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

mkdir -p "batch_output/$CELL/encoding" "batch_output/$CELL/links"
# stage GeneList so train_EPInformer's --use_prm_signal (GENE_LIST default) finds it
cp -f "$GL" "batch_output/$CELL/links/GeneList.txt"

exec "$py" scripts/build_gene_h5_for_cell.py \
  --cell "$CELL" \
  --predictions "$PF" \
  --enhancer_list "$EL" \
  --fasta "$FASTA" \
  --expr_csv "$EXPR_CSV" \
  --output_dir "batch_output/$CELL/encoding" \
  --max_distance 100000 --n_enhancer 60 --max_seq_len 2000 \
  --include_self_promoter
