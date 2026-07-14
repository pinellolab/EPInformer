#!/bin/bash
# Faithful BSCC sequence+activity extraction (scripts/seq_activity_extract.py).
# Reproduces process_EPInformer_data_v2.ipynb cells 35+38: 5-bin regions, pysam read
# counts, Activity = sqrt(DHS.RPM * H3K27ac.RPM). No negatives.
#   CELL=GM12878 NARROWPEAK=... DNASE_BAM=... H3K27AC_BAM=... OUT=... \
#     sbatch --export=ALL,CONDA_ENV=EPInformer_env,... slurm/seq_activity_extract.slurm
#
# GM12878 REPRODUCES BSCC 0.617 (2 filtered reps/assay, mean-pool). Quote the BAM lists so
# the space-separated reps survive into the script (line ~40 expands them UNQUOTED on purpose):
#   CELL=GM12878 POOL_METHOD=mean \
#     NARROWPEAK=reference/GM12878_H3K27ac.ENCFF023LTU.narrowPeak \
#     DNASE_BAM="data/GM12878/DNase/ENCFF729UYK.bam data/GM12878/DNase/ENCFF020WZB.bam" \
#     H3K27AC_BAM="data/GM12878/H3K27ac/ENCFF269GKF.bam data/GM12878/H3K27ac/ENCFF201OHW.bam" \
#     OUT=batch_output/GM12878/links/GM12878_peak_5bins_around_summit_activity_sequence.csv \
#     sbatch --export=ALL,CONDA_ENV=EPInformer_env,CELL,POOL_METHOD,NARROWPEAK,DNASE_BAM,H3K27AC_BAM,OUT slurm/seq_activity_extract.slurm
#
#SBATCH --job-name=seq_activity_extract
#SBATCH --output=log_cpu/%x_%j.out
#SBATCH --error=log_cpu/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/reproducible_pipeline}"
cd "$REPO_ROOT"
mkdir -p log_cpu

CELL="${CELL:-GM12878}"
NARROWPEAK="${NARROWPEAK:?set NARROWPEAK to the H3K27ac narrowPeak}"
DNASE_BAM="${DNASE_BAM:?set DNASE_BAM}"
H3K27AC_BAM="${H3K27AC_BAM:?set H3K27AC_BAM}"
FASTA="${FASTA:-./data/reference/hg38/hg38.fa}"
OUT="${OUT:?set OUT csv path}"
EXTEND="${EXTEND:-0}"   # ABC-style read extension to N bp before counting (0 = raw overlap)
POOL_METHOD="${POOL_METHOD:-mean}"  # mean = avg per-rep RPM (BSCC recipe, DEFAULT); sum = 1e6*Sc/Sm depth-weighted

MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV:-EPInformer_env}"; set -u
export PYTHONUNBUFFERED=1

echo "host=$(hostname) CELL=${CELL} OUT=${OUT}"
echo "DNASE_BAM=${DNASE_BAM}  H3K27AC_BAM=${H3K27AC_BAM}"
# DNASE_BAM / H3K27AC_BAM may be space-separated lists of replicate BAMs (pooled) -> intentionally unquoted.
exec "${CONDA_PREFIX}/bin/python" scripts/seq_activity_extract.py \
  --narrowpeak "$NARROWPEAK" --dnase-bam $DNASE_BAM --h3k27ac-bam $H3K27AC_BAM \
  --fasta "$FASTA" --cell "$CELL" --out "$OUT" --extend "$EXTEND" --pool-method "$POOL_METHOD"
