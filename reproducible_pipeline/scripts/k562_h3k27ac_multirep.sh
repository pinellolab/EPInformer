#!/bin/bash
# K562 H3K27ac multi-rep variant: pool all 3 H3K27ac bio-reps (ENCSR000AKP) + single DNase
# (K562 DNase ENCSR000EKS has only 1 rep). Compare vs single-rep reproduction (0.740).
# NOTE: rep3 (ENCFF232RQF) is far deeper than reps 1/2, so mean-pool may dilute it.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/path/to/EPInformer/reproducible_pipeline}"
cd "$REPO_ROOT"
source ${HOME}/miniconda3/etc/profile.d/conda.sh
set +u; conda activate EPInformer_env; set -u

H3=data/K562/H3K27ac
declare -A ACC=(
  ["$H3/ENCFF600THN.bam"]=ENCFF600THN   # H3K27ac rep1 filtered 465MB
  ["$H3/ENCFF879BWC.bam"]=ENCFF879BWC   # H3K27ac rep2 filtered 190MB
)
for path in "${!ACC[@]}"; do
  acc=${ACC[$path]}
  [[ -f $path ]] || { echo "[dl] $acc"; wget -q -O "$path" "https://www.encodeproject.org/files/$acc/@@download/$acc.bam"; }
  if ! samtools index -@4 "$path" 2>/dev/null; then
    echo "[sort] $acc"; samtools sort -@4 -m 2G -o "${path%.bam}.srt.bam" "$path"; mv "${path%.bam}.srt.bam" "$path"; samtools index -@4 "$path"
  fi
  echo "[stat] $acc mapped=$(samtools idxstats "$path" | awk '{s+=$3} END{print s}')"
done
echo "[ok] rep1+rep2 present; rep3 ENCFF232RQF mapped=$(samtools idxstats $H3/ENCFF232RQF.bam | awk '{s+=$3} END{print s}')"

export CONDA_ENV=EPInformer_env CELL=K562 POOL_METHOD=mean FASTA=./data/reference/hg38/hg38.fa
export NARROWPEAK=reference/K562_H3K27ac.ENCFF544LXB.narrowPeak
export DNASE_BAM="data/K562/DNase/ENCFF257HEE.bam"
export H3K27AC_BAM="data/K562/H3K27ac/ENCFF232RQF.bam data/K562/H3K27ac/ENCFF600THN.bam data/K562/H3K27ac/ENCFF879BWC.bam"
export OUT=batch_output/K562/links/K562_h3k27ac3rep_5bins_around_summit_activity_sequence.csv
EX=$(sbatch --parsable --export=ALL slurm/bscc_extract.slurm)
echo "[submit] extract=$EX (3 H3K27ac reps mean-pool + 1 DNase)"
export DATA_CSV="$OUT" OUTPUT_DIR=./results/seqencoder/K562_multirep_h3k27ac
TR=$(sbatch --parsable --dependency=afterok:$EX --export=ALL slurm/train_seqencoder_12fold.slurm)
echo "[submit] train=$TR"
export CKPT_DIR=./results/seqencoder/K562_multirep_h3k27ac/checkpoints OUTPUT_DIR=./results/seqencoder/K562_multirep_h3k27ac_fwdRC
EV=$(sbatch --parsable --dependency=afterok:$TR --export=ALL slurm/eval_seqencoder_fwdrc.slurm)
echo "[submit] eval=$EV"
echo "[chain] K562-multirep: extract $EX -> train $TR -> eval $EV"
