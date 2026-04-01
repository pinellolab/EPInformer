#!/usr/bin/env bash
# Roadmap ENCODE manifest download via aria2c in the background (nohup).
# Same invocation as slurm/download_roadmap_encode_aria2c.slurm, for hosts with outbound HTTPS
# (e.g. login or transfer node) when compute nodes cannot reach ENCODE.
#
# Usage (from anywhere):
#   bash scripts/roadmap_encode_download_login_nohup.sh
#
# Env overrides (optional):
#   CONDA_BASE=/path/to/miniconda3   CONDA_ENV=EPInformer_env
#   MANIFEST=data/roadmap_download_manifest.json
#   MANIFEST_ROOT=data/roadmap_encode_downloads
#   ARIA2C_CONN=4  ARIA2C_PARALLEL=3
#   ARIA2C_CONNECT_TIMEOUT=60  ARIA2C_TIMEOUT=600
#   LOG=log_cpu/roadmap_encode_dl_login.nohup.out
#
# Monitor: tail -f log_cpu/roadmap_encode_dl_login.nohup.out

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p log_cpu data/roadmap_encode_downloads

CONDA_ENV="${CONDA_ENV:-EPInformer_env}"
CONDA_BASE="${CONDA_BASE:-/lustre/grp/zyjlab/linjc/miniconda3}"
MANIFEST="${MANIFEST:-data/roadmap_download_manifest.json}"
MANIFEST_ROOT="${MANIFEST_ROOT:-data/roadmap_encode_downloads}"
ARIA2C_CONN="${ARIA2C_CONN:-4}"
ARIA2C_PARALLEL="${ARIA2C_PARALLEL:-3}"
ARIA2C_CONNECT_TIMEOUT="${ARIA2C_CONNECT_TIMEOUT:-60}"
ARIA2C_TIMEOUT="${ARIA2C_TIMEOUT:-600}"
LOG="${LOG:-log_cpu/roadmap_encode_dl_login.nohup.out}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh; set CONDA_BASE" >&2
  exit 1
fi

set +u
if ! conda activate "${CONDA_ENV}"; then
  set -u
  echo "ERROR: conda activate ${CONDA_ENV} failed." >&2
  exit 1
fi
set -u

export PATH="$(conda info --base)/bin:${PATH}"

py="${CONDA_PREFIX}/bin/python"
if ! [[ -x "$py" ]]; then
  echo "ERROR: not executable: ${py}" >&2
  exit 1
fi
if ! command -v aria2c >/dev/null 2>&1; then
  echo "ERROR: aria2c not on PATH. Install e.g.: conda install -n base -c conda-forge aria2" >&2
  exit 1
fi

echo "MANIFEST=${MANIFEST}"
echo "MANIFEST_ROOT=${MANIFEST_ROOT}"
echo "ARIA2C_CONN=${ARIA2C_CONN}  ARIA2C_PARALLEL=${ARIA2C_PARALLEL}"
echo "ARIA2C_CONNECT_TIMEOUT=${ARIA2C_CONNECT_TIMEOUT}  ARIA2C_TIMEOUT=${ARIA2C_TIMEOUT}"
echo "log -> ${REPO_ROOT}/${LOG}"

nohup "${py}" -u scripts/download_encode_data.py \
  --from-manifest "${MANIFEST}" \
  --manifest-root "${MANIFEST_ROOT}" \
  --aria2c \
  --aria2c-connections "${ARIA2C_CONN}" \
  --aria2c-parallel "${ARIA2C_PARALLEL}" \
  --aria2c-connect-timeout "${ARIA2C_CONNECT_TIMEOUT}" \
  --aria2c-timeout "${ARIA2C_TIMEOUT}" \
  > "${LOG}" 2>&1 &

echo "PID=$!  log: ${REPO_ROOT}/${LOG}"
echo "Monitor: tail -f ${REPO_ROOT}/${LOG}"
