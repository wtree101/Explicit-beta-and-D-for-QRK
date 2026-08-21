#!/bin/bash

#################### The following is compulsory ####################
#SBATCH -J qrk_beta
### Use queue (partition) q1
#SBATCH -p q1
### Use 1 node and 32 CPU tasks
#SBATCH -N 1 -n 32
#SBATCH --mail-user=twubi@connect.ust.hk
#SBATCH --mail-type=end
#####################################################################

set -euo pipefail

# Always run from repository root
cd /home/twubi/Explicit-beta-and-D-for-QRK

# Activate conda env if available
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate base
fi

# Keep worker count bounded by allocated SLURM tasks
export QRK_NUM_WORKERS="${SLURM_NTASKS:-8}"
export QRK_RANDOM_SEED=20260814

# Use project-local Python environment (configured once on login node)
VENV_PYTHON="/home/twubi/Explicit-beta-and-D-for-QRK/.venv311/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: ${VENV_PYTHON} not found." >&2
  echo "Create it with: /usr/bin/python3.11 -m venv .venv311 && . .venv311/bin/activate && python -m pip install -r requirements.txt" >&2
  exit 1
fi

"${VENV_PYTHON}" heatmap_generation_D_vs_beta_demo.py
