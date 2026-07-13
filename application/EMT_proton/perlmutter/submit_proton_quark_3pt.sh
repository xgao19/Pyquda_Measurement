#!/bin/bash
#SBATCH -A m3760
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J proton_emt_q3pt
#SBATCH -o logs/proton_emt_q3pt.%j.out

set -euo pipefail
if [[ $# -ne 2 || "$1" != "--config_num" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 --config_num CFG" >&2
  exit 2
fi
mkdir -p logs
srun ./run_proton_quark_3pt.sh "$@"
