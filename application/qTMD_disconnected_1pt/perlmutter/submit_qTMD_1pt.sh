#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J qTMD_1pt
#SBATCH -o logs/qTMD_1pt.%j.out
#SBATCH -e logs/qTMD_1pt.%j.err

set -euo pipefail

if [[ $# -ne 2 || "$1" != "--config_num" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 --config_num CFG" >&2
  exit 2
fi
config_num="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/logs"

srun -n 1 -c 32 --cpu-bind=cores bash "$script_dir/run_qTMD_1pt.sh" \
  --config_num "$config_num"
