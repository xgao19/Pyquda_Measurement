#!/bin/bash
#SBATCH --job-name=nucleon_TMD
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/nucleon_TMD.%j.out
#SBATCH --error=logs/nucleon_TMD.%j.err

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_num=""
mg_block="8.8.4.4"
t_separations="2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num)
      config_num="${2:-}"
      shift 2
      ;;
    --mg-block)
      mg_block="${2:-}"
      shift 2
      ;;
    --t_separations)
      t_separations="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ ! "$config_num" =~ ^[0-9]+$ || -z "$mg_block" ]]; then
  echo "Usage: sbatch $0 --config_num CFG [--mg-block X.Y.Z.T[;...]|none] [--t_separations TSEP]" >&2
  exit 2
fi

mkdir -p "$script_dir/logs"

export NUCLEON_TMD_MPI_GEOMETRY="${NUCLEON_TMD_MPI_GEOMETRY:-1.1.1.1}"
srun --cpu-bind=cores "$script_dir/run_nucleon_TMD.sh" \
  --config_num "$config_num" --mg-block "$mg_block" \
  --t_separations "$t_separations"
