#!/bin/bash
#SBATCH --job-name=pion_TMD
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/pion_TMD.%j.out
#SBATCH --error=logs/pion_TMD.%j.err

set -euo pipefail

pos_boost="0.0.0"
neg_boost="0.0.0"
t_separations="2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pos-boost)
      [[ $# -ge 2 ]] || { echo "Missing value for --pos-boost" >&2; exit 2; }
      pos_boost="$2"
      shift 2
      ;;
    --neg-boost)
      [[ $# -ge 2 ]] || { echo "Missing value for --neg-boost" >&2; exit 2; }
      neg_boost="$2"
      shift 2
      ;;
    --t_separations)
      [[ $# -ge 2 ]] || { echo "Missing value for --t_separations" >&2; exit 2; }
      t_separations="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/logs"

export PION_TMD_MPI_GEOMETRY="${PION_TMD_MPI_GEOMETRY:-1.1.1.1}"
srun --cpu-bind=cores "$script_dir/run_pion_TMD.sh" \
  --pos-boost "$pos_boost" \
  --neg-boost "$neg_boost" \
  --t_separations "$t_separations"
