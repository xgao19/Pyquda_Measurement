#!/bin/bash
#SBATCH --job-name=pion_TMD_CG
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/pion_TMD_CG.%j.out
#SBATCH --error=logs/pion_TMD_CG.%j.err

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/logs"

export PION_TMD_MPI_GEOMETRY="${PION_TMD_MPI_GEOMETRY:-1.1.1.1}"
srun --cpu-bind=cores "$script_dir/run_pion_TMD_CG.sh"
