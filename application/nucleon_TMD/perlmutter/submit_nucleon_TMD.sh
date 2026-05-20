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
mkdir -p "$script_dir/logs"

export NUCLEON_TMD_MPI_GEOMETRY="${NUCLEON_TMD_MPI_GEOMETRY:-1.1.1.1}"
srun --cpu-bind=cores "$script_dir/run_nucleon_TMD.sh"
