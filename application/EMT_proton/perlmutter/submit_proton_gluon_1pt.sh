#!/bin/bash
#SBATCH -A m3760
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J proton_emt_g1pt
#SBATCH -o logs/proton_emt_g1pt.%j.out

set -euo pipefail
mkdir -p logs
srun ./run_proton_gluon_1pt.sh
