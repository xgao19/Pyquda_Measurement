#!/bin/bash
#SBATCH -A m3760
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J flowed_disc_g1pt
#SBATCH -o logs/flowed_disc_g1pt.%j.out

set -euo pipefail
mkdir -p logs
srun ./run_flowed_disc_gluon_1pt.sh
