#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J pion_soft_contract
#SBATCH -o logs/pion_soft_contract.%j.out
#SBATCH -e logs/pion_soft_contract.%j.err

mkdir -p logs
srun ./run_pion_soft_factor_contract.sh
