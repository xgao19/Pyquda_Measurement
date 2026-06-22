#!/bin/bash
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J ringed_hp_s8t8
#SBATCH -o log/ringed_hp_s8t8.%j.o
#SBATCH -e log/ringed_hp_s8t8.%j.e
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"
mkdir -p log

module purge
module load cpe/25.09
module load gcc-native/14
module load cudatoolkit/12.9
module load craype-accel-nvidia80
module load cray-mpich-abi/9.0.1
module load cray-hdf5-parallel/1.14.3.7

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

ranks="${FLOWED_RINGED_NRANKS:-${SLURM_NTASKS:-4}}"
mpi_geometry="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.4}"

echo "script_dir=$script_dir"
echo "measurement_root=$measurement_root"
echo "FLOWED_RINGED_NRANKS=$ranks"
echo "FLOWED_RINGED_MPI_GEOMETRY=$mpi_geometry"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

env \
  FLOWED_RINGED_NRANKS="$ranks" \
  FLOWED_RINGED_MPI_GEOMETRY="$mpi_geometry" \
  bash "$script_dir/run_s8t8_hp_convergence.sh"
