#!/bin/bash
#SBATCH -A nph174
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J EMTq3pt
#SBATCH -o log/EMTq3pt.%j.o
#SBATCH -e log/EMTq3pt.%j.e
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

set -euo pipefail

if [[ $# -ne 2 || "$1" != "--config_num" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 --config_num CFG" >&2
  exit 2
fi
config_num="$2"

# Perlmutter job wrapper for the quark EMT connected 3pt measurement.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
emt_root="$(cd "$script_dir/.." && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m4559/xgao/software_gradientflow}"
measurement_root="${REPO_ROOT:-$software_root/Pyquda_Measurement}"

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

export QUDA_PATH="${QUDA_PATH:-$software_root/quda-develop/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-1}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-$QUDA_RESOURCE_PATH/profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export EMT_DATA_DIR="${EMT_DATA_DIR:-$emt_root/data}"
export EMT_GAUGE_PATH="${EMT_GAUGE_PATH:?Set EMT_GAUGE_PATH to the input gauge file before submitting}"
export PYTHONPATH="$measurement_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$EMT_DATA_DIR"

echo "script_dir=$script_dir"
echo "emt_root=$emt_root"
echo "EMT_DATA_DIR=$EMT_DATA_DIR"
echo "EMT_GAUGE_PATH=$EMT_GAUGE_PATH"
echo "QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH"
echo "CUPY_CACHE_DIR=$CUPY_CACHE_DIR"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

main="Pyquda_EMT_quark_3pt.py"
mpi_geometry="${EMT_MPI_GEOMETRY:-2.2.2.4}"

srun --mpi=cray_shasta -n "${SLURM_NTASKS:-32}" \
  python3 "$script_dir/$main" --config_num "$config_num" --mpi_geometry "$mpi_geometry"
