#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 || "$1" != "--config_num" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 --config_num CFG" >&2
  exit 2
fi
config_num="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"
source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/flowed_quark_ringed_norm}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/flowed_quark_ringed_norm}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

mpi_geometry="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.1}"
data_dir="${FLOWED_RINGED_DATA_DIR:-$script_dir/data}"
gauge_path="${FLOWED_RINGED_GAUGE_PATH:-$measurement_root/test_gauge/S8T8_wilson_b6.0}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$data_dir"

echo "Running flowed-quark ringed normalization"
echo "  FLOWED_RINGED_GAUGE_PATH=$gauge_path"
echo "  FLOWED_RINGED_DATA_DIR=$data_dir"
echo "  config_num=$config_num"
echo "  FLOWED_RINGED_MPI_GEOMETRY=$mpi_geometry"
echo "  FLOWED_RINGED_LAT_TAG=${FLOWED_RINGED_LAT_TAG:-S8T8}"
echo "  FLOWED_RINGED_NOISE_SCHEME=${FLOWED_RINGED_NOISE_SCHEME:-zn}"
echo "  FLOWED_RINGED_N_VEC=${FLOWED_RINGED_N_VEC:-1}"
echo "  FLOWED_RINGED_HP_NUM_VECTORS=${FLOWED_RINGED_HP_NUM_VECTORS:-1}"
echo "  FLOWED_RINGED_SPIN_COLOR_DILUTION=${FLOWED_RINGED_SPIN_COLOR_DILUTION:-none}"
echo "  QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH"
echo "  CUPY_CACHE_DIR=$CUPY_CACHE_DIR"

python3 -u "$script_dir/Pyquda_flowed_quark_ringed_norm.py" \
  --config_num "$config_num" \
  --mpi_geometry "$mpi_geometry"
