#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"
source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/flowed_disc_quark_1pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/flowed_disc_quark_1pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export FLOWED_DISC_DATA_DIR="${FLOWED_DISC_DATA_DIR:-$script_dir/data}"
export FLOWED_DISC_GAUGE_PATH="${FLOWED_DISC_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export FLOWED_DISC_CONFIG_NUM="${FLOWED_DISC_CONFIG_NUM:-0}"
export FLOWED_DISC_MPI_GEOMETRY="${FLOWED_DISC_MPI_GEOMETRY:-1.1.1.1}"
export FLOWED_DISC_QMAX="${FLOWED_DISC_QMAX:-0}"
export FLOWED_DISC_FLOW_STEPS="${FLOWED_DISC_FLOW_STEPS:-1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$FLOWED_DISC_DATA_DIR"

echo "Running flowed disconnected quark 1pt"
python3 -u "$script_dir/Pyquda_flowed_disc_quark_1pt.py" \
  --config_num "$FLOWED_DISC_CONFIG_NUM" \
  --mpi_geometry "$FLOWED_DISC_MPI_GEOMETRY"
