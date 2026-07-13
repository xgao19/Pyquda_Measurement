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
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/gluon_1pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/gluon_1pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export EMT_1PT_DATA_DIR="${EMT_1PT_DATA_DIR:-$script_dir/data}"
export EMT_1PT_GAUGE_PATH="${EMT_1PT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export EMT_1PT_MPI_GEOMETRY="${EMT_1PT_MPI_GEOMETRY:-1.1.1.1}"
export EMT_1PT_QMAX="${EMT_1PT_QMAX:-0}"
export EMT_1PT_FLOW_STEPS="${EMT_1PT_FLOW_STEPS:-1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$EMT_1PT_DATA_DIR"

echo "Running disconnected gluon EMT 1pt"
python3 -u "$script_dir/Pyquda_EMT_disconnected_gluon_1pt.py" \
  --config_num "$config_num" \
  --mpi_geometry "$EMT_1PT_MPI_GEOMETRY"
