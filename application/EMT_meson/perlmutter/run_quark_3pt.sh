#!/bin/bash
set -euo pipefail

config_num=""
mg_block="8.8.4.4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num) config_num="${2:-}"; shift 2 ;;
    --mg-block) mg_block="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$config_num" =~ ^[0-9]+$ || -z "$mg_block" ]]; then
  echo "Usage: $0 --config_num CFG [--mg-block X.Y.Z.T[;...]]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m4559/xgao/software_gradientflow}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda-develop/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/quark_3pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/quark_3pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export EMT_DATA_DIR="${EMT_DATA_DIR:-$script_dir/data}"
export EMT_GAUGE_PATH="${EMT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export EMT_MPI_GEOMETRY="${EMT_MPI_GEOMETRY:-1.1.1.1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$EMT_DATA_DIR/EMT3pt"

echo "Running quark EMT 3pt"
echo "  EMT_GAUGE_PATH=$EMT_GAUGE_PATH"
echo "  EMT_DATA_DIR=$EMT_DATA_DIR"
echo "  config_num=$config_num"
echo "  EMT_MPI_GEOMETRY=$EMT_MPI_GEOMETRY"

python3 -u "$script_dir/Pyquda_EMT_quark_3pt.py" \
  --config_num "$config_num" \
  --mpi_geometry "$EMT_MPI_GEOMETRY" \
  --mg-block "$mg_block"
