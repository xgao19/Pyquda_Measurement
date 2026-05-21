#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/pion_current_current_response}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-$QUDA_RESOURCE_PATH/profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/pion_current_current_response}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export PION_CC_RESPONSE_DATA_DIR="${PION_CC_RESPONSE_DATA_DIR:-$script_dir/data}"
export PION_CC_RESPONSE_GAUGE_PATH="${PION_CC_RESPONSE_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export PION_CC_RESPONSE_CONFIG_NUM="${PION_CC_RESPONSE_CONFIG_NUM:-0}"
export PION_CC_RESPONSE_MPI_GEOMETRY="${PION_CC_RESPONSE_MPI_GEOMETRY:-1.1.1.1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$PION_CC_RESPONSE_DATA_DIR"

python3 -u "$script_dir/Pyquda_pion_current_current_response.py" "$@"
