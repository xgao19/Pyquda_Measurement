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
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/quark_1pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/quark_1pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export EMT_1PT_DATA_DIR="${EMT_1PT_DATA_DIR:-$script_dir/data}"
export EMT_1PT_GAUGE_PATH="${EMT_1PT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export EMT_1PT_CONFIG_NUM="${EMT_1PT_CONFIG_NUM:-0}"
export EMT_1PT_MPI_GEOMETRY="${EMT_1PT_MPI_GEOMETRY:-1.1.1.1}"
export EMT_1PT_QMAX="${EMT_1PT_QMAX:-0}"
export EMT_1PT_FLOW_STEPS="${EMT_1PT_FLOW_STEPS:-1}"
export EMT_1PT_TOL="${EMT_1PT_TOL:-1e-10}"
export EMT_1PT_N_ZN="${EMT_1PT_N_ZN:-4}"
export EMT_1PT_RAND_SEED="${EMT_1PT_RAND_SEED:-0}"
export EMT_1PT_NOISE_SCHEME="${EMT_1PT_NOISE_SCHEME:-zn}"
export EMT_1PT_HP_NUM_VECTORS="${EMT_1PT_HP_NUM_VECTORS:-1}"
export EMT_1PT_HP_ORDERING="${EMT_1PT_HP_ORDERING:-interleaved_xyz_binary_projected_to_evenodd}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$EMT_1PT_DATA_DIR"

echo "Running disconnected quark EMT 1pt"
python3 -u "$script_dir/Pyquda_EMT_disconnected_quark_1pt.py" \
  --config_num "$EMT_1PT_CONFIG_NUM" \
  --mpi_geometry "$EMT_1PT_MPI_GEOMETRY"
