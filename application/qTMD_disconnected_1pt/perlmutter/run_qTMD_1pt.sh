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
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/qTMD_1pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/qTMD_1pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export QTMD_1PT_DATA_DIR="${QTMD_1PT_DATA_DIR:-$script_dir/data}"
export QTMD_1PT_GAUGE_PATH="${QTMD_1PT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export QTMD_1PT_CONFIG_NUM="${QTMD_1PT_CONFIG_NUM:-0}"
export QTMD_1PT_MPI_GEOMETRY="${QTMD_1PT_MPI_GEOMETRY:-1.1.1.1}"
export QTMD_1PT_OPERATOR_KIND="${QTMD_1PT_OPERATOR_KIND:-GI_PDF}"
export QTMD_1PT_QMAX="${QTMD_1PT_QMAX:-0}"
export QTMD_1PT_ETA="${QTMD_1PT_ETA:-0}"
export QTMD_1PT_BZ="${QTMD_1PT_BZ:-0}"
export QTMD_1PT_BT="${QTMD_1PT_BT:-0}"
export QTMD_1PT_N_VEC="${QTMD_1PT_N_VEC:-1}"
export QTMD_1PT_N_ZN="${QTMD_1PT_N_ZN:-4}"
export QTMD_1PT_RAND_SEED="${QTMD_1PT_RAND_SEED:-0}"
export QTMD_1PT_TOL="${QTMD_1PT_TOL:-1e-10}"
export QTMD_1PT_NOISE_SCHEME="${QTMD_1PT_NOISE_SCHEME:-zn}"
export QTMD_1PT_HP_NUM_VECTORS="${QTMD_1PT_HP_NUM_VECTORS:-1}"
export QTMD_1PT_HP_ORDERING="${QTMD_1PT_HP_ORDERING:-global_xyzt_gray_projected_to_evenodd}"
export QTMD_1PT_GI_STAPLE_MODE="${QTMD_1PT_GI_STAPLE_MODE:-link_cache}"
export QTMD_1PT_OUTPUT_MODE="${QTMD_1PT_OUTPUT_MODE:-base_shards}"
export QTMD_1PT_BASE_START="${QTMD_1PT_BASE_START:-0}"
export QTMD_1PT_BASE_STOP="${QTMD_1PT_BASE_STOP:-$QTMD_1PT_N_VEC}"
export QTMD_1PT_BLOCK_INTERVAL_SOLVES="${QTMD_1PT_BLOCK_INTERVAL_SOLVES:-64}"
export QTMD_1PT_SHARD_DIR="${QTMD_1PT_SHARD_DIR:-$QTMD_1PT_DATA_DIR/qTMD1pt/shards}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$QTMD_1PT_DATA_DIR"

echo "Running disconnected qTMD 1pt"
python3 -u "$script_dir/Pyquda_Disconnected_qTMD_1pt.py" \
  --config_num "$QTMD_1PT_CONFIG_NUM" \
  --mpi_geometry "$QTMD_1PT_MPI_GEOMETRY"
