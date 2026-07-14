#!/bin/bash
set -euo pipefail

config_num=""
mg_block="8.8.4.4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num)
      [[ -z "$config_num" && $# -ge 2 && "${2:-}" =~ ^[0-9]+$ ]] || {
        echo "--config_num requires one non-negative integer and may appear once" >&2
        exit 2
      }
      config_num="$2"
      shift 2
      ;;
    --mg-block)
      [[ $# -ge 2 && -n "${2:-}" ]] || {
        echo "--mg-block requires a value such as 8.8.4.4" >&2
        exit 2
      }
      mg_block="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -z "$config_num" ]]; then
  echo "Usage: $0 --config_num CFG [--mg-block X.Y.Z.T[;X.Y.Z.T]]" >&2
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
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/quark_1pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/quark_1pt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export EMT_1PT_DATA_DIR="${EMT_1PT_DATA_DIR:-$script_dir/data}"
export EMT_1PT_GAUGE_PATH="${EMT_1PT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export EMT_1PT_MPI_GEOMETRY="${EMT_1PT_MPI_GEOMETRY:-1.1.1.1}"
export EMT_1PT_QMAX="${EMT_1PT_QMAX:-0}"
export EMT_1PT_FLOW_STEPS="${EMT_1PT_FLOW_STEPS:-1}"
export EMT_1PT_TOL="${EMT_1PT_TOL:-1e-10}"
export EMT_1PT_N_ZN="${EMT_1PT_N_ZN:-4}"
export EMT_1PT_RAND_SEED="${EMT_1PT_RAND_SEED:-0}"
export EMT_1PT_NOISE_SCHEME="${EMT_1PT_NOISE_SCHEME:-zn}"
export EMT_1PT_HP_NUM_VECTORS="${EMT_1PT_HP_NUM_VECTORS:-1}"
export EMT_1PT_HP_ORDERING="${EMT_1PT_HP_ORDERING:-interleaved_xyzt_binary_projected_to_evenodd}"
export EMT_1PT_BASE_START="${EMT_1PT_BASE_START:-0}"
export EMT_1PT_BASE_STOP="${EMT_1PT_BASE_STOP:-$EMT_1PT_N_VEC}"
export EMT_1PT_BLOCK_INTERVAL_SOLVES="${EMT_1PT_BLOCK_INTERVAL_SOLVES:-64}"
export EMT_1PT_SHARD_DIR="${EMT_1PT_SHARD_DIR:-$EMT_1PT_DATA_DIR/EMTc/shards}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$EMT_1PT_DATA_DIR"

echo "Running disconnected quark EMT 1pt"
python3 -u "$script_dir/Pyquda_EMT_disconnected_quark_1pt.py" \
  --config_num "$config_num" \
  --mpi_geometry "$EMT_1PT_MPI_GEOMETRY" \
  --mg-block "$mg_block"
