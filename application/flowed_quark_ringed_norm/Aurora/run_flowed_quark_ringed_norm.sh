#!/bin/bash
set -euo pipefail

config_num=""
flow_batch_size=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num)
      [[ -z "$config_num" && $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        echo "Invalid or repeated --config_num" >&2; exit 2;
      }
      config_num="$2"; shift 2 ;;
    --flow-batch-size)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
        echo "--flow-batch-size requires a positive integer" >&2; exit 2;
      }
      flow_batch_size="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$config_num" ]] || {
  echo "Usage: $0 --config_num CFG [--flow-batch-size B]" >&2; exit 2;
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
measurement_root="$(cd "$script_dir/../../.." && pwd)"

cd "$script_dir"
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh

export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.cache/flowed_quark_ringed_norm}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export FLOWED_RINGED_DATA_DIR="${FLOWED_RINGED_DATA_DIR:-$script_dir/data}"
export FLOWED_RINGED_GAUGE_PATH="${FLOWED_RINGED_GAUGE_PATH:-$measurement_root/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export FLOWED_RINGED_LAT_TAG="${FLOWED_RINGED_LAT_TAG:-S8T32}"
export FLOWED_RINGED_MPI_GEOMETRY="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.2}"
export FLOWED_RINGED_FLOW_STEPS="${FLOWED_RINGED_FLOW_STEPS:-1}"
export FLOWED_RINGED_NOISE_SCHEME="${FLOWED_RINGED_NOISE_SCHEME:-zn}"
export FLOWED_RINGED_HP_NUM_VECTORS="${FLOWED_RINGED_HP_NUM_VECTORS:-1}"
export FLOWED_RINGED_HP_ORDERING="${FLOWED_RINGED_HP_ORDERING:-global_xyzt_gray_projected_to_evenodd}"
export FLOWED_RINGED_N_VEC="${FLOWED_RINGED_N_VEC:-1}"
export FLOWED_RINGED_N_ZN="${FLOWED_RINGED_N_ZN:-4}"
export FLOWED_RINGED_RAND_SEED="${FLOWED_RINGED_RAND_SEED:-0}"
export FLOWED_RINGED_MAXITER="${FLOWED_RINGED_MAXITER:-300}"

mkdir -p "$QUDA_RESOURCE_PATH" "$FLOWED_RINGED_DATA_DIR"

echo "Running flowed-quark ringed normalization"
echo "  FLOWED_RINGED_GAUGE_PATH=$FLOWED_RINGED_GAUGE_PATH"
echo "  FLOWED_RINGED_DATA_DIR=$FLOWED_RINGED_DATA_DIR"
echo "  config_num=$config_num"
echo "  FLOWED_RINGED_MPI_GEOMETRY=$FLOWED_RINGED_MPI_GEOMETRY"
echo "  FLOWED_RINGED_LAT_TAG=$FLOWED_RINGED_LAT_TAG"
echo "  FLOWED_RINGED_SPIN_COLOR_DILUTION=${FLOWED_RINGED_SPIN_COLOR_DILUTION:-none}"
echo "  QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH"
echo "  flow_batch_size=$flow_batch_size"

python -u "$script_dir/Pyquda_flowed_quark_ringed_norm.py" \
  --config_num "$config_num" \
  --mpi_geometry "$FLOWED_RINGED_MPI_GEOMETRY" \
  --flow-batch-size "$flow_batch_size"
