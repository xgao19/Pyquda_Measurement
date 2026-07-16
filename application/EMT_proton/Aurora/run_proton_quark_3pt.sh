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
measurement_root="$(cd "$script_dir/../../.." && pwd)"

cd "$script_dir"
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh

export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.cache/proton_quark_3pt}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export EMT_PROTON_DATA_DIR="${EMT_PROTON_DATA_DIR:-$script_dir/data}"
export EMT_PROTON_GAUGE_PATH="${EMT_PROTON_GAUGE_PATH:-$measurement_root/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export EMT_PROTON_LAT_TAG="${EMT_PROTON_LAT_TAG:-S8T32}"
export EMT_PROTON_MPI_GEOMETRY="${EMT_PROTON_MPI_GEOMETRY:-1.1.1.2}"
export EMT_PROTON_QMAX="${EMT_PROTON_QMAX:-0}"
export EMT_PROTON_T_SEPS="${EMT_PROTON_T_SEPS:-2}"
export EMT_PROTON_FLOW_STEPS="${EMT_PROTON_FLOW_STEPS:-1}"
export EMT_PROTON_WIDTH="${EMT_PROTON_WIDTH:-1.0}"
export EMT_PROTON_GAUSS_SMEAR="${EMT_PROTON_GAUSS_SMEAR:-0}"
export EMT_PROTON_SRC_POS="${EMT_PROTON_SRC_POS:-0.0.0}"
export EMT_PROTON_SRC_T="${EMT_PROTON_SRC_T:-0}"
export EMT_PROTON_MAXITER="${EMT_PROTON_MAXITER:-300}"

mkdir -p "$QUDA_RESOURCE_PATH" "$EMT_PROTON_DATA_DIR"

echo "Running proton quark EMT 3pt"
echo "  EMT_PROTON_GAUGE_PATH=$EMT_PROTON_GAUGE_PATH"
echo "  EMT_PROTON_DATA_DIR=$EMT_PROTON_DATA_DIR"
echo "  config_num=$config_num"
echo "  EMT_PROTON_MPI_GEOMETRY=$EMT_PROTON_MPI_GEOMETRY"
echo "  EMT_PROTON_LAT_TAG=$EMT_PROTON_LAT_TAG"
echo "  EMT_PROTON_GAUSS_SMEAR=$EMT_PROTON_GAUSS_SMEAR"
echo "  QUDA_RESOURCE_PATH=$QUDA_RESOURCE_PATH"

python -u "$script_dir/Pyquda_EMT_proton_quark_3pt.py" \
  --config_num "$config_num" \
  --mpi_geometry "$EMT_PROTON_MPI_GEOMETRY" \
  --mg-block "$mg_block"
