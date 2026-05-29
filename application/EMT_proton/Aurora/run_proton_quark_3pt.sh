#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
cd "$script_dir"

export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export EMT_PROTON_STREAM="${EMT_PROTON_STREAM:-b}"
export EMT_PROTON_CONFIG_NUM="${EMT_PROTON_CONFIG_NUM:-220}"
export EMT_PROTON_MPI_GEOMETRY="${EMT_PROTON_MPI_GEOMETRY:-1.5.4.5}"
export EMT_PROTON_DATA_DIR="${EMT_PROTON_DATA_DIR:-/lus/flare/projects/StructNGB/xgao/run/l80c80a050/proton_EMT_pyquda/data_${EMT_PROTON_STREAM}}"
export EMT_PROTON_QMAX="${EMT_PROTON_QMAX:-0}"
export EMT_PROTON_T_SEPS="${EMT_PROTON_T_SEPS:-9}"
export EMT_PROTON_FLOW_STEPS="${EMT_PROTON_FLOW_STEPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.cache}"

mkdir -p "$EMT_PROTON_DATA_DIR" "$QUDA_RESOURCE_PATH"

echo "Running Aurora proton EMT connected quark 3pt"
echo "  EMT_PROTON_STREAM=$EMT_PROTON_STREAM"
echo "  EMT_PROTON_CONFIG_NUM=$EMT_PROTON_CONFIG_NUM"
echo "  EMT_PROTON_MPI_GEOMETRY=$EMT_PROTON_MPI_GEOMETRY"
echo "  EMT_PROTON_DATA_DIR=$EMT_PROTON_DATA_DIR"

python3 -u "$script_dir/Pyquda_EMT_proton_quark_3pt.py" \
  --stream "$EMT_PROTON_STREAM" \
  --config_num "$EMT_PROTON_CONFIG_NUM" \
  --mpi_geometry "$EMT_PROTON_MPI_GEOMETRY"
