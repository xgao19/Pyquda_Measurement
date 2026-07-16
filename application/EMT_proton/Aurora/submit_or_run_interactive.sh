#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --config_num CFG [--mg-block X.Y.Z.T[;...]]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh

export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export EMT_PROTON_NRANKS="${EMT_PROTON_NRANKS:-2}"
export EMT_PROTON_MPI_GEOMETRY="${EMT_PROTON_MPI_GEOMETRY:-1.1.1.2}"

mkdir -p log

echo "Running test-gauge EMT_proton smoke"
echo "  host=$(hostname)"
echo "  ranks=$EMT_PROTON_NRANKS"
echo "  mpi_geometry=$EMT_PROTON_MPI_GEOMETRY"
echo "  python=$(which python)"
echo "  QUDA_PATH=$QUDA_PATH"

/opt/cray/pals/1.8/bin/mpiexec -n "$EMT_PROTON_NRANKS" -envall \
  bash "$script_dir/run_proton_quark_3pt.sh" \
  "$@" \
  > "log/smoke_n${EMT_PROTON_NRANKS}.o" \
  2> "log/smoke_n${EMT_PROTON_NRANKS}.e"

echo "Smoke output log: $script_dir/log/smoke_n${EMT_PROTON_NRANKS}.o"
echo "Smoke error log:  $script_dir/log/smoke_n${EMT_PROTON_NRANKS}.e"
