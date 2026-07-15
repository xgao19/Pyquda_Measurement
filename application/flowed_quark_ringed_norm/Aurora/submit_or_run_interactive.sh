#!/bin/bash
set -euo pipefail

config_num=""
flow_batch_size="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num)
      [[ -z "$config_num" && $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        echo "--config_num requires one non-negative integer" >&2; exit 2;
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
cd "$script_dir"

source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh

export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export FLOWED_RINGED_NRANKS="${FLOWED_RINGED_NRANKS:-2}"
export FLOWED_RINGED_MPI_GEOMETRY="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.2}"

mkdir -p log

echo "Running test-gauge flowed-quark ringed-normalization smoke"
echo "  host=$(hostname)"
echo "  ranks=$FLOWED_RINGED_NRANKS"
echo "  mpi_geometry=$FLOWED_RINGED_MPI_GEOMETRY"
echo "  python=$(which python)"
echo "  QUDA_PATH=$QUDA_PATH"
echo "  flow_batch_size=$flow_batch_size"

/opt/cray/pals/1.8/bin/mpiexec -n "$FLOWED_RINGED_NRANKS" -envall \
  bash "$script_dir/run_flowed_quark_ringed_norm.sh" \
  --config_num "$config_num" --flow-batch-size "$flow_batch_size" \
  > "log/smoke_n${FLOWED_RINGED_NRANKS}.o" \
  2> "log/smoke_n${FLOWED_RINGED_NRANKS}.e"

echo "Smoke output log: $script_dir/log/smoke_n${FLOWED_RINGED_NRANKS}.o"
echo "Smoke error log:  $script_dir/log/smoke_n${FLOWED_RINGED_NRANKS}.e"
