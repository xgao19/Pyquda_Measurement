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

data_dir="${FLOWED_RINGED_DATA_DIR:-$script_dir/benchmark/login_smoke/data}"
mpi_geometry="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.1}"
n_vec="${FLOWED_RINGED_N_VEC:-1}"
block_interval_solves="${FLOWED_RINGED_BLOCK_INTERVAL_SOLVES:-$n_vec}"
quda_cache="${QUDA_RESOURCE_PATH:-$script_dir/benchmark/login_smoke/cache}"
cupy_cache="${CUPY_CACHE_DIR:-$script_dir/benchmark/login_smoke/cupy-cache}"

mkdir -p "$data_dir" "$quda_cache" "$cupy_cache"

echo "Running login-node flowed-quark ringed-normalization smoke"
echo "  host=$(hostname)"
echo "  FLOWED_RINGED_MPI_GEOMETRY=$mpi_geometry"
echo "  FLOWED_RINGED_N_VEC=$n_vec"
echo "  FLOWED_RINGED_BLOCK_INTERVAL_SOLVES=$block_interval_solves"
echo "  FLOWED_RINGED_DATA_DIR=$data_dir"
echo "  flow_batch_size=$flow_batch_size"

env \
  FLOWED_RINGED_DATA_DIR="$data_dir" \
  FLOWED_RINGED_SM_TAG="${FLOWED_RINGED_SM_TAG:-S8T8_login_smoke}" \
  FLOWED_RINGED_MPI_GEOMETRY="$mpi_geometry" \
  FLOWED_RINGED_N_VEC="$n_vec" \
  FLOWED_RINGED_BLOCK_INTERVAL_SOLVES="$block_interval_solves" \
  QUDA_RESOURCE_PATH="$quda_cache" \
  CUPY_CACHE_DIR="$cupy_cache" \
  bash "$script_dir/run_flowed_quark_ringed_norm.sh" \
    --config_num "$config_num" --flow-batch-size "$flow_batch_size"
