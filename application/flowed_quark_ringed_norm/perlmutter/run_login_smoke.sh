#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 || "$1" != "--config_num" || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 --config_num CFG" >&2
  exit 2
fi
config_num="$2"

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

env \
  FLOWED_RINGED_DATA_DIR="$data_dir" \
  FLOWED_RINGED_SM_TAG="${FLOWED_RINGED_SM_TAG:-S8T8_login_smoke}" \
  FLOWED_RINGED_MPI_GEOMETRY="$mpi_geometry" \
  FLOWED_RINGED_N_VEC="$n_vec" \
  FLOWED_RINGED_BLOCK_INTERVAL_SOLVES="$block_interval_solves" \
  QUDA_RESOURCE_PATH="$quda_cache" \
  CUPY_CACHE_DIR="$cupy_cache" \
  bash "$script_dir/run_flowed_quark_ringed_norm.sh" --config_num "$config_num"
