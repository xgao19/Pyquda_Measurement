#!/bin/bash
#
set -euo pipefail

software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
repo_root="${REPO_ROOT:-$software_root/Pyquda_Measurement}"
quda_prefix="${QUDA_PATH:-$software_root/quda/install}"
venv_dir="${VENV_DIR:-$software_root/venv}"
gauge_path="${GAUGE_PATH:-$repo_root/test_gauge/S8T8_wilson_b6.0}"
grid_size="${GRID_SIZE:-1,1,1,1}"
latt_size="${LATT_SIZE:-8,8,8,8}"
flow_steps="${FLOW_STEPS:-1}"
flow_epsilon="${FLOW_EPSILON:-0.01}"

export GAUGE_PATH="$gauge_path"
export GRID_SIZE="$grid_size"
export LATT_SIZE="$latt_size"
export FLOW_STEPS="$flow_steps"
export FLOW_EPSILON="$flow_epsilon"

cd "$repo_root"

echo "[check-gradient-flow] host: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

export QUDA_PATH="$quda_prefix"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$repo_root/.quda-cache}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$repo_root/.cupy-cache}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR"

if [ -f "$repo_root/systems/perlmutter/activate-venv-quda.sh" ]; then
  # shellcheck disable=SC1090
  source "$repo_root/systems/perlmutter/activate-venv-quda.sh"
elif [ -f "$software_root/activate-venv-quda.sh" ]; then
  # shellcheck disable=SC1090
  source "$software_root/activate-venv-quda.sh"
else
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
fi

python3 -u "$repo_root/systems/perlmutter/check-gradient-flow.py"
