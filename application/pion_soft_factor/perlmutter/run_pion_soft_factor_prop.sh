#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"

source "$software_root/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export PYTHONPATH="$software_root/Pyquda_Measurement:$software_root/PyQUDA:${PYTHONPATH:-}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/pion_soft_factor_prop}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/pion_soft_factor_prop}"

export PION_SOFT_DATA_DIR="${PION_SOFT_DATA_DIR:-$script_dir/data}"
export PION_SOFT_GAUGE_PATH="${PION_SOFT_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export PION_SOFT_CONFIG_NUM="${PION_SOFT_CONFIG_NUM:-0}"
export PION_SOFT_MPI_GEOMETRY="${PION_SOFT_MPI_GEOMETRY:-1.1.1.1}"
export PION_SOFT_LAT_TAG="${PION_SOFT_LAT_TAG:-S8T32}"
export PION_SOFT_SM_TAG="${PION_SOFT_SM_TAG:-1HYP_wall}"
export PION_SOFT_T_START="${PION_SOFT_T_START:-0}"
export PION_SOFT_T_COUNT="${PION_SOFT_T_COUNT:-0}"
export PION_SOFT_QUARK_MOM_Z="${PION_SOFT_QUARK_MOM_Z:-0}"
export PION_SOFT_MASS="${PION_SOFT_MASS:-0.236}"
export PION_SOFT_CSW="${PION_SOFT_CSW:-1.0372}"
export PION_SOFT_TOL="${PION_SOFT_TOL:-1e-15}"
export PION_SOFT_MAXITER="${PION_SOFT_MAXITER:-300}"

echo "Running pion soft-factor wall-propagator generation"
python3 -u "$script_dir/Pyquda_pion_soft_factor_prop.py" "$@"
