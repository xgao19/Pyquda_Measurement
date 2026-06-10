#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"
source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export EMT_1PT_DATA_DIR="${EMT_1PT_DATA_DIR:-$script_dir/data}"
export EMT_1PT_CONFIG_NUM="${EMT_1PT_CONFIG_NUM:-0}"
export EMT_1PT_LAT_TAG="${EMT_1PT_LAT_TAG:-S8T32}"
export EMT_1PT_SRC_POS="${EMT_1PT_SRC_POS:-0.0.0}"
export EMT_1PT_SRC_T="${EMT_1PT_SRC_T:-0}"
export EMT_DISC_INTERPOLATOR="${EMT_DISC_INTERPOLATOR:-5}"
export EMT_DISC_T_SEPS="${EMT_DISC_T_SEPS:-2}"
export EMT_DISC_C2_GAMMA="${EMT_DISC_C2_GAMMA:-$EMT_DISC_INTERPOLATOR}"
export EMT_DISC_C2_MOMENTUM="${EMT_DISC_C2_MOMENTUM:-PX0PY0PZ0}"

mkdir -p "$EMT_1PT_DATA_DIR"

echo "Building disconnected EMT 3pt diagnostic"
python3 -u "$script_dir/Pyquda_EMT_disconnected_build_3pt.py" \
  --config_num "$EMT_1PT_CONFIG_NUM" \
  --interpolator "$EMT_DISC_INTERPOLATOR"
