#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"
python3 -u "$script_dir/Pyquda_EMT_disconnected_finalize_quark_1pt.py" \
  --config_num "${EMT_1PT_CONFIG_NUM:-0}"
