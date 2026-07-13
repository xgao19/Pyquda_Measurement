#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/pion_TMD}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/pion_TMD}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export PION_TMD_DATA_DIR="${PION_TMD_DATA_DIR:-$script_dir/data}"
export PION_TMD_GAUGE_PATH="${PION_TMD_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export PION_TMD_CONFIG_NUM="${PION_TMD_CONFIG_NUM:-0}"
export PION_TMD_MPI_GEOMETRY="${PION_TMD_MPI_GEOMETRY:-1.1.1.1}"
export PION_TMD_NUM_SRC="${PION_TMD_NUM_SRC:-1}"
export PION_TMD_QMAX="${PION_TMD_QMAX:-1}"
export PION_TMD_BZ="${PION_TMD_BZ:-2}"
export PION_TMD_BT="${PION_TMD_BT:-1}"
export PION_TMD_ETA="${PION_TMD_ETA:-1}"
export PION_TMD_T_INSERT="${PION_TMD_T_INSERT:-2}"
export PION_TMD_WIDTH="${PION_TMD_WIDTH:-1.0}"
export PION_TMD_RUN_CG_QTMD="${PION_TMD_RUN_CG_QTMD:-1}"
export PION_TMD_RUN_GI_QTMD="${PION_TMD_RUN_GI_QTMD:-1}"
export PION_TMD_RUN_PDF="${PION_TMD_RUN_PDF:-1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$PION_TMD_DATA_DIR"

echo "Running pion qTMD"
echo "  PION_TMD_GAUGE_PATH=$PION_TMD_GAUGE_PATH"
echo "  PION_TMD_DATA_DIR=$PION_TMD_DATA_DIR"
echo "  PION_TMD_CONFIG_NUM=$PION_TMD_CONFIG_NUM"
echo "  PION_TMD_MPI_GEOMETRY=$PION_TMD_MPI_GEOMETRY"
echo "  PION_TMD_NUM_SRC=$PION_TMD_NUM_SRC"
echo "  PION_TMD_QMAX=$PION_TMD_QMAX"
echo "  PION_TMD_BZ=$PION_TMD_BZ"
echo "  PION_TMD_BT=$PION_TMD_BT"
echo "  PION_TMD_ETA=$PION_TMD_ETA"
echo "  PION_TMD_T_INSERT=$PION_TMD_T_INSERT"
echo "  PION_TMD_RUN_CG_QTMD=$PION_TMD_RUN_CG_QTMD"
echo "  PION_TMD_RUN_GI_QTMD=$PION_TMD_RUN_GI_QTMD"
echo "  PION_TMD_RUN_PDF=$PION_TMD_RUN_PDF"

python3 -u "$script_dir/Pyquda_pion_TMD.py" \
  --config_num "$PION_TMD_CONFIG_NUM" \
  --mpi_geometry "$PION_TMD_MPI_GEOMETRY" \
  --gauge_path "$PION_TMD_GAUGE_PATH" \
  --data_dir "$PION_TMD_DATA_DIR" \
  --num_src "$PION_TMD_NUM_SRC" \
  --qmax "$PION_TMD_QMAX" \
  --b_z "$PION_TMD_BZ" \
  --b_T "$PION_TMD_BT" \
  --eta "$PION_TMD_ETA" \
  --t_insert "$PION_TMD_T_INSERT" \
  --width "$PION_TMD_WIDTH" \
  --run_cg_qtmd "$PION_TMD_RUN_CG_QTMD" \
  --run_gi_qtmd "$PION_TMD_RUN_GI_QTMD" \
  --run_pdf "$PION_TMD_RUN_PDF"
