#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_num=""
mg_block="8.8.4.4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_num)
      config_num="${2:-}"
      shift 2
      ;;
    --mg-block)
      mg_block="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ ! "$config_num" =~ ^[0-9]+$ || -z "$mg_block" ]]; then
  echo "Usage: $0 --config_num CFG [--mg-block X.Y.Z.T[;...]|none]" >&2
  exit 2
fi

software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/nucleon_TMD}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/nucleon_TMD}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export NUCLEON_TMD_DATA_DIR="${NUCLEON_TMD_DATA_DIR:-$script_dir/data}"
export NUCLEON_TMD_GAUGE_PATH="${NUCLEON_TMD_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export NUCLEON_TMD_MPI_GEOMETRY="${NUCLEON_TMD_MPI_GEOMETRY:-1.1.1.1}"
export NUCLEON_TMD_NUM_SRC="${NUCLEON_TMD_NUM_SRC:-1}"
export NUCLEON_TMD_QMAX="${NUCLEON_TMD_QMAX:-0}"
export NUCLEON_TMD_BZ="${NUCLEON_TMD_BZ:-2}"
export NUCLEON_TMD_BT="${NUCLEON_TMD_BT:-1}"
export NUCLEON_TMD_ETA="${NUCLEON_TMD_ETA:-1}"
export NUCLEON_TMD_T_INSERT="${NUCLEON_TMD_T_INSERT:-2}"
export NUCLEON_TMD_WIDTH="${NUCLEON_TMD_WIDTH:-1.0}"
export NUCLEON_TMD_INTERPOLATOR="${NUCLEON_TMD_INTERPOLATOR:-5}"
export NUCLEON_TMD_POL="${NUCLEON_TMD_POL:-PpUnpol}"
export NUCLEON_TMD_RUN_CG_QTMD="${NUCLEON_TMD_RUN_CG_QTMD:-1}"
export NUCLEON_TMD_RUN_GI_QTMD="${NUCLEON_TMD_RUN_GI_QTMD:-1}"
export NUCLEON_TMD_RUN_PDF="${NUCLEON_TMD_RUN_PDF:-1}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$NUCLEON_TMD_DATA_DIR"

echo "Running nucleon qTMD"
echo "  NUCLEON_TMD_GAUGE_PATH=$NUCLEON_TMD_GAUGE_PATH"
echo "  NUCLEON_TMD_DATA_DIR=$NUCLEON_TMD_DATA_DIR"
echo "  config_num=$config_num"
echo "  mg_block=$mg_block"
echo "  NUCLEON_TMD_MPI_GEOMETRY=$NUCLEON_TMD_MPI_GEOMETRY"
echo "  NUCLEON_TMD_NUM_SRC=$NUCLEON_TMD_NUM_SRC"
echo "  NUCLEON_TMD_QMAX=$NUCLEON_TMD_QMAX"
echo "  NUCLEON_TMD_BZ=$NUCLEON_TMD_BZ"
echo "  NUCLEON_TMD_BT=$NUCLEON_TMD_BT"
echo "  NUCLEON_TMD_ETA=$NUCLEON_TMD_ETA"
echo "  NUCLEON_TMD_T_INSERT=$NUCLEON_TMD_T_INSERT"
echo "  NUCLEON_TMD_RUN_CG_QTMD=$NUCLEON_TMD_RUN_CG_QTMD"
echo "  NUCLEON_TMD_RUN_GI_QTMD=$NUCLEON_TMD_RUN_GI_QTMD"
echo "  NUCLEON_TMD_RUN_PDF=$NUCLEON_TMD_RUN_PDF"

python3 -u "$script_dir/Pyquda_nucleon_TMD.py" \
  --config_num "$config_num" \
  --mg-block "$mg_block" \
  --mpi_geometry "$NUCLEON_TMD_MPI_GEOMETRY" \
  --gauge_path "$NUCLEON_TMD_GAUGE_PATH" \
  --data_dir "$NUCLEON_TMD_DATA_DIR" \
  --num_src "$NUCLEON_TMD_NUM_SRC" \
  --qmax "$NUCLEON_TMD_QMAX" \
  --b_z "$NUCLEON_TMD_BZ" \
  --b_T "$NUCLEON_TMD_BT" \
  --eta "$NUCLEON_TMD_ETA" \
  --t_insert "$NUCLEON_TMD_T_INSERT" \
  --width "$NUCLEON_TMD_WIDTH" \
  --interpolator "$NUCLEON_TMD_INTERPOLATOR" \
  --pol "$NUCLEON_TMD_POL" \
  --run_cg_qtmd "$NUCLEON_TMD_RUN_CG_QTMD" \
  --run_gi_qtmd "$NUCLEON_TMD_RUN_GI_QTMD" \
  --run_pdf "$NUCLEON_TMD_RUN_PDF"
