#!/bin/bash
set -euo pipefail

t_separations="2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --t_separations)
      [[ $# -ge 2 ]] || { echo "Missing value for --t_separations" >&2; exit 2; }
      t_separations="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"

source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$script_dir/.quda-cache/pion_EMFF}"
export QUDA_PROFILE_OUTPUT_BASE="${QUDA_PROFILE_OUTPUT_BASE:-profile_}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$script_dir/.cupy-cache/pion_EMFF}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export PION_EMFF_DATA_DIR="${PION_EMFF_DATA_DIR:-$script_dir/data}"
export PION_EMFF_GAUGE_PATH="${PION_EMFF_GAUGE_PATH:-$software_root/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0}"
export PION_EMFF_CONFIG_NUM="${PION_EMFF_CONFIG_NUM:-0}"
export PION_EMFF_MPI_GEOMETRY="${PION_EMFF_MPI_GEOMETRY:-1.1.1.1}"
export PION_EMFF_NUM_SRC="${PION_EMFF_NUM_SRC:-1}"
export PION_EMFF_QMAX="${PION_EMFF_QMAX:-1}"
export PION_EMFF_PF="${PION_EMFF_PF:-0.0.0}"
export PION_EMFF_WIDTH="${PION_EMFF_WIDTH:-1.0}"
export PION_EMFF_POS_BOOST_SRC="${PION_EMFF_POS_BOOST_SRC:-${PION_EMFF_POS_BOOST:-0.0.0}}"
export PION_EMFF_POS_BOOST_SINK="${PION_EMFF_POS_BOOST_SINK:-${PION_EMFF_POS_BOOST:-0.0.0}}"
export PION_EMFF_NEG_BOOST_SRC="${PION_EMFF_NEG_BOOST_SRC:-${PION_EMFF_NEG_BOOST:-0.0.0}}"
export PION_EMFF_NEG_BOOST_SINK="${PION_EMFF_NEG_BOOST_SINK:-${PION_EMFF_NEG_BOOST:-0.0.0}}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR" "$PION_EMFF_DATA_DIR"

echo "Running pion EMFF"
echo "  PION_EMFF_GAUGE_PATH=$PION_EMFF_GAUGE_PATH"
echo "  PION_EMFF_DATA_DIR=$PION_EMFF_DATA_DIR"
echo "  PION_EMFF_CONFIG_NUM=$PION_EMFF_CONFIG_NUM"
echo "  PION_EMFF_MPI_GEOMETRY=$PION_EMFF_MPI_GEOMETRY"
echo "  PION_EMFF_NUM_SRC=$PION_EMFF_NUM_SRC"
echo "  PION_EMFF_QMAX=$PION_EMFF_QMAX"
echo "  PION_EMFF_PF=$PION_EMFF_PF"
echo "  t_separations=$t_separations"

python3 -u "$script_dir/Pyquda_pion_EMFF.py" \
  --config_num "$PION_EMFF_CONFIG_NUM" \
  --mpi_geometry "$PION_EMFF_MPI_GEOMETRY" \
  --gauge_path "$PION_EMFF_GAUGE_PATH" \
  --data_dir "$PION_EMFF_DATA_DIR" \
  --num_src "$PION_EMFF_NUM_SRC" \
  --qmax "$PION_EMFF_QMAX" \
  --pf "$PION_EMFF_PF" \
  --t_separations "$t_separations" \
  --width "$PION_EMFF_WIDTH" \
  --pos_boost_src "$PION_EMFF_POS_BOOST_SRC" \
  --pos_boost_sink "$PION_EMFF_POS_BOOST_SINK" \
  --neg_boost_src "$PION_EMFF_NEG_BOOST_SRC" \
  --neg_boost_sink "$PION_EMFF_NEG_BOOST_SINK"
