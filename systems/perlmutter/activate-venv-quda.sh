#!/bin/bash
#
# Perlmutter PyQUDA environment helper.
#
# Source this file before running PyQUDA jobs on Perlmutter.
#

set -euo pipefail

software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m4559/xgao/software_gradientflow}"
venv_dir="${VENV_DIR:-$software_root/venv-quda-develop}"

module load cray-mpich-abi/9.0.1
source "$venv_dir/bin/activate"

if command -v nvidia-smi >/dev/null 2>&1; then
  # Login nodes may expose nvidia-smi without an attached GPU; do not abort here.
  nvidia-smi || true
fi

export HDF5_DIR="${HDF5_DIR:-/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3}"
export QUDA_PATH="${QUDA_PATH:-$software_root/quda-develop/install}"
export PYQUDA_ROOT="${PYQUDA_ROOT:-$software_root/Pyquda_Measurement}"
export PYTHONPATH="$PYQUDA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export LD_PRELOAD="${LD_PRELOAD:-}"
case ":$LD_PRELOAD:" in
  *":$HDF5_DIR/lib/libhdf5.so.310:"*) ;;
  *) LD_PRELOAD="/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0:$HDF5_DIR/lib/libhdf5.so.310:$HDF5_DIR/lib/libhdf5_hl.so.310${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac
export LD_PRELOAD
