#!/bin/bash
#
# Perlmutter PyQUDA batch template
#
# This is a starting point for CUDA / MPI runs on Perlmutter with the QUDA
# build configured by systems/perlmutter/configure-quda.
#

#SBATCH -A REPLACE_WITH_ACCOUNT
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J pyquda
#SBATCH -o log/%x.%j.out
#SBATCH -e log/%x.%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

set -euo pipefail

# ----------------------------
# Working directory
# ----------------------------

rundir="${SLURM_SUBMIT_DIR}"
cd "$rundir"
date

mkdir -p log

# ----------------------------
# Load modules
# ----------------------------

module purge
module load cpe/25.09
module load gcc-native/14
module load cudatoolkit/12.9
module load craype-accel-nvidia80
module load cray-mpich-abi/9.0.1
module load cray-hdf5-parallel/1.14.3.7
module list

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# ----------------------------
# Paths / environment
# ----------------------------

software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m4559/xgao/software_gradientflow}"
repo_root="${REPO_ROOT:-$software_root/Pyquda_Measurement}"
quda_prefix="${QUDA_PATH:-$software_root/quda-develop/install}"
venv_dir="${VENV_DIR:-$software_root/venv-quda-develop}"
py_driver="${PYQUDA_DRIVER:-pyquda_main.py}"
py_args="${PYQUDA_ARGS:---mpi_geometry 2.2.2.2}"

export QUDA_PATH="$quda_prefix"
export HDF5_DIR="${HDF5_DIR:-/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-1}"
export QUDA_RESOURCE_PATH="${QUDA_RESOURCE_PATH:-$rundir/.quda-cache}"
export QUDA_PROFILE_OUTPUT="${QUDA_PROFILE_OUTPUT:-profile.tsv}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$rundir/.cupy-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR"

# Prefer the repo-local helper if present, otherwise fall back to the shared one.
if [ -f "$repo_root/systems/perlmutter/activate-venv-quda.sh" ]; then
  source "$repo_root/systems/perlmutter/activate-venv-quda.sh"
elif [ -f "$software_root/activate-venv-quda.sh" ]; then
  source "$software_root/activate-venv-quda.sh"
else
  source "$venv_dir/bin/activate"
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
  export LD_PRELOAD="$HDF5_DIR/lib/libhdf5.so.310:$HDF5_DIR/lib/libhdf5_hl.so.310${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export LD_LIBRARY_PATH="$QUDA_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

echo "QUDA_PATH: $QUDA_PATH"
echo "QUDA_RESOURCE_PATH: $QUDA_RESOURCE_PATH"
echo "CUPY_CACHE_DIR: $CUPY_CACHE_DIR"
echo "HDF5_DIR: $HDF5_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "python: $(which python3)"
python3 --version

# ----------------------------
# Run
# ----------------------------

# Replace pyquda_main.py / pyquda_prop.py with your actual driver.
read -r -a py_args_array <<< "$py_args"
srun --mpi=cray_shasta -n "${SLURM_NTASKS:-4}" python3 "$py_driver" "${py_args_array[@]}"
