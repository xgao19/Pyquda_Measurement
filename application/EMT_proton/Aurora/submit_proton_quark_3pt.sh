#!/bin/bash
#PBS -q debug-scaling
#PBS -N proton_EMT_q3pt
#PBS -l select=10:ngpus=5
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -k doe
#PBS -l place=scatter
#PBS -A StructNGB
#PBS -o log/proton_EMT_q3pt.o
#PBS -e log/proton_EMT_q3pt.e

set -euo pipefail

script_dir="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$script_dir"
mkdir -p log

echo ">>> PBS_NODEFILE content:"
cat "$PBS_NODEFILE"
nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
tasks=$(wc -l < "$PBS_NODEFILE")
echo "${nodes}n*${tasks}t"

module load oneapi/release/2025.2.0
module load ninja/1.12.1
module load python/3.10.14
module load cmake/3.31.8
module load mpich/opt/develop-git.6037a7a
module list

export QUDA_PATH="${QUDA_PATH:-/lus/flare/projects/StructNGB/xgao/software/build/quda/build}"
source "${AURORA_PYQUDA_VENV:-/lus/flare/projects/StructNGB/xgao/software/myenv/bin/activate}"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export PYTHONPATH="$script_dir/../../..:${PYTHONPATH:-}"
export LIBMPI="${LIBMPI:-/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/oneapi-2025.2.0/mpich-develop-git.6037a7a-cym6jg6/lib}"
export PYQ_LIB_PATH="${PYQ_LIB_PATH:-/lus/flare/projects/StructNGB/xgao/software/myenv/lib}"
export LD_LIBRARY_PATH="$LIBMPI:$PYQ_LIB_PATH:${LD_LIBRARY_PATH:-}"
export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"

export EMT_PROTON_STREAM="${EMT_PROTON_STREAM:-b}"
export EMT_PROTON_CONFIG_NUM="${EMT_PROTON_CONFIG_NUM:-220}"
export EMT_PROTON_MPI_GEOMETRY="${EMT_PROTON_MPI_GEOMETRY:-1.5.4.5}"

nranks="${EMT_PROTON_NRANKS:-100}"
ppn="${EMT_PROTON_PPN:-10}"

python --version
echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

/opt/cray/pals/1.8/bin/mpiexec -n "$nranks" -ppn "$ppn" \
  bash "$script_dir/run_proton_quark_3pt.sh"
