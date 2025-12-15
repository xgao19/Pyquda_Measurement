#!/bin/bash
#PBS -q debug-scaling
#PBS -N qTMD
#PBS -l select=16
#PBS -l walltime=0:30:00
#PBS -l filesystems=flare
#PBS -k doe
#PBS -l place=scatter
#PBS -A StructNGB
#PBS -o log/TEST.o
#PBS -e log/TEST.e

# switch to the submit directory
WORKDIR=/lus/flare/projects/StructNGB/xgao/run/l64c64a076/full_TMD
cd $WORKDIR

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# Initialize python and pyquda properly
module load ninja/1.12.1
module load python/3.10.14
module load cmake/3.31.8

export QUDA_PATH=/lus/flare/projects/StructNGB/xgao/software/install/quda
source /lus/flare/projects/StructNGB/xgao/software/myenv/bin/activate

# check python version
python --version

# check python path
export PYTHONPATH="/home/gaox/latwork/PyQUDA_Measurement"
echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

export LIBMPI=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/oneapi-2025.2.0/mpich-develop-git.6037a7a-cym6jg6/lib
export PYQ_LIB_PATH=/lus/flare/projects/StructNGB/xgao/software/myenv/lib
export SITE_PACKAGES=/lus/flare/projects/StructNGB/xgao/software/myenv/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LIBMPI:$PYQ_LIB_PATH:$LD_LIBRARY_PATH


echo ">>> Running pyquda_main.py"
mpiexec -np 128 -ppn 8 -envall ./gpu_tile_compact.sh python3 pyquda_main.py --config_num 1050 --mpi_geometry 2.4.4.4
