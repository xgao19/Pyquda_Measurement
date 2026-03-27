#!/bin/bash
#PBS -q debug-scaling
#PBS -N qTMD
#PBS -l select=10:ngpus=5
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -k doe
#PBS -l place=scatter
#PBS -A StructNGB
#PBS -o log/production_TMD.o
#PBS -e log/production_TMD.e

# switch to the submit directory
WORKDIR=/lus/flare/projects/StructNGB/xgao/run/l80c80a050/nucleon_TMD_pyquda
cd $WORKDIR
#rm .cache/*

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# Initialize python and pyquda properly
module load oneapi/release/2025.2.0
module load ninja/1.12.1
module load python/3.10.14
module load cmake/3.31.8
module load mpich/opt/develop-git.6037a7a 
module list

export QUDA_PATH=/lus/flare/projects/StructNGB/xgao/software/build/quda/build
source /lus/flare/projects/StructNGB/xgao/software/myenv/bin/activate

# check python version
python --version

echo ">>> ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
echo ">>> ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY}"

echo ">>> ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
echo ">>> ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"

# check python path
export PYTHONPATH="/lus/flare/projects/StructNGB/xgao/run/PyQUDA_Measurement"
echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

export LIBMPI=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/oneapi-2025.2.0/mpich-develop-git.6037a7a-cym6jg6/lib
export PYQ_LIB_PATH=/lus/flare/projects/StructNGB/xgao/software/myenv/lib
export SITE_PACKAGES=/lus/flare/projects/StructNGB/xgao/software/myenv/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LIBMPI:$PYQ_LIB_PATH:$LD_LIBRARY_PATH

mkdir .cache
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1

/opt/cray/pals/1.8/bin/mpiexec -n 100 -ppn 10 python3 pyquda_nucleon_TMD.py --stream b --config_num 220 --mpi_geometry 1.5.4.5 >log/production_TMD_1x2x5x5_ppn10_220_new.o 2>log/production_TMD_1x2x5x5_ppn10_220_new.e
wait
