#!/bin/bash
#PBS -q debug-scaling
#PBS -N TMDWF_k0_b
#PBS -l select=20:ngpus=5
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -k doe
#PBS -l place=scatter
#PBS -A StructNGB
#PBS -o log/production_TMDWF.o
#PBS -e log/production_TMDWF.e

stream="b"
cfgmin=1
cfgmax=2
cfglist="/lus/flare/projects/StructNGB/xgao/ensembles/s8080b7596/gauge_fixed/list_cfg_${stream}"
main=pyquda_qTMDWF_k0.py

# switch to the submit directory
WORKDIR=/lus/flare/projects/StructNGB/xgao/run/l80c80a050/TMDWF_pyquda
cd $WORKDIR
logdir=/lus/flare/projects/StructNGB/xgao/run/l80c80a050/TMDWF_pyquda/log
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

# Settings for each run: 10 nodes, 5 MPI ranks per node spread evenly across cores
NUM_NODES_PER_MPI=10
NRANKS_PER_NODE=10

echo ">>> Running pyquda_qTMDWF_k0.py"
n_ranks=$((NUM_NODES_PER_MPI * NRANKS_PER_NODE))

#split -l ${NUM_NODES_PER_MPI} -d -a 2 hostfile/allnodes.uniq hostfile/qTMDWF_hostfile.
mkdir hostfile
split --lines=${NUM_NODES_PER_MPI} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE hostfile/qTMDWF_hostfile_stream${stream}_${cfgmin}to${cfgmax}.

k=1
for cfg in $(sed -n "${cfgmin},${cfgmax}p" ${cfglist}); do
  out=${logdir}/qTMDWF_stream${stream}_${cfg}.out
  err=${logdir}/qTMDWF_stream${stream}_${cfg}.err

  hf=$(printf "hostfile/qTMDWF_hostfile_stream${stream}_${cfgmin}to${cfgmax}.%02d" $k)

  /opt/cray/pals/1.8/bin/mpiexec -n ${n_ranks} -ppn ${NRANKS_PER_NODE} --hostfile ${hf} \
    python3 ${main} --stream ${stream} --config_num ${cfg} --mpi 1.5.4.5 \
    >${out} 2>${err} &

  k=$((k+1))
done
wait
