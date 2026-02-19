#!/bin/bash
# Begin LSF Directives
#SBATCH -A nph174
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -J TEST
#SBATCH -o log/N.%J
#SBATCH -e log/N.%J
#SBATCH -N 4
#SBATCH -n 32
#SBATCH --distribution=cyclic

# Set working directory
rundir=$SLURM_SUBMIT_DIR
cd $rundir
date

# === setup cupy cache ===
export CUPY_CACHE_DIR=/ccs/home/xiangg/latwork/l64c64a076/qTMDWF_pyquda/cupy_cache
mkdir -p $CUPY_CACHE_DIR
echo $TMPDIR
echo $CUPY_CACHE_DIR

# Load environment
module purge
module load PrgEnv-gnu craype-accel-amd-gfx90a amd-mixed rocm/6.2.4 cray-mpich craype-x86-trento cray-fftw
module load cray-python/3.9.13.1 cray-hdf5-parallel/1.14.3.3 miniforge3/23.11.0-0
module list

export ROCM_HOME=/opt/rocm-6.2.4
export HIP_PATH=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64/gcc/x86_64-suse-linux/14/../../../../lib64/:/lustre/orion/nph174/proj-shared/xgao/software/install/libxcrypt/lib:$LD_LIBRARY_PATH

source /lustre/orion/nph158/world-shared/software/py_venv/bin/activate
# source /lustre/orion/nph158/proj-shared/jinchen/software/lat/bin/activate

export LD_PRELOAD="/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so:$LD_PRELOAD"
export LD_LIBRARY_PATH=/lustre/orion/nph158/world-shared/software/build/quda/build/lib:$LD_LIBRARY_PATH
export QUDA_PATH=/lustre/orion/nph158/world-shared/software/build/quda/build
export PYTHONPATH=$PYTHONPATH:/ccs/home/xiangg/latwork/Pyquda_Measurement/pyquda_measurement_utils:/ccs/home/xiangg/latwork/Pyquda_Measurement
which python

export MPICH_GPU_SUPPORT_ENABLED=1

# Output LD_LIBRARY_PATH
echo -e "\n>>> Output LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH

# QUDA global environment
QUDA_RPATH=${rundir}/.cache
mkdir -p ${QUDA_RPATH}

#export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=32
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=${QUDA_RPATH}
export QUDA_PROFILE_OUTPUT_BASE=${QUDA_RPATH}/profile_
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0

cat << EOF > scripts/select_gpu_Ncfg
#!/bin/bash
export GPU_MAP=(0 1 2 3 7 6 5 4)
export NUMA_MAP=(3 3 1 1 2 2 0 0)
export GPU=\${GPU_MAP[\$SLURM_LOCALID]}
export NUMA=\${NUMA_MAP[\$SLURM_LOCALID]}
export HIP_VISIBLE_DEVICES=\$GPU
unset ROCR_VISIBLE_DEVICES
echo RANK \$SLURM_LOCALID using GPU \$GPU
exec numactl -m \$NUMA -N \$NUMA \$*
EOF

chmod +x scripts/select_gpu_Ncfg
sleep 10s
offset=0

srun -u --nodes 4 -n 32 -r $offset scripts/./select_gpu_Ncfg python pyquda_qTMDWF_einsum2.py  --config_num 1050 --mpi_geometry 2.2.2.4 >/ccs/home/xiangg/latwork/l64c64a076/qTMDWF_pyquda/log/qTMDWF_einsum_1050.out 2>/ccs/home/xiangg/latwork/l64c64a076/qTMDWF_pyquda/log/qTMDWF_einsum_1050.err &

offset=$((offset + 4))
sleep 0.5s
wait
date
