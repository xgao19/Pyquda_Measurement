# Initialize python and pyquda properly
module load ninja/1.12.1
module load python/3.10.14
module load cmake/3.31.8
module load oneapi/release/2025.2.0
module load mpich/opt/develop-git.6037a7a 
module list

export QUDA_PATH=/lus/flare/projects/StructNGB/xgao/software/build/quda/build
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

export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1


echo ">>> Running pyquda_nucleon_TMD_einsum2.py"
/opt/cray/pals/1.8/bin/mpiexec -n 32 -ppn 4 python3 pyquda_nucleon_TMD_einsum2.py --config_num 1050 --mpi_geometry 2.2.2.4 >log/nucleon_TMD_node8_n32_1050.o 2>log/nucleon_TMD_node8_n32_1050.e
# /opt/cray/pals/1.8/bin/mpiexec -n 32 -ppn 6 python3 pyquda_nucleon_TMD_einsum2.py --config_num 1050 --mpi_geometry 2.2.2.4 >log/nucleon_TMD_node6_n32_1050.o 2>log/nucleon_TMD_node6_n32_1050.e
# /opt/cray/pals/1.8/bin/mpiexec -n 64 -ppn 6 python3 pyquda_nucleon_TMD_einsum.py --config_num 1050 --mpi_geometry 2.2.4.4 >log/nucleon_TMD_node11_n64_1050.o 2>log/nucleon_TMD_node11_n64_1050.e
wait
