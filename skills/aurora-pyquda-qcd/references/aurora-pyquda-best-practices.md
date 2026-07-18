# Aurora PyQUDA Best Practices

This reference captures the validated Aurora route for this repository.  Keep
`systems/Aurora/README.md` as the human-facing source of truth and update this
skill reference when operational practices change.

## Validated Layout

```text
Measurement repo: /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement
PyQUDA source:    /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
Python venv:      /lus/flare/projects/StructNGB/xgao/software_gradientflow/pyquda_env
QUDA install:     /lus/flare/projects/StructNGB/xgao/software_260507/install/quda
Activation:       /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

Reuse the existing QUDA install unless the user explicitly needs a different
QUDA build.

## Module And Activation Sequence

```bash
module load gcc/13.4.0
module load oneapi/release/2025.3.1
module load mpich/opt/5.0.0.aurora_test.3c70a61
module load libfabric/1.22.0
module load cray-pals/1.8.0
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

The activation script should provide `QUDA_PATH`, `PYQUDA_ROOT`, `PYTHONPATH`,
`LD_LIBRARY_PATH`, and `MPICH_GPU_SUPPORT_ENABLED=1`.

## PyQUDA Develop Environment

Create the venv with Aurora's loaded Python 3.12, not `/usr/bin/python3`.

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==2.4.4 Cython==3.2.4 opt-einsum==3.4.0 packaging pkgconfig
python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp==0.19.0
python -m pip install mpi4py==4.1.1
CC=mpicc HDF5_MPI=ON \
HDF5_DIR=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/hdf5-1.14.6-ehlefog \
  python -m pip install --no-binary=h5py h5py==3.15.1
python -m pip install -e /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
python -m pip install -e /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA/pyquda_utils
```

PyQUDA source should be cloned from `https://github.com/CLQCD/PyQUDA.git` and
checked out to `develop`.

## Required Verification

Run these after activation:

```bash
python -c "import h5py; print(h5py.get_config().mpi)"
python -c "import pyquda, pyquda_utils; print(pyquda.__file__)"
```

Expected:

```text
h5py.get_config().mpi == True
backend="dpnp", backend_target="sycl"
```

For MPI checks, use a compute allocation:

```bash
/opt/cray/pals/1.8/bin/mpiexec -n 2 -envall \
  python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

## GPU Runtime Defaults

For the current dpnp route, prefer:

```bash
unset ZE_FLAT_DEVICE_HIERARCHY
unset ZE_AFFINITY_MASK
unset ONEAPI_DEVICE_SELECTOR
unset ONEAPI_DEVICE_FILTER
export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
```

Use a fresh cache when enabling tuning:

```bash
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=/path/to/fresh/cache
```

## PALS Launch Pattern

Request an interactive allocation, SSH to one allocated compute node, then use
PALS:

```bash
qsub -I -q debug-scaling -A StructNGB -N pyquda_test \
  -l select=6:ngpus=6 -l walltime=01:00:00 -l filesystems=flare -l place=scatter

ssh <allocated-node>
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement
/opt/cray/pals/1.8/bin/mpiexec -n 32 --hosts host0,host1,host2,host3,host4,host5 \
  -envall --cpu-bind=depth bash run_script.sh
```

Do not force `ppn` for the validated 32-rank l64 smoke on 6 nodes.

## S8T32 EMT Proton Smoke

Validated template:

```text
application/EMT_proton/Aurora
application/EMT_proton/Aurora/submit_or_run_interactive.sh
```

Run from an allocation or allocated compute-node SSH shell:

```bash
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement/application/EMT_proton/Aurora
bash submit_or_run_interactive.sh
```

Defaults:

```text
EMT_PROTON_NRANKS=2
EMT_PROTON_MPI_GEOMETRY=1.1.1.2
EMT_PROTON_CONFIG_NUM=0
EMT_PROTON_LAT_TAG=S8T32
EMT_PROTON_GAUGE_PATH=/lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
EMT_PROTON_QMAX=0
t_separations CLI default: --t_separations 2
EMT_PROTON_FLOW_STEPS=1
EMT_PROTON_WIDTH=1.0
EMT_PROTON_GAUSS_SMEAR=0
EMT_PROTON_MAXITER=300
```

Success means PALS exits with status 0, no Python traceback appears in stderr,
and 2pt plus 3pt HDF5 outputs are readable.

## l64 fixed_GLU 32-Rank Smoke

Validated gauge:

```text
/lus/flare/projects/StructNGB/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.1050.coulomb.1e-14
```

Smoke settings:

```bash
export EMT_PROTON_NRANKS=32
export EMT_PROTON_MPI_GEOMETRY=2.2.2.4
export EMT_PROTON_CONFIG_NUM=1050
export EMT_PROTON_LAT_TAG=l64c64a076
export EMT_PROTON_GAUGE_PATH=/lus/flare/projects/StructNGB/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.1050.coulomb.1e-14
export EMT_PROTON_MG_BLOCK=none
export EMT_PROTON_WIDTH=1.0
export EMT_PROTON_FLOW_STEPS=0
export EMT_PROTON_TOL=1e-2
export EMT_PROTON_MAXITER=50
```

Pass `--t_separations 1` on the run-script CLI for this smoke setup.

These are smoke settings only.  Restore physics tolerances, source/sink
separations, smearing width, flow schedule, and output tags for production.

## Known Failure Modes

- QUDA multigrid can fail with `BlockOrtho` tuning parameters whose work group
  exceeds Aurora device limits.  Use `EMT_PROTON_MG_BLOCK=none` for the
  validated smoke route until MG is retuned or fixed.
- Stale QUDA tune caches can replay bad kernel parameters.  Use fresh
  `QUDA_RESOURCE_PATH` directories when changing geometry or backend settings.
- Torch XPU can introduce incompatible oneAPI/MPI runtime libraries.  Default
  to dpnp only.
- `ZE_AFFINITY_MASK` tile wrappers can cause the dpnp backend to see no GPUs.
  Let PyQUDA/QUDA choose devices unless a new affinity strategy is validated.
- Login-shell MPI/GPU runs are unreliable.  Use PBS allocations and PALS.
