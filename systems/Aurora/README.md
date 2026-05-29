# PyQUDA On Aurora

This note records the current Aurora best practice for this measurement
repository.  The validated route is Intel GPU / SYCL QUDA with the PyQUDA
`develop` branch and the `dpnp` backend.

## Validated Layout

Current shared installation paths:

```text
Measurement repo: /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement
PyQUDA source:    /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
Python venv:      /lus/flare/projects/StructNGB/xgao/software_gradientflow/pyquda_env
QUDA install:     /lus/flare/projects/StructNGB/xgao/software_260507/install/quda
Activation:       /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

The QUDA install is reused from `software_260507`; do not rebuild QUDA unless
the QUDA side itself needs to change.

## Module Environment

Use the Aurora programming environment that matches the existing QUDA build:

```bash
module load gcc/13.4.0
module load oneapi/release/2025.3.1
module load mpich/opt/5.0.0.aurora_test.3c70a61
module load libfabric/1.22.0
module load cray-pals/1.8.0
```

Then activate the PyQUDA environment:

```bash
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

The activation script sets the key paths and runtime knobs:

```text
QUDA_PATH
PYQUDA_ROOT
PYTHONPATH
LD_LIBRARY_PATH
MPICH_GPU_SUPPORT_ENABLED=1
```

## Python Environment

The venv should be created with Aurora's system Python 3.12 from the loaded
module stack, not `/usr/bin/python3`.

Install PyQUDA from source and check out `develop`:

```bash
git clone https://github.com/CLQCD/PyQUDA.git /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
git checkout develop
```

Recommended packages:

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

`h5py` must be parallel-enabled.  If a later pip operation replaces it with a
non-MPI wheel, uninstall it and rebuild from source with `CC=mpicc` and
`HDF5_MPI=ON`.

## Backend Choice

Use `dpnp` on SYCL:

```python
from pyquda import init

init(mpi_geometry, backend="dpnp", backend_target="sycl", enable_mps=True)
```

Avoid Torch XPU for the default Aurora environment.  Torch XPU can introduce
additional oneAPI and MPI runtime libraries into the process, and that can
pollute `mpi4py`, `h5py`, MPICH, and libfabric resolution.  The current tested
route does not need Torch.

## Runtime Checks

Run lightweight checks after activating the environment:

```bash
python -c "import h5py; print(h5py.get_config().mpi)"
python -c "import pyquda, pyquda_utils; print(pyquda.__file__)"
python - <<'PY'
import h5py, mpi4py.MPI
print("h5py mpi:", h5py.get_config().mpi)
print("mpi size:", mpi4py.MPI.COMM_WORLD.Get_size())
PY
```

Expected `h5py.get_config().mpi` is `True`.

For multi-rank checks, use an interactive allocation or a PBS job.  Avoid
forcing PALS launches from an ordinary login shell.

```bash
/opt/cray/pals/1.8/bin/mpiexec -n 2 -envall \
  python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

## Launching On Compute Nodes

For interactive testing, request nodes with PBS, then SSH to one allocated
compute node and launch with PALS:

```bash
qsub -I -q debug-scaling -A StructNGB -N pyquda_test \
  -l select=6:ngpus=6 -l walltime=01:00:00 -l filesystems=flare -l place=scatter

ssh <allocated-node>
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement
```

Use PALS directly:

```bash
/opt/cray/pals/1.8/bin/mpiexec -n 32 --hosts host0,host1,host2,host3,host4,host5 \
  -envall --cpu-bind=depth bash run_script.sh
```

Do not force `ppn` when using 32 ranks on 6 nodes if the goal is to spread ranks
unevenly while still using the available GPUs.

## GPU Visibility Notes

For the current PyQUDA `develop` plus `dpnp` path, let PyQUDA/QUDA select the
device per rank.  Avoid tile-affinity wrappers that set `ZE_AFFINITY_MASK=0.0`
or similar masks, because they can make the `dpnp` backend see zero devices.

Recommended defaults for the current route:

```bash
unset ZE_FLAT_DEVICE_HIERARCHY
unset ZE_AFFINITY_MASK
unset ONEAPI_DEVICE_SELECTOR
unset ONEAPI_DEVICE_FILTER
export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
```

Enable tuning in a fresh cache only when needed:

```bash
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=/path/to/new/cache
```

Avoid reusing stale tune caches after changing rank geometry, lattice size, or
QUDA/PyQUDA versions.

## EMT Proton Smoke Tests

The shortest validated Aurora template is:

```text
application/EMT_proton/Aurora
```

It runs the bundled S8T32 gauge with 2 ranks:

```bash
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement/application/EMT_proton/Aurora
bash submit_or_run_interactive.sh
```

For a 32-rank l64 validation, the tested gauge is:

```text
/lus/flare/projects/StructNGB/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.1050.coulomb.1e-14
```

A successful smoke used:

```text
nranks=32
mpi_geometry=2.2.2.4
EMT_PROTON_MG_BLOCK=none
EMT_PROTON_T_SEPS=1
EMT_PROTON_WIDTH=1.0
EMT_PROTON_FLOW_STEPS=0
EMT_PROTON_TOL=1e-2
EMT_PROTON_MAXITER=50
```

That run produced readable 2pt and 3pt HDF5 outputs.  These are smoke settings,
not production physics settings.

## Known Issues

- The current QUDA multigrid path can fail on Aurora with `BlockOrtho` tuning
  parameters whose work group exceeds the device limit.  Use
  `EMT_PROTON_MG_BLOCK=none` for the validated smoke path until the MG tuning
  issue is resolved.
- `QUDA_ENABLE_TUNING=0` with a stale cache can reuse bad kernel parameters.
  Use a fresh cache and `QUDA_ENABLE_TUNING=1` when investigating tuning issues.
- Source/sink smearing and production tolerances are much more expensive than
  the smoke settings above.  First validate the gauge, MPI geometry, and HDF5
  output path with a small smoke run, then restore stricter parameters.
