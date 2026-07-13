# Perlmutter Setup

This directory contains the Perlmutter-specific QUDA / PyQUDA setup that was
validated on `login32` with an A100 GPU.

## What is here

- `configure-quda`
  - CMake configuration for building QUDA against the current Perlmutter CUDA
    stack.
- `activate-venv-quda.sh`
  - Helper for activating the shared Python virtual environment together with
    QUDA, Cray HDF5, MPI, and the PyQUDA path settings.
- `submit_batch_template.sh`
  - Batch-job template for running a PyQUDA driver under `srun`.
- `check-gradient-flow.sh`
  - Minimal smoke test that exercises `gradientGaugeFlow` and fermion
    `gradientFlow` using the bundled test gauge.
- `check-gradient-flow.py`
  - The Python implementation of the smoke test. The shell wrapper just sets
    up the runtime environment and then calls this file.
- `requirements.txt`
  - Python packages that match the currently validated venv.

## Validated software stack

The working stack that was tested on Perlmutter is:

- `cpe/25.09`
- `gcc-native/14`
- `cudatoolkit/12.9`
- `craype-accel-nvidia80`
- `cray-mpich-abi/9.0.1`
- `cray-hdf5-parallel/1.14.3.7`

The QUDA install prefix is:

- `/global/cfs/cdirs/m4559/xgao/software_gradientflow/quda-develop/install`

The shared Python environment is:

- `/global/cfs/cdirs/m4559/xgao/software_gradientflow/venv-quda-develop`

The PyQUDA checkout is expected at:

- `/global/cfs/cdirs/m4559/xgao/software_gradientflow/PyQUDA-develop`

## Python requirements

The file `requirements.txt` pins the Python packages that were present in the
validated environment:

- `Cython==3.2.4`
- `cupy-cuda12x==14.0.1`
- `h5py==3.16.0`
- `mpi4py==4.1.1`
- `numpy==2.4.4`
- `opt-einsum==3.4.0`
- `packaging==26.2`

Install them with:

```bash
source /global/cfs/cdirs/m4559/xgao/software_gradientflow/venv-quda-develop/bin/activate
python -m pip install -r /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/requirements.txt
```

If you need the full PyQUDA editable install, use:

```bash
export QUDA_PATH=/global/cfs/cdirs/m4559/xgao/software_gradientflow/quda-develop/install
cd /global/cfs/cdirs/m4559/xgao/software_gradientflow/PyQUDA-develop/pyquda_core
python -m pip install -e .
cd /global/cfs/cdirs/m4559/xgao/software_gradientflow/PyQUDA-develop
python -m pip install -e .
```

## Activation flow

The easiest way to set up the runtime environment is:

```bash
source /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
```

That helper:

- activates the shared `venv`
- exports `QUDA_PATH`
- exports the PyQUDA repository path into `PYTHONPATH`
- sets `HDF5_DIR` to the Cray parallel HDF5 runtime
- preloads the Cray MPI GPU transport and HDF5 libraries
- keeps `MPICH_GPU_SUPPORT_ENABLED=1`

After sourcing it, you should be able to run:

```bash
python -c "import pyquda, cupy, h5py, mpi4py"
```

## Smoke test

The bundled smoke test uses the local NERSC gauge file:

- `test_gauge/S8T8_wilson_b6.0`

Run it with:

```bash
bash /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh
```

If you already have the environment loaded, you can run the Python entry
directly:

```bash
python /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.py
```

It checks:

- `nvidia-smi`
- `readNERSCGauge(...)`
- `gradientGaugeFlow("wilson", ...)`
- `gradientGaugeFlow("symanzik", ...)`
- fermion `gradientFlow(...)`

The default test parameters are:

- `GRID_SIZE=1,1,1,1`
- `LATT_SIZE=8,8,8,8`
- `FLOW_STEPS=1`
- `FLOW_EPSILON=0.01`

You can override them without editing the script, for example:

```bash
GRID_SIZE=1,1,1,1 FLOW_STEPS=2 FLOW_EPSILON=0.005 \
  bash /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh
```

## Batch jobs

The template batch script is:

- `submit_batch_template.sh`

It currently:

- loads the validated Perlmutter module stack
- prints `nvidia-smi`
- activates the shared `venv`
- sets `QUDA_PATH`, `QUDA_RESOURCE_PATH`, `CUPY_CACHE_DIR`, and `OMP_NUM_THREADS`
- runs `srun --mpi=cray_shasta`

Override the driver and arguments with:

- `PYQUDA_DRIVER`
- `PYQUDA_ARGS`

For example:

```bash
PYQUDA_DRIVER=pyquda_main.py PYQUDA_ARGS="--mpi_geometry 2.2.2.2" \
  sbatch /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/submit_batch_template.sh
```

## Environment variables

The main knobs you can override are:

- `SOFTWARE_ROOT`
- `REPO_ROOT`
- `QUDA_PATH`
- `VENV_DIR`
- `PYQUDA_ROOT`
- `PYQUDA_DRIVER`
- `PYQUDA_ARGS`
- `QUDA_RESOURCE_PATH`
- `CUPY_CACHE_DIR`
- `OMP_NUM_THREADS`
- `MPICH_GPU_SUPPORT_ENABLED`
- `GAUGE_PATH`
- `GRID_SIZE`
- `LATT_SIZE`
- `FLOW_STEPS`
- `FLOW_EPSILON`

## Notes

- The helper script sets `LD_PRELOAD` so `h5py` uses the Cray parallel HDF5
  runtime and avoids the HDF5 version mismatch warning.
- `check-gradient-flow.sh` is intended as a fast correctness check, not a full
  performance benchmark.
- The checked-in test gauge is small, so the smoke test should finish quickly
  on a GPU node or a GPU-capable login session.
