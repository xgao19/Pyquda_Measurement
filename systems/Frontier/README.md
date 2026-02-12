# PyQUDA on Frontier (AMD MI250X / ROCm)

This guide provides step-by-step instructions for setting up and running PyQUDA on Frontier with AMD MI250X GPUs using the ROCm/HIP stack. **CPU-only backends are not supported in this setup.** PyQUDA must use **CuPy (HIP)**.

## Overview

PyQUDA is a Python interface for QUDA (QCD on CUDA), a library for lattice QCD computations. On Frontier, QUDA is built with the **HIP** backend, and PyQUDA uses **CuPy (HIP)** as the GPU array backend for tensor contractions (einsum/contract). MPI-enabled I/O can be provided via **mpi4py** and **h5py** (built against the system MPI + MPI-enabled HDF5).

## Prerequisites

- Access to Frontier (OLCF)
- ROCm toolchain available on the system
- A Python environment (virtualenv recommended)
- Ninja (for QUDA builds)
- CuPy built/installed with HIP support (**required**)
- MPI-enabled `mpi4py` and `h5py`

## Python Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate

python3 -m pip install -U pip setuptools wheel
python3 -m pip install ninja numpy opt_einsum
```

### Install CuPy (HIP) — Required

CuPy is the required GPU backend on Frontier via HIP/ROCm.

```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm-6.2.4
export HCC_AMDGPU_TARGET=gfx90a

export CFLAGS="-I${ROCM_HOME}/include \
-I${ROCM_HOME}/include/hipblas \
-I${ROCM_HOME}/include/hipsparse \
-I${ROCM_HOME}/include/hipfft \
-I${ROCM_HOME}/include/rocsolver"

python3 -m pip install cupy
```

## MPI-enabled h5py and mpi4py (Optional but Common)

If your workflow requires MPI I/O with HDF5, build `mpi4py` and `h5py` from source against the system MPI and MPI-enabled HDF5:

```bash
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

HDF5_MPI="ON" CC=cc HDF5_DIR=${HDF5_ROOT} \
  pip install --no-cache-dir --no-binary=h5py h5py
```

## Installation Steps

### 1. Build and Install QUDA (HIP)

This setup uses QUDA `develop` at the following commit:

- `a76d22e9bc9008c640454e30763e5d6e6b304c63` (merge PR #1526: adjoint flow)

Clone and check out the correct commit:

```bash
git clone https://github.com/lattice/quda.git
cd quda
git checkout a76d22e9bc9008c640454e30763e5d6e6b304c63
```

Configure QUDA for Frontier (MI250X / gfx90a) with HIP:

```bash
mkdir -p build && cd build

cmake .. \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DQUDA_GPU_ARCH=gfx90a \
  -DQUDA_MPI=ON \
  -DQUDA_COVDEV=ON \
  -DQUDA_MULTIGRID=ON \
  -DQUDA_DIRAC_DEFAULT_OFF=ON \
  -DQUDA_DIRAC_WILSON=ON \
  -DQUDA_DIRAC_CLOVER=ON \
  -DQUDA_DIRAC_STAGGERED=ON \
  -DQUDA_DIRAC_LAPLACE=ON \
  -DQUDA_CLOVER_DYNAMIC=OFF \
  -DQUDA_CLOVER_RECONSTRUCT=OFF \
  -DQUDA_TARGET_TYPE=HIP \
  -DROCM_PATH=$ROCM_PATH \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=hipcc

ninja
ninja install
```

### 2. Install PyQUDA and PyQUDA-Utils

Install PyQUDA and PyQUDA-Utils into the same Python environment:

```bash
python3 -m pip install PyQUDA==0.10.32 PyQUDA-Utils==0.10.30.post0
```

## Configure PyQUDA Backend (CuPy Only)

CPU-only backends are **not supported** in this README. Use CuPy (HIP):

```python
import pyquda
import cupy as cp  # required

pyquda.init(backend="cupy")
```

**Pros**
- GPU-accelerated contractions on MI250X via ROCm/HIP

**Cons**
- Requires installing CuPy with HIP support

## Usage Example (CuPy backend)

```python
import pyquda
import cupy as cp
from opt_einsum import contract

pyquda.init(backend="cupy")

# Example contraction pattern (replace tensors with your objects)
# out = contract("wtzyxjiba,wtzyxjiba->t", A.conj(), A)
```

## Reference Package Versions (Known Working)

The following package versions were used in a working Frontier setup:

```
Package      Version
------------ -------------
cupy         13.5.1
Cython       3.1.4
fastrlock    0.8.3
h5py         3.14.0
mpi4py       4.1.1
ninja        1.13.0
numpy        1.26.4
opt_einsum   3.4.0
packaging    25.0
pip          25.2
pycparser    2.23
PyQUDA       0.10.14
PyQUDA-Utils 0.10.14.post1
setuptools   80.9.0
wheel        0.45.1
```

## Log (Hacking PyQUDA on Frontier)

1. QUDA build validation:
   - Built QUDA from `develop` at commit `a76d22e9bc9008c640454e30763e5d6e6b304c63`.
   - Configured with HIP backend (`QUDA_TARGET_TYPE=HIP`) and MI250X architecture (`gfx90a`).
   - Enabled multigrid and CovDev, and enabled the required Dirac operators.

2. Python stack validation (CuPy required):
   - Installed PyQUDA `0.10.14` and PyQUDA-Utils `0.10.14.post1`.
   - Installed CuPy `13.5.1` with HIP support using `CUPY_INSTALL_USE_HIP=1`.
   - Installed `opt_einsum` for contraction routing.

3. MPI-enabled HDF5 stack (when needed):
   - Built `mpi4py` and `h5py` from source to ensure correct linkage with system MPI and MPI-enabled HDF5.

