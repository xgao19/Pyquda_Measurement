# PyQUDA on Aurora (Intel GPU)

This guide provides step-by-step instructions for setting up and running PyQUDA on Aurora supercomputer with Intel GPU support.

## Overview

PyQUDA is a Python interface for QUDA (QCD on CUDA), a library for lattice QCD computations. This setup enables PyQUDA to run on Intel GPUs using SYCL backend on Aurora.

## Prerequisites

- Access to Aurora supercomputer
- Intel oneAPI toolkit with SYCL support
- Python environment: PyTorch (for Torch backend) or dpnp, and opt-einsum

## Python Environment Setup

It's recommended to use a virtual environment for this setup:

```bash
# Create a virtual environment
python3 -m venv pyquda_env
source pyquda_env/bin/activate

# Choose one of the following backend options:
# Option 1: NumPy backend (slowest but most compatible)
No additional packages needed beyond PyQUDA dependencies

# Option 2: Torch XPU backend (requires --no-deps to avoid MPI pollution)
python3 -m pip install --no-deps torch==2.9.0+xpu torchvision==0.24.0+xpu torchaudio==2.9.0+xpu --index-url https://download.pytorch.org/whl/xpu

# Option 3: dpnp backend (recommended for Intel GPU)
python3 -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

## Installation Steps

### 1. Install QUDA with SYCL Support

Clone the QUDA repository with SYCL support:

```bash
git clone -b feature/sycl https://github.com/lattice/quda.git
cd quda
```

Configure and build QUDA for Intel GPU:

**Important**: Before running `./configure-quda`, you need to modify the paths in the script:

1. **Update the prefix path** (around line 74, 78, 88, 94):
   ```bash
   # Change these lines to your desired installation directory
   prefix="/your/desired/installation/path"
   ```

2. **Update the QUDA source path** (around line 170-171):
   ```bash
   # Change this line to point to your QUDA source directory
   echo $CMAKE --fresh $o /path/to/your/quda/source
   $CMAKE --fresh $o /path/to/your/quda/source
   ```

Then run the configuration and build:

```bash
./configure-quda
ninja
```

The `configure-quda` script sets up the following key configurations:
- **Target**: SYCL backend for Intel GPU
- **SYCL Targets**: `intel_gpu_pvc` (Intel PVC GPU)
- **Compilers**: Intel SYCL compilers (`icpx`, `mpicxx`)
- **Features**: Multi-grid, distance preconditioning, QDP-JIT interface

### 2. Configure PyQUDA Backend

Choose one of the following three backend options:

#### Option 1: NumPy Backend (Baseline)

```python
import pyquda
pyquda.init(backend="numpy")
```

**Pros**: Most compatible, no additional setup required
**Cons**: Slowest performance, CPU-only operations

#### Option 2: Torch XPU Backend (with workarounds)

First, install Torch XPU with `--no-deps` to avoid MPI pollution:

```bash
python3 -m pip install --no-deps torch==2.9.0+xpu torchvision==0.24.0+xpu torchaudio==2.9.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

Configure PyQUDA:

```python
import pyquda
pyquda.init(backend="torch", torch_backend="xpu")
```

**Complex tensor workaround**: Since XPU Torch doesn't support complex matrix multiplication, use the provided `sitecustomize.py`:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**What the sitecustomize.py does:**
- Automatically patches opt-einsum's contract function
- Detects XPU complex tensors and falls back to CPU computation
- Transfers results back to XPU device when appropriate

**Pros**: Good performance for non-complex operations
**Cons**: Complex operations require CPU fallback, MPI environment pollution risk

#### Option 3: dpnp Backend (Recommended)

Install dpnp:

```bash
python3 -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

Configure PyQUDA:

```python
import pyquda
import dpnp as np
pyquda.init(backend="dpnp", backend_target="sycl")
```

**Pros**: 
- Native complex tensor support on Intel GPU
- No CPU fallbacks needed
- Intel-optimized performance
- No PyTorch dependency (avoids MPI pollution)

**Cons**: Requires dpnp installation

## Usage Examples

### NumPy backend
```python
import pyquda
pyquda.init(backend="numpy")
```

### Torch XPU backend
```python
import pyquda
import torch  # ensure XPU build installed with --no-deps

pyquda.init(backend="torch", torch_backend="xpu")
```

### dpnp backend
```python
import pyquda
import dpnp as np

pyquda.init(backend="dpnp", backend_target="sycl")

# point_pion.append(
#    core.gatherLattice(
#       contract("wtzyxjiba,wtzyxjiba->t", point_propag.data.conj(), point_propag.data).real.get(), [0, -1, -1, -1]
#    )
# )

point_pion.append(
   core.gatherLattice(
      dnp.asnumpy(dnp.einsum("wtzyxjiba,wtzyxjiba->t", point_propag.data.conj(), point_propag.data)).real, [0, -1, -1, -1]
   )
)

```

## Log (Hacking PyQUDA on Aurora)

1. Initial validation (NumPy backend works):
   - QUDA and PyQUDA were built and installed successfully.
   - Because the `cupy` backend is not usable on Intel GPUs, we first validated functionality with the `numpy` backend.
   - Both pion 2pt and proton 2pt tests passed, but performance was slow.

2. Trying the Torch XPU backend (install with --no-deps):
   - To improve performance, we switched the backend to Torch (XPU build).
   - When installing the XPU build via pip, add `--no-deps` to avoid polluting the MPI environment: the XPU build of Torch brings Intel oneAPI/IMPI runtime shared libraries that can overshadow the system MPI and libfabric paths, causing `mpi4py` to link against the wrong objects during initialization.

3. Complex matmul limitation in XPU Torch and the temporary workaround:
   - We found XPU Torch lacks support for complex matrix multiplication. Both `opt_einsum` and `torch.einsum` fail on complex contractions with errors like: `RuntimeError: Complex data type matmul is not supported in oneDNN`.
   - Temporary workaround: use `sitecustomize.py` to force `opt_einsum` contractions to run on the CPU and then move the result back to XPU. This works but adds CPU round-trips and overhead.

4. Adopting the dpnp backend (native complex einsum):
   - We evaluated Intel-maintained `dpnp` (Data Parallel Extension for NumPy). Tests show `dpnp.einsum` natively supports complex matrix multiplication on Intel GPUs.
   - Both pion 2pt and proton 2pt tests passed successfully with the dpnp backend, confirming full functionality.
   - We added `dpnp` as a PyQUDA backend option and prefer `dpnp.einsum` on contraction paths to keep Aurora behavior as close as possible to the `cupy` backend.
   - Note: when choosing the `dpnp` path, do not install Torch (XPU) to avoid introducing oneAPI/IMPI runtimes that can pollute MPI. See the `dpnp` repository for details: https://github.com/IntelPython/dpnp

 