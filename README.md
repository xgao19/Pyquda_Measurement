# PyQUDA_qTMD

A Python-based repo for computing quasi TMDs on the lattice with **PyQUDA**.

## Upstream dependency: PyQUDA

This repository is built on top of **PyQUDA**:
- PyQUDA: https://github.com/CLQCD/PyQUDA

Please follow PyQUDA’s official documentation for the base installation, runtime requirements, and backend configuration (CUDA/ROCm/oneAPI).

---

## Repository layout

Current top-level structure:

```bash
application  develop  logs  pyquda_measurement_utils  README.md  systems  test_gauge
```

### `systems/`
This folder contains **system-specific setup notes** for installing and running **PyQUDA** and this
measurement repository on different HPC platforms. It typically documents environment modules,
build options, runtime settings, and runnable job examples.

Currently supported systems:
- **OLCF Frontier** (AMD GPUs / ROCm)
- **ALCF Aurora** (Intel GPUs / oneAPI)
- **NERSC Perlmutter** (NVIDIA GPUs / CUDA)

How to use:
- For Frontier/Aurora, see the corresponding subfolders:
  - `systems/Frontier/`
  - `systems/Aurora/`
- For NVIDIA/CUDA systems (e.g., Perlmutter), refer to the official PyQUDA instructions:
  - https://github.com/CLQCD/PyQUDA

Typical contents include:
- recommended module loads and environment variables
- build & install notes for PyQUDA (and related dependencies)
- job script examples and run instructions
- sanity checks, known issues, and performance notes

> If you are running on a new machine/partition, please add a new subfolder under `systems/`
> and document the environment following the same style as the existing Frontier/Aurora setups.

---

### `pyquda_measurement_utils/`
This folder contains the **core reusable utilities**: common programs/algorithms/functions used by
multiple measurements and workflows.

Current files include (example):
```bash
boosted_smearing_pyquda.py
bw_seq_pyquda.py
io_corr.py
pion_qTMDWF_pyquda.py
proton_qTMD_pyquda.py
tools.py
```

What you typically find here:
- smearing and boosted smearing utilities
- sequential/bwd-propagator helpers
- IO helpers for correlators and metadata
- pion/proton measurement building blocks
- shared tools/utilities used across applications

---

### `application/`
This folder contains **the quasi-TMD measurement applications**, with runnable scripts and
end-to-end workflows.

Key points:
- measurement codes for quasi TMD / qTMDWF (and related observables)
- **examples tailored for different supercomputers**, including job/run examples and practical templates
- usually uses functions from `pyquda_measurement_utils/` to avoid duplication

---

### `develop/`
Development and experimental work area:
- prototypes
- refactors / performance tests
- work-in-progress implementations

---

## Notes

- This repository focuses on **measurement-level code and workflows**.
- For PyQUDA internals and base configuration, please refer to the upstream PyQUDA repository.

---

## Acknowledgements

- PyQUDA: https://github.com/CLQCD/PyQUDA
