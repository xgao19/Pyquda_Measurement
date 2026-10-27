# PyQUDA Measurement Tests

The default smoke-test runner avoids hard dependencies on GPU nodes or
pre-existing HDF5 output fixtures:

```bash
source systems/perlmutter/activate-venv-quda.sh
python tests/run_smoke_tests.py
```

Tests marked with `TEST_REQUIRES = "gpu"` or
`TEST_REQUIRES = "external_hdf5"` are skipped by default.  To attempt those
optional checks as well, run:

```bash
source systems/perlmutter/activate-venv-quda.sh
python tests/run_smoke_tests.py --include-optional
```

The optional GPU tests still skip cleanly if PyQUDA cannot initialize a CUDA
device.  Optional external-HDF5 tests skip cleanly if their reference output
files are not present.

The soft-factor contractions have independent CPU references using full 12x12
matrix products, adjoints and traces. They cover native GPT SS spatial sums,
the 16 SP sink channels, C4 color closure, phases, shifts and HDF5 provenance.
Run them without a GPU using a Python environment with NumPy, pytest and h5py:

```bash
python tests/run_probing_cpu_tests.py --pyquda-root ../LQCD/PyQUDA \
  tests/test_pion_soft_factor_contraction_algebra.py \
  tests/test_pion_soft_factor_optimized.py \
  tests/test_pion_soft_factor_shifts.py \
  tests/test_pion_soft_factor_hdf5_schema.py -q
```

Adjust `--pyquda-root` to the installed PyQUDA source tree. This runner loads
the real Gamma definitions but explicitly stubs GPU/MPI calls; neighbor and
reduction fixtures are CPU simulations, not distributed hardware validation.

To compare the corrected production methods directly with native GPT CPU tensor
operators and the original gpt-run contraction expressions, run:

```bash
python tests/check_pion_soft_factor_gpt_cpu.py \
  --gpt-root /path/to/gpt --gpt-run-root ../gpt-run \
  --pyquda-root ../LQCD/PyQUDA --output-dir /tmp/soft-factor-gpt-comparison
```

This source-based check does not require or execute compiled `cgpt`. It runs the
native tensor/object-type/Gamma definitions and original contraction expressions;
Python site loops implement shifts, spatial sums and the documented `slice_trDA`
kernel loop. PyQUDA uses its real checkerboard conversions and `MomentumPhase`.
The check compares zero/nonzero momenta over three seeds, includes ordered C4
insertions, and explicitly applies the stored raw-to-physical Gamma signs.
It writes `channels.csv`, `comparison.json` (including source hashes), and
`paired_arrays.npz` with both results. This validates CPU algebra, not MPI/GPU
kernels, gauge inversion or production measurements.
