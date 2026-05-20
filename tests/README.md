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
