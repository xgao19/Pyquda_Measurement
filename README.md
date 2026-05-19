# Pyquda_Measurement

PyQUDA-based lattice QCD measurement workflows for qTMD, pion EMFF, and flowed
energy-momentum tensor observables.

This repository contains measurement-level Python code built on top of
[PyQUDA](https://github.com/CLQCD/PyQUDA).  PyQUDA and QUDA provide the lattice
field objects, gauge/fermion operations, Dirac inversions, and GPU backends;
this repository provides reusable contraction utilities and runnable
application scripts.

## Current Focus

The actively maintained workflows are:

- Pion qTMD / PDF-style connected measurements.
- Pion electromagnetic form-factor connected measurements.
- Pion/meson EMT measurements with flowed quark and gluon observables.
- Proton EMT connected measurements plus shared quark/gluon one-point outputs.
- Shared EMT disconnected one-point workflows with optional hierarchical probing.
- Existing proton qTMD and pion qTMDWF utilities used as mature references.

The most validated runtime target is currently NERSC Perlmutter with NVIDIA
GPUs.

## Repository Layout

```text
application/                 Runnable measurement workflows.
develop/                     Experimental and scratch work.
pyquda_measurement_utils/    Shared contraction, smearing, sequential-source, and IO code.
systems/                     System-specific environment and build helpers.
test_gauge/                  Small gauges used for smoke tests.
SESSION_MEMORY.md            Collaboration memory and validated state.
```

## Core Utilities

Important files in `pyquda_measurement_utils/`:

```text
boosted_smearing_pyquda.py      Gaussian/boosted smearing helpers.
bw_seq_pyquda.py                Proton and meson fixed-sink sequential-source builders.
io_corr.py                      HDF5 tag helpers and correlator writers.
pion_EMFF_vibe_develop.py       Pion local-current EMFF contractions.
pion_EMT_vibe_develop.py        Pion/meson flowed EMT contractions and one-point observables.
pion_qTMD_vibe_develop.py       Pion qTMD and PDF-style connected contractions.
pion_qTMDWF_pyquda.py           Mature pion qTMDWF reference workflow.
proton_EMT_vibe_develop.py      Proton flowed EMT connected contractions.
proton_qTMD_pyquda.py           Mature proton qTMD/PDF reference workflow.
tools.py                        Shared MPI/backend utility helpers.
```

Naming convention:

- `pion_EMT_vibe_develop.py` is the active renamed version of the previous
  meson EMT development file.
- `proton_EMT_vibe_develop.py` reuses the shared quark/gluon one-point EMT
  utilities from `pion_EMT_vibe_develop.py`.
- The older legacy `EMT_meson.py` source was removed after migration to
  `pion_EMT_vibe_develop.py`; recover it from Git history if needed.

Validation status:

- Files named `*_vibe_develop.py` are active development implementations.
- They have been checked with syntax tests and small smoke tests where noted,
  but they still need stricter hands-on validation before production use.
- Before using them for final physics production, run targeted comparisons
  against trusted baselines, nonzero-momentum checks, multiple source/sink
  separations, and any relevant Ward-identity or symmetry checks.

## Applications

Runnable workflows live under `application/`.

```text
application/EMT_meson/          Pion/meson EMT workflows.
application/EMT_proton/         Proton EMT workflows.
application/EMT_disconnected_1pt/ Shared quark/gluon EMT one-point workflows.
application/EMFF_pion/          Pion electromagnetic form factor workflow.
application/pion_TMD_CG/        Pion qTMD/PDF-style workflow.
```

Each actively maintained application directory has its own README with
measurement-specific parameters and output conventions.

Perlmutter examples:

```bash
# Pion/meson EMT
bash application/EMT_meson/perlmutter/run_quark_3pt.sh
bash application/EMT_meson/perlmutter/run_quark_1pt.sh
bash application/EMT_meson/perlmutter/run_gluon_1pt.sh

# Proton EMT
bash application/EMT_proton/perlmutter/run_proton_quark_3pt.sh
bash application/EMT_proton/perlmutter/run_proton_quark_1pt.sh
bash application/EMT_proton/perlmutter/run_proton_gluon_1pt.sh

# Shared EMT disconnected one-point workflows
bash application/EMT_disconnected_1pt/perlmutter/run_quark_1pt.sh
bash application/EMT_disconnected_1pt/perlmutter/run_gluon_1pt.sh

# Pion qTMD and pion EMFF
bash application/pion_TMD_CG/perlmutter/run_pion_TMD_CG.sh
bash application/EMFF_pion/perlmutter/run_pion_EMFF.sh
```

The `run_*.sh` wrappers are intended for smoke tests or interactive/login-node
validation.  The `submit_*.sh` wrappers provide Perlmutter batch-job templates.

## Perlmutter Runtime

Perlmutter helper files live in:

```text
systems/perlmutter/
```

Useful files:

```text
activate-venv-quda.sh       Activates the validated Python/QUDA runtime.
configure-quda              QUDA configure helper.
requirements.txt            Python package versions for the validated venv.
check-gradient-flow.py      Minimal gradient-flow smoke test.
check-gradient-flow.sh      Shell wrapper for the smoke test.
submit_batch_template.sh    Generic Perlmutter batch template.
README.md                   Detailed Perlmutter notes.
```

Validated shared paths on the current Perlmutter setup:

```text
QUDA install:  /global/cfs/cdirs/m3760/xgao/software/quda/install
Python venv:   /global/cfs/cdirs/m3760/xgao/software/venv
PyQUDA source: /global/cfs/cdirs/m3760/xgao/software/PyQUDA
Repo root:     /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement
```

Activate the environment:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
export QUDA_PATH=/global/cfs/cdirs/m3760/xgao/software/quda/install
```

Run the baseline gradient-flow smoke test:

```bash
bash /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh
```

## Test Gauge

Most current smoke tests use:

```text
test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

Some older system smoke tests use:

```text
test_gauge/S8T8_wilson_b6.0
```

Applications expose environment variables such as `*_GAUGE_PATH`, `*_DATA_DIR`,
`*_QMAX`, and `*_FLOW_STEPS` so the same scripts can be used for both tiny
validation runs and larger production-style tests.

## Output Conventions

- Active workflows write HDF5 output.
- Per-application `.gitignore` files exclude generated data, logs, QUDA tuning
  caches, CuPy caches, and profile files.
- EMT HDF5 helpers live in `io_corr.py`.
- qTMD-like HDF5 file names are built with tag helpers such as
  `get_qTMD_file_tag(...)`.
- EMT file names include the standard `lat / cfg / ama / src / sm` tags.

## EMT Notes

The EMT development files include detailed English module docstrings with the
correlation-function definitions and the contraction formulas used by the code.

For pion/meson EMT:

- Meson 2pt scans all 16 sink gamma structures.
- Connected quark 3pt uses the fixed-sink meson sequential-source convention
  referred to during development as convention B.
- `meson_sign = 1` is the active convention.
- Quark and gluon flow schedules are aligned: measure first, then flow.
- `step = 0` is the unflowed measurement.
- The first flow interval is subdivided into 10 smaller steps.

For quark/gluon one-point data:

- Quark 1pt stores stochastic `Tmunu` and `CHI` outputs.
- Quark 1pt can use either ordinary `zn` noise or `hierarchical_probing`.
- For hierarchical probing, `effective_n_inversions = n_base_noise * hp_num_vectors`.
- Raw quark 1pt files store `source_index`, `base_noise_index`, and `hp_index`
  bookkeeping datasets.
- Current HP ordering choices are `global_xyzt_gray_projected_to_evenodd` and
  `spatial_xyz_then_t_gray_projected_to_evenodd`.
- The ringed-fermion kinetic normalization can be reconstructed at q=0 from
  `avg/Tmunu/T11`, `T22`, `T33`, and `T44`.
- `CHI` is a scalar trace/noise diagnostic, not the standard ringed-fermion
  normalization by itself.
- Gluon 1pt stores the flowed gluonic EMT building block.
- Renormalized gradient-flow EMT combinations, vacuum subtractions, and mixing
  coefficients are applied in downstream analysis, not inside these kernels.

For proton EMT:

- The current proton EMT path computes connected U and D insertions.
- Disconnected diagrams and renormalization/mixing factors are not included.
- Proton 2pt reuses the mature proton qTMD two-point contraction.
- Proton quark/gluon one-point workflows reuse the shared EMT one-point code.

## Development Guidelines

- Keep reusable physics kernels in `pyquda_measurement_utils/`.
- Keep runnable machine-specific workflows in `application/<measurement>/<system>/`.
- Prefer HDF5 output for new workflows.
- Keep code comments in English.
- Use the small test gauges and Perlmutter smoke scripts before scaling up.
- Do not overwrite validated baseline data unless the task explicitly requires
  refreshing the baseline.

## Session Memory

`SESSION_MEMORY.md` records validated environment details, smoke-test results,
measurement baselines, and important physics/convention decisions.  New agent
sessions should read it before making substantial changes.

## Upstream Dependency

This repository depends on PyQUDA:

- PyQUDA: https://github.com/CLQCD/PyQUDA

For PyQUDA internals, installation details, and backend-specific requirements,
refer to the upstream PyQUDA documentation.
