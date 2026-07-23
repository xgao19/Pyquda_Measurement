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
- A guided EMT disconnected quark one-point workflow for fixed-cost stochastic and 4D hierarchical-probing convergence tests.
- Shared qTMD/PDF disconnected one-point workflows for pion/proton analysis.
- Existing proton qTMD and pion qTMDWF utilities used as mature references.

Quark EMT production now measures a common complete Dirac basis at every flow
time: 16 local bilinears and (16\times4) unsymmetrized one-derivative
bilinears.  The connected and disconnected EMT tensors are derived from the
same four vector channels, while axial twist-two and local tensor-current
channels remain available for later analysis without additional inversions.
Disconnected derivative primitives contain the complete two-sided
`overleftrightarrow_D` operator.  Their left term is reconstructed from the
right contraction at opposite momentum with gamma5 hermiticity, and the saved
loop includes exactly one closed-fermion-loop Wick minus.

The most validated runtime targets are currently NERSC Perlmutter with NVIDIA
GPUs and Aurora with Intel GPU / SYCL QUDA.  The Aurora route uses the PyQUDA
`develop` branch with the `dpnp` backend.

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
Disconnected_1pt_qTMD_vibe_develop.py Shared disconnected qTMD/PDF one-point loops.
qtmd_operator_utils.py          Shared CG/GI qTMD/PDF displacement and staple transport.
fermion_bilinear_basis.py       Canonical 16-Gamma ordering and physics-basis transform.
pion_qTMDWF_pyquda.py           Mature pion qTMDWF reference workflow.
proton_EMT_vibe_develop.py      Proton flowed EMT connected contractions.
proton_utils_vibe_develop.py    Shared calculation-only proton C2 kernel.
proton_qTMD_pyquda.py           Mature proton qTMD/PDF reference workflow.
tools.py                        Shared MPI/backend utility helpers.
```

Connected pion/proton and disconnected qTMD production share operator
geometry and transport through `qtmd_operator_utils.py`; connected
measurements do not import disconnected production modules. Backend arrays are
converted to host NumPy arrays only through `tools.array_to_numpy`.

Connected proton qTMD/PDF production uses the single backend-independent
runner in `application/nucleon_TMD/shared_runner.py`; Perlmutter and Aurora
entrypoints only provide platform defaults. Connected pion/proton qTMD/PDF
files use one dense `[wilson,momentum,gamma,time]` dataset per operator
(and per proton flavor/polarization), rather than one file per Gamma channel.

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
application/EMT_disconnected_1pt/ Guided quark-loop, convergence, and optional proton C2/3pt workflow; gluon is an advanced add-on.
application/qTMD_disconnected_1pt/ Shared qTMD/PDF one-point workflows.
application/EMFF_pion/          Pion electromagnetic form factor workflow.
application/pion_TMD/           Connected pion CG/GI qTMD and PDF workflow.
```

Each actively maintained application directory has its own README with
measurement-specific parameters and output conventions.

Perlmutter examples:

```bash
# Pion/meson EMT
bash application/EMT_meson/perlmutter/run_quark_3pt.sh \
  --config_num 1000 --pos-boost 0.0.1 --neg-boost 0.0.-1

# Proton EMT
bash application/EMT_proton/perlmutter/run_proton_quark_3pt.sh
bash application/EMT_proton/perlmutter/run_proton_quark_1pt.sh

# EMT disconnected quark one-point smoke (see its README for required run parameters)
bash application/EMT_disconnected_1pt/perlmutter/run_quark_1pt.sh --config_num 1000

# Shared qTMD/PDF disconnected one-point workflow
bash application/qTMD_disconnected_1pt/perlmutter/run_qTMD_1pt.sh --config_num 1000

# Pion qTMD and pion EMFF
bash application/pion_TMD/perlmutter/run_pion_TMD.sh \
  --pos-boost 0.0.1 --neg-boost 0.0.-1
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
QUDA install:  /global/cfs/cdirs/m4559/xgao/software_gradientflow/quda-develop/install
Python venv:   /global/cfs/cdirs/m4559/xgao/software_gradientflow/venv-quda-develop
PyQUDA source: /global/cfs/cdirs/m4559/xgao/software_gradientflow/PyQUDA-develop
Repo root:     /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement
```

Activate the environment:

```bash
source /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
export QUDA_PATH=/global/cfs/cdirs/m4559/xgao/software_gradientflow/quda-develop/install
```

Run the baseline gradient-flow smoke test:

```bash
bash /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh
```

## Aurora Runtime

Aurora helper notes live in:

```text
systems/Aurora/
```

Current validated shared paths:

```text
QUDA install:  /lus/flare/projects/StructNGB/xgao/software_260507/install/quda
Python venv:   /lus/flare/projects/StructNGB/xgao/software_gradientflow/pyquda_env
PyQUDA source: /lus/flare/projects/StructNGB/xgao/software_gradientflow/PyQUDA
Repo root:     /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement
Activation:    /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

Activate the environment:

```bash
module load gcc/13.4.0
module load oneapi/release/2025.3.1
module load mpich/opt/5.0.0.aurora_test.3c70a61
module load libfabric/1.22.0
module load cray-pals/1.8.0
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
```

Aurora best practices:

- Use PyQUDA `develop` with `backend="dpnp", backend_target="sycl"`.
- Keep `h5py` parallel-enabled; `h5py.get_config().mpi` should be `True`.
- Prefer PBS/PALS launches from allocated compute nodes for MPI checks.
- Avoid Torch XPU in the default environment because it can pollute MPI and
  oneAPI runtime resolution.
- Avoid tile-affinity wrappers that set `ZE_AFFINITY_MASK` for the current
  `dpnp` route; let PyQUDA/QUDA select the device per rank.

Run the validated Aurora proton EMT smoke:

```bash
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement/application/EMT_proton/Aurora
bash submit_or_run_interactive.sh
```

See `systems/Aurora/README.md` for install details, PALS launch examples, and
the 32-rank l64 fixed_GLU smoke settings.

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

All active workflows that apply HYP smearing use
`gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)`.  In the PyQUDA/QUDA API the final
argument is `dir_ignore`, not a projection count.  QUDA normalizes both the
historical value `4` and the canonical value `-1` to four-dimensional HYP;
the repository fixes the spelling to `-1` and does not expose it through an
environment variable or CLI option.

## Output Conventions

- Active workflows write HDF5 output.
- Per-application `.gitignore` files exclude generated data, logs, QUDA tuning
  caches, CuPy caches, and profile files.
- EMT HDF5 helpers live in `io_corr.py`.
- qTMD-like HDF5 file names are built with tag helpers such as
  `get_qTMD_file_tag(...)`.
- Hadron correlator EMT file names include standard `lat / cfg / ama / src` fields followed by an explicit setup and channel tag.
- Hadron-independent quark-loop files omit `src` and use `lat / cfg / ama / sm`.
- Proton connected C2 channels use `<setup>.src<SRC>`, while C3 channels use `<setup>.src<SRC>.sink<SINK>.<polarizations>`. The setup identifies smearing and boost parameters, for example `1HYP_GSRC_W9_binx0y0z0_boutx0y0z0`.
- Proton connected 3pt names additionally encode the sink kinematics as `PX<px>PY<py>PZ<pz>dt<tsep>`. Multiple separations are written as separate files, and `source_jobs[*]["tags"]` may select any nonempty subset declared by the measurement.

## EMT Notes

The EMT development files include detailed English module docstrings with the
correlation-function definitions and the contraction formulas used by the code.
The shared numerical Gamma definition, the
`pyquda_bitmask16_with_physics_transform_v1` HDF5 schema, the relation between
the raw PyQUDA matrices actually used in contractions and physics-labelled
axial/tensor channels, analysis recipes, and per-file storage weights are
documented in
[`docs/EMT_gamma_and_raw_bilinears.md`](docs/EMT_gamma_and_raw_bilinears.md).

For pion/meson EMT:

- Meson 2pt scans all 16 sink gamma structures.
- Connected quark 3pt uses the fixed-sink meson sequential-source convention
  referred to during development as convention B.
- With unequal pion momentum-smearing boosts, the positive-boost line is the
  fixed-sink spectator and the negative-boost line is the EMT insertion line.
  Momentum smearing acts only at hadron endpoints; both insertion fields still
  undergo the same four-dimensional fermion flow.
- `meson_sign = 1` is the active convention.
- Quark and gluon flow schedules are aligned: measure first, then flow.
- `step = 0` is the unflowed measurement.
- The first flow interval is subdivided into 10 smaller steps.

For quark/gluon one-point data:

- Quark 1pt schema v5 stores all 16 local and 64 unsymmetrized derivative
  primitive bilinears plus an explicitly named flowed-noise norm.
- Quark 1pt defaults to decomposition-independent full-volume counter-based `Z4` noise.
- Never seed an ordinary array-backend RNG identically on every MPI rank to
  build a distributed stochastic source.  Equal local shapes then receive
  repeated local noise, violating the intended global covariance.  EMT, qTMD,
  and standalone ringed production all use global-coordinate counter noise;
  old backend-RNG data should not be mixed with these outputs.
- Disconnected qTMD loops use
  `xi^dagger P(q,tau) Gamma O_b D^{-1} xi`: apply every displacement or Wilson
  line to the solved field, not to the noise.  The removed reversed-trace qTMD
  outputs are invalid and must be regenerated.
- Quark 1pt can use either ordinary `zn` noise or `hierarchical_probing`.
- For hierarchical probing, `effective_n_inversions = n_base_noise * hp_num_vectors`.
- Raw quark 1pt files store only `base_noise_index` and `hp_index` bookkeeping.
  Reconstruct the effective source index as
  `base_noise_index * hp_vectors_per_base + hp_index`.
- Current HP ordering choices include `interleaved_xyz_binary_projected_to_evenodd`
  with a time-independent spatial sign pattern, plus 4D orderings such as
  `interleaved_xyzt_binary_projected_to_evenodd`,
  `global_xyzt_gray_projected_to_evenodd`, and
  `spatial_xyz_then_t_gray_projected_to_evenodd`.
- Flowed EMT production defaults to the isotropic 4D
  `interleaved_xyzt_binary_projected_to_evenodd` ordering; spatial HP must be
  selected explicitly for diagnostic comparisons.
- All HP choices multiply a full-volume base source; spatial HP is not time
  dilution.  Four-dimensional fermion flow spreads the flowed fields in time,
  while the saved insertion-time axis remains explicit.
- A single source-independent EMTc loop file per configuration stores all
  absolute insertion times and is reused for every hadron source time.
- The identity local bilinear is stored only once in the 16-Gamma primitive;
  `flowed_noise_norm` contains only the flowed source norm.
- EMT-derived ringed kinetic data live under `derived/ringed` in the same EMTc
  file, so one atomic rename publishes the loop and kinetic data together.
- The standalone `application/flowed_quark_ringed_norm` workflow remains
  available for dedicated kinetic-only high-statistics runs. Its
  `RingedQuark1pt` implementation inherits the EMT production runner but uses
  an independent four-vector-diagonal contraction, so it does not allocate the
  full EMT primitive basis. It uses the same base/HP-part shards, base-level
  sample log, and explicit finalization as
  EMT/qTMD.  Production resume trusts the log and does not probe shard files,
  so completed parts may be moved before the remaining bases run. Final ringed
  factors are computed from the ensemble-averaged kinetic expectation value,
  not by averaging per-configuration inverse factors.
- Production quark 1pt wrappers default to base/HP interval shards.  Completed
  bases are recorded as exact lines in a lightweight text log. An explicit
  destination-side streaming finalizer checks parts while merging and publishes
  one canonical EMTc file only after complete base coverage.
- Gluon 1pt stores the flowed gluonic EMT building block.
- Renormalized gradient-flow EMT combinations, vacuum subtractions, and mixing
  coefficients are applied in downstream analysis, not inside these kernels.

For proton EMT:

- The current proton EMT path computes connected U and D insertions.
- Disconnected diagrams and renormalization/mixing factors are not included.
- Proton EMT and qTMD call the same calculation-only proton two-point kernel;
  each workflow retains its own file writer and provenance.
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
