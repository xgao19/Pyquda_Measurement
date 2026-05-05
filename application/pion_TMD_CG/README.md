# Pion TMD CG Applications

This directory contains platform-specific entry scripts for the pion connected
qTMD and PDF measurement workflows.

## Layout

- `perlmutter/`: current Perlmutter scripts for the pion qTMD/PDF workflow.

## Measurement Source

The active measurement implementation is:

```text
pyquda_measurement_utils.pion_qTMD_vibe_develop
```

This is the pion analogue of the proton qTMD workflow, but with pion-specific
quark and antiquark propagator conventions:

- `pos_boost` is used for the quark forward propagator.
- `neg_boost` is used for the antiquark/sink-side propagator.
- There is no U/D flavor split.
- There is no nucleon spin polarization projection.
- The current workflow computes connected diagrams only.

## Perlmutter Workflow

Main Python entry point:

- `perlmutter/Pyquda_pion_TMD_CG.py`

Convenience wrapper:

- `perlmutter/run_pion_TMD_CG.sh`

Batch wrapper:

- `perlmutter/submit_pion_TMD_CG.sh`

For login-node smoke tests, use the run wrapper directly:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/pion_TMD_CG/perlmutter
PION_TMD_QMAX=0 PION_TMD_BZ=0 PION_TMD_BT=0 PION_TMD_NUM_SRC=1 PION_TMD_T_INSERT=2 ./run_pion_TMD_CG.sh
```

For scheduled production runs, submit the batch wrapper:

```bash
sbatch submit_pion_TMD_CG.sh
```

## Default Environment

The run wrapper activates the validated Perlmutter PyQUDA/QUDA environment:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
```

It defaults to the bundled S8T32 test gauge:

```text
/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

Outputs default to:

```text
perlmutter/data
```

The local `perlmutter/.gitignore` ignores generated data, logs, and QUDA/CuPy
caches.

## Main Parameters

Common environment-variable overrides:

- `PION_TMD_GAUGE_PATH`: gauge-file path.
- `PION_TMD_DATA_DIR`: output directory.
- `PION_TMD_CONFIG_NUM`: configuration number.
- `PION_TMD_MPI_GEOMETRY`: MPI geometry, for example `1.1.1.1`.
- `PION_TMD_NUM_SRC`: number of source positions.
- `PION_TMD_QMAX`: momentum range `[-qmax, qmax]`.
- `PION_TMD_BZ`: maximum straight z separation.
- `PION_TMD_BT`: maximum transverse separation for qTMD.
- `PION_TMD_T_INSERT`: fixed sink-source separation.
- `PION_TMD_WIDTH`: Gaussian smearing width.
- `PION_TMD_SRC_INTERPOLATOR`: source gamma convention, default `fixed_g5`.
- `PION_TMD_SINK_INTERPOLATOR`: sink interpolator used for the sequential
  source, default `5`.

The application also accepts matching command-line arguments.

## Current Outputs

The application writes HDF5 correlators only.

Two-point functions:

```text
data/c2pt/<lat>.c2pt.<cfg>.<ama>.<src>.<sm>.h5
```

Three-point qTMD/PDF functions:

```text
data/qTMD/<lat>.qTMD.<cfg>.<kind>.ex.<src>.<sm>.<pf_tag>.<gamma>.h5
```

Current `kind` values:

- `CG`: coordinate-gauge style qTMD displacement without explicit gauge links.
- `GI_PDF`: straight-z PDF with gauge-covariant link shifts.
- `CG_PDF`: straight-z PDF with ordinary coordinate shifts.

HDF5 layout follows the shared qTMD writer convention:

```text
SS/<gamma>/PX<p_x>PY<p_y>PZ<p_z>/<b_X or b_Y>/eta0/bT<b_T>/bz<b_z>
```

For PDF outputs, `bT = 0` and the path is stored under `b_X`.

## Validated Smoke Test

A minimal S8T32 smoke test has been run on Perlmutter `login32` with:

```text
PION_TMD_QMAX=0
PION_TMD_BZ=0
PION_TMD_BT=0
PION_TMD_NUM_SRC=1
PION_TMD_T_INSERT=2
PION_TMD_WIDTH=1.0
```

This test verifies:

- forward propagator inversion,
- pion two-point contraction,
- meson fixed-sink sequential propagator inversion,
- connected CG qTMD contraction,
- connected GI_PDF and CG_PDF contraction,
- HDF5 output for all 16 gamma structures.

For nonzero Wilson-line validation, run an additional smoke test with
`PION_TMD_BZ=1`.
