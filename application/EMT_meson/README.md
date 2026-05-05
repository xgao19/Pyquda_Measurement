# EMT Meson Applications

This directory contains platform-specific entry scripts for the EMT meson
measurement workflows.

## Layout

- `perlmutter/`: current Perlmutter scripts used for the active EMT meson work.
- `frontier/`: older Frontier-oriented scripts kept as a reference.

## Perlmutter Workflow

The Perlmutter scripts are thin application entry points around
`pyquda_measurement_utils.EMT_meson_vibe_develop`.

Current measurements:

- `Pyquda_EMT_gluon_1pt.py`: flowed gluon EMT one-point function.
- `Pyquda_EMT_quark_1pt.py`: stochastic flowed quark EMT one-point function.
- `Pyquda_EMT_quark_3pt.py`: meson two-point function and connected quark EMT
  three-point function.

Convenience wrappers:

- `run_gluon_1pt.sh`
- `run_quark_1pt.sh`
- `run_quark_3pt.sh`

Batch wrappers:

- `submit_gluon_1pt.sh`
- `submit_quark_1pt.sh`
- `submit_quark_3pt.sh`

## Environment

On Perlmutter, activate the validated PyQUDA/QUDA environment with:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
```

The scripts default to the bundled S8T32 test gauge:

```text
/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

Outputs default to:

```text
/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data
```

You can override paths with environment variables such as `EMT_DATA_DIR` and
`EMT_GAUGE_PATH`.

## Current Conventions

- Output is HDF5-only.
- Meson two-point data are saved under `EMT2pt`.
- Quark connected three-point data are saved under `EMT3pt`.
- The active connected three-point contraction convention is convention B with
  `meson_sign = 1`.
- Quark and gluon gradient-flow schedules measure first, then flow; `step = 0`
  is the unflowed measurement.

For detailed run history, validated baselines, and physics conventions, read:

```text
/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/SESSION_MEMORY.md
```
