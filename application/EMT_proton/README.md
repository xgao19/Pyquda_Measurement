# Proton EMT Perlmutter Workflow

This directory contains Perlmutter entry points for proton EMT measurements.

## Files

- `perlmutter/Pyquda_EMT_proton_quark_3pt.py`: connected proton U/D quark EMT three-point functions.
- `perlmutter/Pyquda_EMT_proton_quark_1pt.py`: stochastic quark one-point functions and ringed-fermion normalization data.
- `perlmutter/Pyquda_EMT_proton_gluon_1pt.py`: flowed gluon one-point EMT building blocks.
- `perlmutter/run_*.sh`: login-node smoke-test wrappers.
- `perlmutter/submit_*.sh`: Perlmutter batch wrappers.

## Defaults

The scripts default to the local S8T32 smoke-test gauge:

```text
Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

Outputs are written under `perlmutter/data` unless `EMT_PROTON_DATA_DIR` is set.

## Main Parameters

- `EMT_PROTON_QMAX`: builds `qext = [-qmax, qmax]^3`.
- `EMT_PROTON_PF`: final proton momentum as `px.py.pz`.
- `EMT_PROTON_T_SEPS`: comma-separated sink separations, for example `2,3`.
- `EMT_PROTON_POL`: comma-separated polarization names from `bw_seq_pyquda.PolProjections`.
- `EMT_PROTON_INTERPOLATOR`: proton interpolator, for example `5`, `T5`, or `Z5`.
- `EMT_PROTON_FLOW_EPSILON`: gradient-flow step size.
- `EMT_PROTON_FLOW_STEPS`: number of output flow steps.

The connected three-point output stores U and D insertions on the first axis:

```text
flavor axis: 0 = U, 1 = D
```

