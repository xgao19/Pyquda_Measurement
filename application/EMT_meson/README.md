# EMT Meson Applications

This directory contains platform-specific entry scripts for the EMT meson
measurement workflows.

## Layout

- `perlmutter/`: current Perlmutter scripts used for the active EMT meson work.
- `frontier/`: older Frontier-oriented scripts kept as a reference.

## Perlmutter Workflow

The Perlmutter scripts are thin application entry points around
`pyquda_measurement_utils.pion_EMT_vibe_develop`.

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

## Using 1pt Data For Disconnected Diagrams

The meson connected 3pt script saves the usual connected valence insertion,

```text
C3_conn(q, tau; pf, tsep)
  = < O_sink(pf, tsep) O_EMT(q, tau) O_src^\dagger(0) >_connected .
```

The quark and gluon 1pt scripts save operator loops on each gauge
configuration.  For a flowed operator `O(t_flow, q, tau)` these are
schematically

```text
L_q^{mu nu}(q, tau, t_flow)
  = sum_x Phi_q(x) O_q^{mu nu}(x, tau, t_flow),

L_g^{mu nu}(q, tau, t_flow)
  = sum_x Phi_q(x) O_g^{mu nu}(x, tau, t_flow).
```

The disconnected meson EMT three-point function is not just the ensemble
average of the 1pt loop.  It is the covariance between the hadron two-point
function and the loop, measured on the same gauge configuration:

```text
C3_disc^{mu nu}(q, tau; pf, tsep, t_flow)
  = < C2(pf, tsep) L^{mu nu}(q, tau, t_flow) >_cfg
    - < C2(pf, tsep) >_cfg < L^{mu nu}(q, tau, t_flow) >_cfg .
```

Here `L` may be a quark loop, a gluon loop, or the final renormalized
gradient-flow operator assembled from quark and gluon loops.  The subtraction
is the vacuum subtraction.  For nonzero momentum transfer it may vanish in the
infinite-statistics limit, but keeping it explicitly is safer and keeps the
analysis formula uniform.

The practical analysis order should be:

1. For each configuration, read the meson `EMT2pt` file and select the same
   source position, smearing, sink gamma, and final momentum used for the target
   matrix element.
2. Read the quark `EMTc` and/or gluon `EMTg` 1pt file from the same
   configuration.
3. Align the loop time with the source time.  If the source is at `t0`, use
   `tau = (t_loop - t0) mod Nt` before combining with a two-point function at
   sink separation `tsep`.
4. Form the per-configuration product `C2 * L` first.
5. Only after the per-configuration product is formed, take the ensemble
   average and subtract `<C2><L>`.
6. Add the connected contribution from `EMT3pt` only after the connected and
   disconnected pieces are in the same momentum, time, flow-time, and
   normalization convention.

The full meson EMT matrix element at the contraction level is therefore

```text
C3_total^{mu nu}
  = C3_conn,valence^{mu nu}
    + C3_disc,quark^{mu nu}
    + C3_disc,gluon^{mu nu},
```

before applying gradient-flow matching coefficients, trace terms, flavor
mixing, and any final ratio or fit procedure.

For the quark 1pt files, the diagonal datasets

```text
avg/Tmunu/T11, avg/Tmunu/T22, avg/Tmunu/T33, avg/Tmunu/T44
```

also reconstruct the flowed-fermion kinetic expectation value used for ringed
fermion normalization.  That normalization should be applied consistently to
both connected and disconnected quark EMT pieces at the same flow time.

## Future 1pt Variance Reduction

The current quark 1pt estimator uses plain stochastic noise.  Two planned
upgrade directions are:

- Hierarchical probing: arXiv:1302.4018,
  `https://arxiv.org/abs/1302.4018`.
- Frequency splitting / propagator-decomposition variance reduction:
  arXiv:2605.00643, `https://arxiv.org/abs/2605.00643`.

These are not implemented yet.  If they are added, the HDF5 metadata should
record the estimator type, probing depth or frequency split, and noise budget
so mixed-estimator disconnected analyses remain auditable.

For detailed run history, validated baselines, and physics conventions, read:

```text
/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/SESSION_MEMORY.md
```
