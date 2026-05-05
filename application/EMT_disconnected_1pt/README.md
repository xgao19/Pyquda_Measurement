# EMT Disconnected 1pt Applications

This directory contains hadron-independent Perlmutter entry points for flowed
EMT one-point loops.  The same quark and gluon 1pt files can be combined with
pion, proton, or other hadron two-point functions in analysis to build
disconnected diagrams.

## Files

- `perlmutter/Pyquda_EMT_disconnected_quark_1pt.py`: stochastic quark EMT loops
  and ringed-fermion kinetic normalization data.
- `perlmutter/Pyquda_EMT_disconnected_gluon_1pt.py`: flowed gluon EMT loops.
- `perlmutter/run_quark_1pt.sh`: login-node smoke wrapper for the quark loop.
- `perlmutter/run_gluon_1pt.sh`: login-node smoke wrapper for the gluon loop.

## Analysis Use

For a hadron two-point function `C2_H` and an EMT loop `L`, form the
disconnected contribution configuration by configuration:

```text
C3_disc,H^{mu nu}(q, tau, t_flow)
  = < C2_H L^{mu nu}(q, tau, t_flow) >_cfg
    - < C2_H >_cfg < L^{mu nu}(q, tau, t_flow) >_cfg .
```

The product `C2_H * L` must be formed before the ensemble average.  The 1pt
loop is hadron independent; pion/proton differences enter through the chosen
two-point function, source/sink setup, spin projection, and final analysis
ratio or fit.

## Defaults

The Perlmutter scripts default to the S8T32 test gauge:

```text
/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

Outputs default to `perlmutter/data` unless `EMT_1PT_DATA_DIR` is set.

