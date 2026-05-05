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

## Using 1pt Data For Disconnected Diagrams

The proton quark 3pt script computes connected U/D insertions with fixed-sink
sequential propagators:

```text
C3_conn,f^{mu nu}(q, tau; pf, tsep, P)
  = < N(pf, tsep) O_f^{mu nu}(q, tau) \bar N(0) >_connected ,
```

where `f = U, D` and `P` denotes the selected spin projection.  Disconnected
quark and gluon EMT insertions are built in analysis from the separately saved
1pt loops and the proton two-point function.

For each gauge configuration, the quark and gluon 1pt measurements provide
flowed loop operators

```text
L_q^{mu nu}(q, tau, t_flow)
  = sum_x Phi_q(x) O_q^{mu nu}(x, tau, t_flow),

L_g^{mu nu}(q, tau, t_flow)
  = sum_x Phi_q(x) O_g^{mu nu}(x, tau, t_flow).
```

The disconnected proton three-point function is the gauge-ensemble covariance
between the proton two-point function and the loop:

```text
C3_disc^{mu nu}(q, tau; pf, tsep, P, t_flow)
  = < C2(pf, tsep, P) L^{mu nu}(q, tau, t_flow) >_cfg
    - < C2(pf, tsep, P) >_cfg < L^{mu nu}(q, tau, t_flow) >_cfg .
```

This subtraction is the vacuum subtraction.  It should be kept explicitly even
when symmetry suggests that the loop expectation value should vanish at
nonzero momentum transfer, because finite statistics and finite-volume effects
can leave a residual signal.

The important operational point is that the product is formed on each
configuration before the ensemble average.  A safe analysis pipeline is:

1. Read `EMTproton2pt` for the target source, smearing, polarization, sink
   separation, and final momentum.
2. Read the same-configuration `EMTc` and/or `EMTg` 1pt loop file.
3. Shift loop times relative to the proton source time, using
   `tau = (t_loop - t0) mod Nt`.
4. For each flow step, momentum transfer, Lorentz component, and insertion
   time, form `C2 * L` on that configuration.
5. Average `C2 * L`, `C2`, and `L` over configurations.
6. Build `C3_disc = <C2 L> - <C2><L>`.
7. Combine with the connected U/D proton EMT output only after all pieces use
   the same momentum, time, flow-time, normalization, and spin-projection
   conventions.

At the contraction-output level the total proton EMT correlator is

```text
C3_total^{mu nu}
  = C3_conn,U^{mu nu}
    + C3_conn,D^{mu nu}
    + C3_disc,quark^{mu nu}
    + C3_disc,gluon^{mu nu}.
```

The final physical EMT still requires the gradient-flow matching coefficients,
trace terms, flavor mixing, ringed-fermion normalization, and any chosen ratio
or fit strategy.  In the quark 1pt files, the diagonal datasets

```text
avg/Tmunu/T11, avg/Tmunu/T22, avg/Tmunu/T33, avg/Tmunu/T44
```

reconstruct the flowed-fermion kinetic expectation value used for ringed
fermion normalization.  The same normalization convention should be applied to
connected and disconnected quark EMT contributions at a fixed flow time.

## Future 1pt Variance Reduction

The disconnected quark EMT contribution is expected to be noise sensitive, so
the shared quark 1pt estimator is the natural place for variance-reduction
upgrades:

- Hierarchical probing: arXiv:1302.4018,
  `https://arxiv.org/abs/1302.4018`.
- Frequency splitting / propagator-decomposition variance reduction:
  arXiv:2605.00643, `https://arxiv.org/abs/2605.00643`.

These are roadmap items, not current features.  Future output should record the
estimator type and parameters in HDF5 metadata before the data are mixed with
plain-noise 1pt loops.
