# Pion EMFF Background-Response Diagnostic

This directory contains a minimal pion EMFF background-field linear-response
diagnostic.  It does not modify the QUDA Dirac operator.  Instead it constructs
the first-order response propagator

```text
S_response = D^{-1} Gamma_T Phi_q S
```

and compares its pion two-point-like contraction with the summed explicit
EMFF three-point function.

For a true finite-difference derivative of `(D + lambda O)^{-1}`, multiply this
response by an additional overall minus sign.

Default smoke kinematics:

```text
pf = [0, 0, 0]
qext = [0, 0, 0]
pi = pf - qext = [0, 0, 0]
current = T
source gamma = gamma5
sink gamma = gamma5
tsep = 2
```

Useful optional controls:

```bash
export PION_EMFF_BG_QEXT_LIST="0.0.0;0.0.1;0.0.2"
export PION_EMFF_BG_TSEP_LIST="2.4"
export PION_EMFF_BG_CURRENT_GAMMAS="T.Z.T5"
export PION_EMFF_BG_TAU_WINDOW="restricted"
export PION_EMFF_BG_TAU_MIN="1"
```

The saved file uses schema version 2.  Each record stores `pf`, `qext`,
`pi = pf - qext`, `tsep`, the current gamma, the tau-window definition, C2,
the explicit summed C3, the response C2-like correlator, and both summed
ratios.  The `summary/` group also stores table-like arrays for `rel_diff`,
`response_R_sum`, `explicit_R_sum`, momenta, gamma labels, and window labels so
analysis code can scan all records directly.  The implementation intentionally
does not save per-tau response propagators.

Run on a GPU node, for example `login32`:

```bash
bash application/EMFF_pion_background_response/perlmutter/run_pion_EMFF_background_response.sh
```

The output HDF5 file is written under `data/background_response/`.
