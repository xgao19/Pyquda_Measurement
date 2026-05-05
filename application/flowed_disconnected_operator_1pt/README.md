# Flowed Disconnected Operator 1pt Workflow

This workflow computes flowed one-point operator loops that can be used as
disconnected-diagram building blocks.

## Scope

Current operators:

- Stochastic flowed quark EMT-like loops: `Tmunu` and `CHI`.
- Flowed gluon EMT-like loops from clover field strengths.
- Ringed-fermion kinetic normalization can be reconstructed from the quark
  `Tmunu` diagonal trace at q=0.

This workflow does not combine the loops with hadron two-point functions, does
not perform vacuum subtraction, and does not apply renormalization or mixing
coefficients.  Those steps belong in downstream analysis.

## Perlmutter Entry Points

```bash
bash perlmutter/run_flowed_disc_quark_1pt.sh
bash perlmutter/run_flowed_disc_gluon_1pt.sh
```

The scripts default to the small S8T32 test gauge and write HDF5 output under
`perlmutter/data`.

Useful environment variables:

- `FLOWED_DISC_GAUGE_PATH`
- `FLOWED_DISC_DATA_DIR`
- `FLOWED_DISC_QMAX`
- `FLOWED_DISC_FLOW_STEPS`
- `FLOWED_DISC_FLOW_EPSILON`
- `FLOWED_DISC_N_VEC`
- `FLOWED_DISC_N_ZN`

## Output

Quark output uses the same layout as EMT quark 1pt:

```text
raw/Tmunu_pervec
raw/CHI_pervec
avg/Tmunu/T11 ... avg/Tmunu/T44
avg/CHI
```

Gluon output uses:

```text
Tmunu/T11 ... Tmunu/T44
```

Ringed-fermion kinetic normalization:

```text
sum_mu T_{mu mu}^q(q=0,t_flow,t)
```

from `avg/Tmunu/T11`, `T22`, `T33`, and `T44`.

