# Pion Soft Factor

This workflow is the PyQUDA-native replacement for the legacy mixed GPT/PyQUDA
soft-factor script:

```text
/global/cfs/cdirs/m3760/xgao/legacy_runs/l64c64a076_frontier/PyQUDA_qTMD_ff_4pt_einsum.py
```

The calculation is split into two stages because the legacy workflow first
generated Coulomb-gauge wall-source propagators on every time slice and then
performed the contractions from saved propagators.

## Perlmutter Scripts

```bash
cd Pyquda_Measurement/application/pion_soft_factor/perlmutter

# Generate wall propagators.
bash run_pion_soft_factor_prop.sh

# Contract the soft-factor four-point functions.
bash run_pion_soft_factor_contract.sh
```

The default scripts use the small `S8T32` test gauge and write under
`perlmutter/data`.  Production runs should override the gauge path, lattice tag,
MPI geometry, momentum list, and time-slice range through environment variables
or command-line arguments.

## Stages

`Pyquda_pion_soft_factor_prop.py`

- Loads the gauge field.
- Optionally Coulomb-gauge fixes it if requested.
- Applies one HYP smearing step.
- Builds wall sources with momentum phase `exp(+i k . x)`.
- Inverts and saves one HDF5 propagator per `(t_source, quark momentum)`.

`Pyquda_pion_soft_factor_contract.py`

- Reads saved wall propagators at the source and sink time slices.
- Builds the wall-to-wall two-point and qTMDWF diagnostic contractions.
- Contracts the soft-factor four-point function with the same gamma/interpolator
  defaults as the legacy script.
- Saves HDF5 output through the shared `io_corr.py` writer.

The contraction stage writes three output families:

- `pion_soft_factor_c2pt`: wall-to-wall pion two-point diagnostics.
- `pion_soft_factor_qTMDWF`: wall-source qTMDWF diagnostics used for analysis
  and renormalization.
- `pion_soft_factor`: the soft-factor four-point functions.

## Validation Status

This is a first PyQUDA port of the legacy method.  It should be treated as a
develop version until it is compared numerically against the legacy GPT output
on the same gauge, source time, momenta, gamma choices, and `bT/tsep` ranges.
