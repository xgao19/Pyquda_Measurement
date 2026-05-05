# Pion EMFF Applications

This directory contains platform-specific entry scripts for the pion connected
electromagnetic form-factor workflow.

## Measurement Source

The active implementation is:

```text
pyquda_measurement_utils.pion_EMFF_vibe_develop
```

It uses the same meson fixed-sink sequential-source idea as the pion qTMD
workflow, but with a local current insertion and no spatial Wilson-line
separation.  The final-state pion momentum is set by `parameters["pf"]`; the
momentum-transfer list is set by `parameters["qext"]`.

## Perlmutter Workflow

Main Python entry point:

- `perlmutter/Pyquda_pion_EMFF.py`

Convenience wrapper:

- `perlmutter/run_pion_EMFF.sh`

Batch wrapper:

- `perlmutter/submit_pion_EMFF.sh`

For a login-node smoke test:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/EMFF_pion/perlmutter
PION_EMFF_QMAX=0 PION_EMFF_NUM_SRC=1 PION_EMFF_T_INSERT=2 ./run_pion_EMFF.sh
```

## Boost Parameters

The pion EMFF workflow supports independent quark and antiquark boosts at the
source and sink:

- `PION_EMFF_POS_BOOST_SRC`
- `PION_EMFF_POS_BOOST_SINK`
- `PION_EMFF_NEG_BOOST_SRC`
- `PION_EMFF_NEG_BOOST_SINK`

Use dot-separated triples, for example `1.0.0`.  If source and sink boosts are
equal, the behavior reduces to the older `pos_boost` / `neg_boost` convention.
For convenience, `PION_EMFF_POS_BOOST` and `PION_EMFF_NEG_BOOST` can be used as
fallbacks for both source and sink.

## Outputs

The application writes HDF5-only output:

```text
data/c2pt/
data/pion_EMFF/
```

EMFF datasets are stored as:

```text
SS/<gamma>/PX<q_x>PY<q_y>PZ<q_z>
```

The current workflow is connected-only and scans all 16 standard gamma
structures.
