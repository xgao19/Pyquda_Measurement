# Proton EMT on Aurora

This directory contains only the connected proton quark EMT workflow:

- proton two-point function
- connected quark EMT three-point function

It does not run quark/gluon one-point disconnected measurements.  The physics
contraction is implemented in `pyquda_measurement_utils/proton_EMT_vibe_develop.py`;
this directory only provides Aurora-oriented run wrappers and defaults.

Typical interactive/smoke usage after loading the Aurora PyQUDA environment:

```bash
bash run_proton_quark_3pt.sh
```

PBS usage:

```bash
qsub submit_proton_quark_3pt.sh
```

Important environment knobs:

- `EMT_PROTON_STREAM`: ensemble stream, default `b`
- `EMT_PROTON_CONFIG_NUM`: configuration number, default `220`
- `EMT_PROTON_MPI_GEOMETRY`: default `1.5.4.5`
- `EMT_PROTON_DATA_DIR`: output directory
- `EMT_PROTON_GAUGE_PATH`: gauge path template, may use `{stream}` and `{conf}`
- `EMT_PROTON_QMAX`: momentum-transfer cube half-width
- `EMT_PROTON_T_SEPS`: comma-separated source-sink separations
- `EMT_PROTON_FLOW_STEPS`: number of gradient-flow steps
