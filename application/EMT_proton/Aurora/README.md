# Aurora Proton EMT Smoke

This directory runs the connected proton quark EMT workflow on the bundled
`S8T32` test gauge using the Aurora PyQUDA `develop` environment and the
`dpnp` SYCL backend.  It is intended as the first smoke test before moving to a
large l64 gauge.

Run from an interactive allocation or an SSH shell on one allocated compute
node:

```bash
bash submit_or_run_interactive.sh --config_num 0
```

Defaults:

- `EMT_PROTON_NRANKS=2`
- `EMT_PROTON_MPI_GEOMETRY=1.1.1.2`
- configuration is required as `--config_num CFG`; it is never read from the environment
- `EMT_PROTON_LAT_TAG=S8T32`
- `EMT_PROTON_GAUGE_PATH=/lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0`
- `EMT_PROTON_QMAX=0`
- `EMT_PROTON_T_SEPS=2`
- `EMT_PROTON_FLOW_STEPS=1`
- `EMT_PROTON_WIDTH=1.0`
- `EMT_PROTON_GAUSS_SMEAR=0`
- `EMT_PROTON_MAXITER=300`

Outputs go under `data/`; logs go under `log/`; QUDA tuning/profile files go
under `.cache/`.

The connected code keeps its existing one-branch execution model. It performs
one initial full gauge setup, uses thin gauge updates before later source and
sequential inversions, and shares one flowed-gauge context across all
covariant derivatives at each output flow time. There is no connected branch
batch-size option.

For l64 fixed_GLU smoke tests, keep the same scripts and override the ensemble
settings from the launch environment, for example:

```bash
export EMT_PROTON_NRANKS=32
export EMT_PROTON_MPI_GEOMETRY=2.2.2.4
export EMT_PROTON_LAT_TAG=l64c64a076
export EMT_PROTON_GAUGE_PATH=/lus/flare/projects/StructNGB/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.1050.coulomb.1e-14
export EMT_PROTON_MG_BLOCK=none
export EMT_PROTON_T_SEPS=1
export EMT_PROTON_WIDTH=1.0
export EMT_PROTON_FLOW_STEPS=0
export EMT_PROTON_TOL=1e-2
export EMT_PROTON_MAXITER=50
bash submit_or_run_interactive.sh --config_num 1050
```

Those l64 values are smoke settings only; restore production tolerances,
source/sink separations, smearing width, and flow schedule for physics runs.
