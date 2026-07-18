# Connected Proton qTMD/PDF

This application computes the connected proton C2 together with optional CG
qTMD, GI qTMD, CG PDF, and GI PDF contractions.  Perlmutter and Aurora call the
same production implementation in `shared_runner.py`; their thin entry points
only select backend and ensemble defaults.

## Entry points

Perlmutter:

```bash
bash application/nucleon_TMD/perlmutter/run_nucleon_TMD.sh \
  --config_num 1000 --mg-block 8.8.4.4
```

Aurora:

```bash
python application/nucleon_TMD/Aurora/Pyquda_nucleon_TMD.py \
  --stream b --config_num 1000 --mpi_geometry 1.5.4.5 \
  --mg-block 8.8.4.4
```

`--config_num` is mandatory and is never read from an environment variable.
Unknown command-line arguments are errors.  `--mg-block` accepts one or more
levels such as `8.8.4.4` or `4.4.4.4;4.4.4.4`; use `none` to disable
multigrid.  MG blocks, solver tolerance, and maximum iteration count are
runtime choices only.  They are not stored in HDF5 and do not enter the
sample-log identity.

The operator switches are `NUCLEON_TMD_RUN_CG_QTMD`,
`NUCLEON_TMD_RUN_GI_QTMD`, and `NUCLEON_TMD_RUN_PDF`.  The PDF switch produces
both CG and GI straight-link measurements.

The sink separation is supplied directly as `--t_separations 2` to either the
Python entrypoint or run wrapper.  Proton qTMD currently requires exactly one
separation per invocation; use separate runs for different separations.  The
former insertion-time option and environment-variable configuration are not
retained.

## Resume

The text sample log is the only resume state.  Rank 0 reads exact, non-empty
lines once at startup and broadcasts the completed source set.  A completed
source is skipped before source construction or inversion, even if its HDF5
files have already been transferred elsewhere.

A source is appended only after C2 and every product enabled for that run have
closed successfully.  The log does not contain a parameter fingerprint and the
code does not inspect HDF5 files.  Therefore a log may be reused only when the
enabled products, momentum grids, Wilson-line grids, sink separation,
interpolator, and smearing setup are unchanged.  Concurrent writers to the
same log are not supported.

## Conventions

`boost_in` is source smearing.  `boost_out` is C2 sink smearing and the
fixed-sink sequential smearing.  The current standard production setup uses
zero boosts at both endpoints.

The GI qTMD path uses the cached fixed-length staple convention shared with the
disconnected qTMD code.  The index is

```text
[b_T, b_z, eta, transverse_direction]
```

and the staple length is `2*eta+b_T`, independent of `b_z`.
