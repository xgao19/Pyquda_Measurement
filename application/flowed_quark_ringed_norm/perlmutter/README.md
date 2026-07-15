# Perlmutter standalone ringed workflow

Run `run_flowed_quark_ringed_norm.sh --config_num 1000` to produce base/HP shards. Useful controls
are `FLOWED_RINGED_N_VEC`, `FLOWED_RINGED_BASE_START/STOP`,
`FLOWED_RINGED_SHARD_DIR`, `FLOWED_RINGED_BLOCK_INTERVAL_SOLVES`, noise/HP
settings. Configuration identity is required on the CLI and is not read from
an environment variable.

Fermion-flow source batching is optional:

```bash
run_flowed_quark_ringed_norm.sh --config_num 1000 --flow-batch-size 4
```

The default is `1`. Larger values flow several `[xi, D^-1 xi]` pairs in one
double-precision QUDA call and reuse one flowed-gauge context per output flow
time. Use the largest value that leaves safe GPU-memory headroom; an OOM
terminates the MPI job and is not retried automatically. The batch size is a
performance choice only, so it is not stored as physics provenance and may be
changed between non-overlapping base jobs.

For the cfg1050 light-quark \(64^4\) test on 16 A100 GPUs, the warmed median
costs were 5.31, 4.32, 3.88, and 3.64 seconds/source for `B=1,2,4,8`.
Measured device-memory use was 26.6, 28.7, 34.0, and 44.8 GiB/GPU. Thus `B=8`
is the tested 80-GB starting point. `B=4` is the practical 40-GB starting
point based on its measured 34.0-GiB footprint, but it should still be checked
with a short local smoke because the lattice, multigrid hierarchy, and QUDA
build all affect memory. These values are guidance, not automatic defaults.

Finalize with `../finalize_ringed_shards.py`; combine explicit canonical
configuration files with `../analyze_ringed_ensemble.py`. Per-configuration
files are kinetic-only. See
`docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md` for schema and
normalization rules.

The former `.block*.h5` format is not supported. Resume uses the same
fingerprinted base-level text log as EMT and qTMD and deliberately does not
probe HDF5 files, allowing completed shards to be transferred immediately.
