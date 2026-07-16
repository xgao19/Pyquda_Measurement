# Aurora standalone ringed workflow

Run `run_flowed_quark_ringed_norm.sh --config_num 1000` to produce base/HP
shards. Configuration identity is required on the CLI and is not read from an
environment variable. Use `FLOWED_RINGED_BASE_START/STOP`,
`FLOWED_RINGED_SHARD_DIR`, and `FLOWED_RINGED_BLOCK_INTERVAL_SOLVES` for job
partitioning and checkpoints.

Pass `--flow-batch-size B` to batch several source/solution pairs in one
double-precision fermion-flow call. The default is `B=1`; increase it only
after checking device-memory headroom. The value affects scheduling, not the
estimator or HDF5 provenance, and there is no automatic OOM fallback.

Finalize with `../finalize_ringed_shards.py`. The per-configuration file stores
kinetic data only; ensemble normalization is outside this production workflow.
Full conventions are in
`docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md`.

The old platform-specific HP convergence scripts were removed. Resume now
uses the shared fingerprinted base-level sample log and does not probe HDF5,
so logged shards may be transferred before later bases run.

Production uses the `RingedQuark1pt` kinetic-only subclass of the EMT shared
runner. The runner infrastructure is shared, while the standalone contraction
does not construct the full EMT primitive basis.
