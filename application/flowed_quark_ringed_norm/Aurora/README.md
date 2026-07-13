# Aurora standalone ringed workflow

Run `run_flowed_quark_ringed_norm.sh --config_num 1000` to produce base/HP
shards. Configuration identity is required on the CLI and is not read from an
environment variable. Use `FLOWED_RINGED_BASE_START/STOP`,
`FLOWED_RINGED_SHARD_DIR`, and `FLOWED_RINGED_BLOCK_INTERVAL_SOLVES` for job
partitioning and checkpoints.

Finalize with `../finalize_ringed_shards.py`; compute factors from explicit
canonical configuration files with `../analyze_ringed_ensemble.py`. The
per-configuration file stores kinetic data only. Full conventions are in
`docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md`.

The old platform-specific HP convergence scripts were removed. Resume now
uses the shared fingerprinted base-level sample log and does not probe HDF5,
so logged shards may be transferred before later bases run.
