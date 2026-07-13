# Perlmutter standalone ringed workflow

Run `run_flowed_quark_ringed_norm.sh --config_num 1000` to produce base/HP shards. Useful controls
are `FLOWED_RINGED_N_VEC`, `FLOWED_RINGED_BASE_START/STOP`,
`FLOWED_RINGED_SHARD_DIR`, `FLOWED_RINGED_BLOCK_INTERVAL_SOLVES`, noise/HP
settings. Configuration identity is required on the CLI and is not read from
an environment variable.

Finalize with `../finalize_ringed_shards.py`; combine explicit canonical
configuration files with `../analyze_ringed_ensemble.py`. Per-configuration
files are kinetic-only. See
`docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md` for schema and
normalization rules.

The former platform-specific HP convergence scripts, `.block*.h5` checkpoint
format, and text sample log were removed. Resume is now the same strict
base/HP-part validation used by EMT and qTMD.
