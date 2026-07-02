# Flowed-Quark Ringed Normalization on Aurora

This directory runs the standalone flowed-quark ringed-normalization workflow.
It produces `FlowedQuarkRinged/*.h5` files containing the kinetic expectation
value and ringed factors for flowed quark fields.  The output is reusable by EMT
and non-EMT flowed-quark operators with matching quark and flow conventions.

Small S8T32 smoke:

```bash
FLOWED_RINGED_FLOW_STEPS=1 \
FLOWED_RINGED_N_VEC=1 \
FLOWED_RINGED_NOISE_SCHEME=zn \
bash submit_or_run_interactive.sh
```

HP smoke:

```bash
FLOWED_RINGED_FLOW_STEPS=1 \
FLOWED_RINGED_N_VEC=1 \
FLOWED_RINGED_NOISE_SCHEME=hierarchical_probing \
FLOWED_RINGED_HP_NUM_VECTORS=2 \
bash submit_or_run_interactive.sh
```

S8T8 stochastic-vs-HP convergence benchmark from an interactive node:

```bash
bash run_s8t8_hp_convergence.sh
```

This runs four estimator cases:

```text
zn1024        pure stochastic, 1024 solves
hp64x16       HP16, 1024 solves
hp4x256       HP256, 1024 solves
hp6x16sc12    HP16 plus spin-color point dilution, 1152 solves
```

It then writes a matched-cost-style convergence summary and PDF under
`benchmark/s8t8_hp_convergence/`.

Important controls:

```text
FLOWED_RINGED_DATA_DIR
FLOWED_RINGED_GAUGE_PATH
FLOWED_RINGED_CONFIG_NUM
FLOWED_RINGED_MPI_GEOMETRY
FLOWED_RINGED_FLOW_TYPE
FLOWED_RINGED_FLOW_EPSILON
FLOWED_RINGED_FLOW_STEPS
FLOWED_RINGED_MASS
FLOWED_RINGED_CSW
FLOWED_RINGED_N_VEC
FLOWED_RINGED_N_ZN
FLOWED_RINGED_NOISE_SCHEME
FLOWED_RINGED_HP_NUM_VECTORS
FLOWED_RINGED_HP_ORDERING
FLOWED_RINGED_SPIN_COLOR_DILUTION
```

For l64 connected-production matching, use the same gauge preprocessing, mass,
clover coefficient, and flow schedule as the quark operator data being
normalized.

## l64c64a076 Real-Volume Benchmark Note

The scripts in this directory remain the S8T8/S8T32 Aurora application.  The
real l64 benchmark was staged outside the repository under:

```text
/lus/flare/projects/StructNGB/xgao/run/l64c64a076/EMT_proton/flowed_ringed_l64_prod256_benchmark/
```

The current partial convergence plot from that run is archived here:

```text
application/flowed_quark_ringed_norm/Aurora/l64_hp_convergence_prod256_results.pdf
```

That l64 test used config `1050` and the connected-proton production quark
inversion conventions:

```text
FLOWED_RINGED_LAT_TAG=l64c64a076
FLOWED_RINGED_CONFIG_NUM=1050
FLOWED_RINGED_FLOW_TYPE=symanzik
FLOWED_RINGED_FLOW_EPSILON=0.09
FLOWED_RINGED_FLOW_STEPS=1
FLOWED_RINGED_MASS=-0.049
FLOWED_RINGED_CSW=1.0372
FLOWED_RINGED_TOL=1e-10
FLOWED_RINGED_MAXITER=5000
FLOWED_RINGED_MG_BLOCK=4.4.4.4;4.4.2.2
FLOWED_RINGED_HYP_PROJECT=-1
FLOWED_RINGED_T_BOUNDARY=-1
FLOWED_RINGED_MPI_GEOMETRY=2.2.4.4
```

The production-queue benchmark was organized as 32 independent shards:

```text
4 estimator cases x 8 random-seed shards
8 Aurora nodes / 64 ranks per shard
256 nodes total
```

The estimator cases were:

```text
zn1024        pure stochastic, 1024 solves per seed shard
hp64x16       stochastic HP16, 1024 solves per seed shard
hp4x256       stochastic HP256, 1024 solves per seed shard
hp6x16sc12    HP16 plus spin-color point dilution, 1152 solves per seed shard
```

Current interval-block output always writes HDF5 files.  The interval length is controlled by:

```text
FLOWED_RINGED_BLOCK_INTERVAL_SOLVES=64
```

Each `.blockXXXX.srcSTART-END.h5` file contains one fixed interval.  The file
name and the range attrs `block_index`, `block_start`, and
`block_stop_exclusive` identify the interval.  The attrs
`estimator_complete`, `complete_estimator_units`, and `estimator_remainder`
state whether that interval is a complete estimator block.

The 7-hour prod job was killed by walltime after `07:01:28`.  The checkpoint
files preserved these completed solves:

```text
zn1024:       1536 / 8192 solves
hp64x16:      1792 / 8192 solves
hp4x256:      1536 / 8192 solves
hp6x16sc12:    384 / 9216 solves
```

All written block files confirm `mass=-0.049`, `csw=1.0372`, `tol=1e-10`, and
`maxiter=5000` in their HDF5 attributes.  The one fully completed shard,
`hp64x16_seed04`, took `21773.63 s` for `1024` solves, or about `21.3 s/solve`.
Across the 32 concurrent shards, however, the observed throughput varied by
more than an order of magnitude.  This is the main reason most shards did not
finish in 7 hours.

Current partial-data conclusion:

```text
At matched available solve counts, HP256 has the smallest observed block SEM.
HP16 is clearly better than pure stochastic.
HP16 plus spin-color dilution is not yet fairly comparable because only
384 completed solves were written.
```

Representative relative SEM values from the partial summary are:

```text
pure stochastic @ 1024 solves: 4.11e-5
HP16            @ 1024 solves: 2.46e-5
HP256           @ 1024 solves: 1.42e-5
HP16+SC         @  384 solves: 2.68e-5
```

The current numerical values are mutually consistent, with
`K_spacetime(flow=1) ~= 1.4189` and `Z_ring_bilinear(flow=1) ~= -3.3059`.
