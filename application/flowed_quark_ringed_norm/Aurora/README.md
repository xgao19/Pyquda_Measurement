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
