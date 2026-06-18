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

This runs 1024 pure stochastic solves and 64 base noises times 16
interleaved-4D HP vectors, then writes a matched-cost convergence summary under
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
```

For l64 connected-production matching, use the same gauge preprocessing, mass,
clover coefficient, and flow schedule as the quark operator data being
normalized.
