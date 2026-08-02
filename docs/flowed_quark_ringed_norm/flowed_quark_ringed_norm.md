# Standalone flowed-quark ringed kinetic measurement

The standalone workflow measures the per-configuration kinetic trace

```text
K_r(t_f,tau) = 1 / Vs * sum_mu
               xi_r^dag gamma_mu (Dplus_mu-Dminus_mu) eta_r,
eta_r = D^-1 xi_r.
```

It uses full-volume, global-coordinate counter-based `Z4` noise. Plain noise,
4D HP16, and 4D HP256 are supported. Time and spin-color dilution are not part
of this workflow.

## Relation to disconnected EMT

`RingedQuark1pt` inherits the production runner from
`EMTDisconnectedQuark1pt`. The shared runner owns counter-noise generation,
base/HP-part scheduling, inversions, source batching, double-precision fermion
flow, one flowed-gauge context per flow time, sample-log resume, and atomic
shard writes.

The contraction remains independent and kinetic-only. It evaluates four
vector-diagonal covariant derivatives directly and never constructs the 16
local or `16x4` derivative EMT primitive arrays. For identical sources and
parameters it obeys

```text
K_r = -2 / Vs * sum_mu L_D[gamma_mu,mu](q=0).
```

This equality is both a convention statement and the primary numerical
cross-check against `EMTquarkLoop/derived/ringed`.

## Resume and batching

Each complete stochastic base is recorded in the shared fingerprinted text
sample log. Resume trusts the log and does not require already transferred
HDF5 parts to remain on the production filesystem. HP parts are checkpoints;
only a complete HP base is an estimator.

`--flow-batch-size B` controls how many `[xi,D^-1 xi]` pairs are passed to one
QUDA fermion-flow call. The default is one. It is a performance-only choice and
does not enter physics provenance or the sample-log fingerprint. Increase it
only after checking GPU memory; MPI jobs do not automatically recover from
OOM.

Detailed restore/inversion/flow/contraction/write timers are disabled by
default. Set `PYQUDA_MEASUREMENT_TIMERS=1` to print them.

## Finalization and schema

Finalize complete shards with

```bash
python application/flowed_quark_ringed_norm/finalize_ringed_shards.py \
  --shard-dir <dir> \
  --canonical-tag <tag-without-.h5> \
  --n-base-noise <N>
```

The canonical file contains only

```text
flow_times
raw/kinetic_pervec
raw/base_noise_index
raw/hp_index
avg/kinetic_spacetime
```

The effective source index is reconstructed as
`base_noise_index * hp_vectors_per_base + hp_index` and is not stored.

It stores no ringed factor. Any nonlinear normalization must be constructed in
downstream ensemble analysis after averaging `K` over gauge configurations;
never average configuration-local `1/K` values.

The removed spin-color-diluted schema, factor datasets, analyzer, and former
`.block*.h5` workflow are unsupported and are not migrated.
