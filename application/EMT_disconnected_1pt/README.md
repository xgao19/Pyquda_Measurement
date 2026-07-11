# EMT Disconnected One-Point Workflows

This application measures hadron-independent flowed one-point building blocks
used for EMT disconnected diagrams.
The Perlmutter entry points are:

```bash
bash perlmutter/run_quark_1pt.sh
bash perlmutter/run_gluon_1pt.sh
```

Quark production runs default to base-oriented shards.  A measurement job
writes only recoverable parts; after every requested base is complete, run:

```bash
bash perlmutter/run_finalize_quark_1pt.sh
```

The finalizer validates every base/HP interval before atomically publishing the
canonical EMTc and kinetic-only FlowedQuarkRinged files.

The default smoke-test gauge is:

```text
Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

## Observables

The quark workflow estimates the stochastic loop

```text
L_q,munu(q,tau;t_f) = sum_x exp(i q.x) Tr_sc[eta^\dagger(x,t_f) Gamma_munu xi(x,t_f)]
```

where `t_f` is the gradient-flow time.  The implementation stores the
symmetrized upper-triangle components `mu <= nu` of `Tmunu`; the missing lower
triangle is obtained by symmetry in analysis.

The quark workflow also stores `CHI` as a scalar trace and stochastic-noise
diagnostic.  In the same run it derives the ringed-fermion kinetic trace from
the zero-momentum diagonal EMT components and writes a kinetic-only
`FlowedQuarkRinged` companion.  This reuses the exact EMT stochastic vectors
and adds no inversion, fermion-flow, derivative, or MPI-gather work.  The
standalone `application/flowed_quark_ringed_norm` workflow remains available
for dedicated high-statistics and resumable measurements.

The gluon workflow stores the flowed gluonic EMT building block

```text
L_g,munu(q,t_f) = sum_x exp(i q.x) O_g,munu(x,t_f)
```

again with only the upper triangle written.

## Disconnected Diagram Combination

For a hadron two-point function `C2_H(p,t)` and a one-point loop
`L_munu(q,tau;t_f)`, the disconnected three-point building block is formed in
analysis as

```text
C3_disc,munu(pf,pi;t,tau;t_f)
  = < C2_H(pf,t) L_munu(q,tau;t_f) >cfg
    - < C2_H(pf,t) >cfg < L_munu(q,tau;t_f) >cfg
```

with

```text
q = pf - pi
```

The corresponding ratio is usually built as

```text
R_disc,munu(t,tau;t_f) = C3_disc,munu(t,tau;t_f) / C2_H(t)
```

up to the same kinematic, renormalization, and gradient-flow matching factors
used for the connected EMT analysis.  Vacuum subtraction must be performed at
the ensemble-analysis level because the one-point function is hadron
independent.

## Hierarchical Probing

The quark workflow supports two stochastic-source schemes:

```text
EMT_1PT_NOISE_SCHEME=zn
EMT_1PT_NOISE_SCHEME=hierarchical_probing
```

The default base source is full-volume counter-based `Z4` noise.  Its fixed
hash key contains the global space-time coordinate, spin, color, configuration,
base-noise index, and `EMT_1PT_RAND_SEED` stream salt, so the global source is
unchanged when the MPI decomposition changes.  `EMT_1PT_N_ZN` defaults to `4`
and the stream salt defaults to `0`.

Do not replace this generator with identical calls to `xp.random.seed` on all
MPI ranks.  Equal-shaped local lattices would receive the same local noise,
creating artificial cross-rank correlations.  Adding the rank to the seed is
still decomposition dependent; global-coordinate counters are required for
production reproducibility.

For hierarchical probing,

```text
effective_n_inversions = n_base_noise * hp_num_vectors
```

where `n_base_noise` is `EMT_1PT_N_VEC` and `hp_num_vectors` is
`EMT_1PT_HP_NUM_VECTORS`.  `hp_num_vectors` must be a positive power of two.

Site-orderings currently available for quark HP include:

```text
EMT_1PT_HP_ORDERING=interleaved_xyz_binary_projected_to_evenodd
EMT_1PT_HP_ORDERING=interleaved_xyzt_binary_projected_to_evenodd
EMT_1PT_HP_ORDERING=global_xyzt_gray_projected_to_evenodd
EMT_1PT_HP_ORDERING=spatial_xyz_then_t_gray_projected_to_evenodd
```

The default is `interleaved_xyz_binary_projected_to_evenodd`.  The full-volume
base noise still has independent counter values at every time coordinate; only
the optional HP sign pattern is time independent.  The 4D orderings remain
available for direct variance comparisons.

No spin-color dilution or time dilution is currently implemented in this
workflow.

## Why the Source Remains Four Dimensional

The code flows the noise and solution together,

```text
xi_f  = K(t_f) xi
eta_f = K(t_f) D^{-1} xi
```

where `K(t_f)` is the four-dimensional gauge-covariant fermion-flow kernel.
For the projector `P_tau` onto one absolute insertion time, the estimator is

```text
L_hat(tau,t_f) = xi^dag K^dag P_tau Gamma K D^{-1} xi
E[L_hat]       = Tr[P_tau Gamma K D^{-1} K^dag].
```

`P_tau` keeps the physical output resolved in time, so the observable still
sums only over space.  The initial source is nevertheless full-volume because
fermion flow spreads in all four Euclidean directions with characteristic
radius about `sqrt(8*t_f)`.  A source restricted to one initial time projector
generally omits finite-flow contributions.  A complete time-dilution basis
would remain unbiased after summing every projector, but would require the
corresponding extra inversions.  The detailed derivation is in
`docs/EMT_disconnected_1pt/EMT_disconnected_1pt.tex`.

Spatial HP does not change this conclusion: its sign pattern is independent of
time, but it still multiplies a full-volume `Z4` base source that is nonzero on
every time slice.  Isotropic 4D HP also remains full-volume and differs only in
how probing signs are assigned.

## HDF5 Layout

Quark output:

```text
attrs/
  measurement
  flow_type, flow_epsilon, flow_steps, flow_times
  qext, pf, p_2pt
  volume_norm
  upper_triangle_only
  mass, csw, tol, maxiter
  n_vec, n_base_noise, effective_n_inversions
  n_zn, config_num, rand_seed, noise_stream
  noise_generator, noise_counter_order
  noise_scheme, hp_num_vectors, hp_ordering

raw/Tmunu_pervec
raw/CHI_pervec
raw/source_index
raw/base_noise_index
raw/hp_index

avg/CHI
avg/Tmunu/T11, T12, ..., T44
```

Canonical quark-loop files are source independent and use
`EMTc/<lat>.EMTc.<cfg>.<ama>.<sm>.h5`.  A single full-time loop file is shared
by all hadron two-point source times on that configuration.

The same run also writes
`FlowedQuarkRinged/<lat>.FlowedQuarkRinged.<cfg>.<ama>.<sm>.h5`.  This companion
is the kinetic-only subset of the standalone schema:

```text
flow_times
raw/kinetic_pervec
raw/source_index
raw/base_noise_index
raw/hp_index
raw/spin_index
raw/color_index
avg/kinetic_spacetime
```

It intentionally omits `avg/Z_ring_field_sqrt` and `avg/Z_ring_bilinear`.
Form the ensemble mean of `avg/kinetic_spacetime` first and apply the ringed
factor formula only afterward; averaging configuration-local inverse factors
would be biased.  The identity

```text
raw/kinetic_pervec = -2/Vs * sum_mu raw/Tmunu_pervec[:,mu,mu,q0,:,:]
```

is an exact file-level cross-check.  `qext` must contain exactly one zero
momentum when companion output is enabled.

The bookkeeping datasets mean:

```text
source_index      effective source index after HP expansion
base_noise_index  original stochastic base-noise index
hp_index          hierarchical-probing vector index for that base noise
```

Gluon output:

```text
attrs/
  measurement
  flow_type, flow_epsilon, flow_steps, flow_times
  qext, pf, p_2pt
  volume_norm
  upper_triangle_only

Tmunu/T11, T12, ..., T44
```

## Minimal HP Smoke Test

Example:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/EMT_disconnected_1pt/perlmutter

EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
EMT_1PT_N_VEC=1 \
EMT_1PT_NOISE_SCHEME=hierarchical_probing \
EMT_1PT_HP_NUM_VECTORS=2 \
EMT_1PT_HP_ORDERING=interleaved_xyz_binary_projected_to_evenodd \
bash run_quark_1pt.sh
```

Expected checks:

```text
attrs/noise_scheme = hierarchical_probing
attrs/hp_num_vectors = 2
attrs/effective_n_inversions = 2
raw/source_index = [0, 1]
raw/base_noise_index = [0, 0]
raw/hp_index = [0, 1]
```

## Base Shards And Resume

Production controls are:

```text
EMT_1PT_OUTPUT_MODE=base_shards
EMT_1PT_BASE_START=0
EMT_1PT_BASE_STOP=EMT_1PT_N_VEC
EMT_1PT_BLOCK_INTERVAL_SOLVES=64
EMT_1PT_SHARD_DIR=<data>/EMTc/shards
```

Each part is named with its base, part, and half-open HP interval, for example
`base000003.part0001.hp0064-0127.h5`.  Existing compatible parts are skipped;
missing parts are computed.  A base completion marker is written only after
all its parts are reopened and validated.  Partial HP parts are checkpoints,
not complete estimators.  Jobs may process non-overlapping base ranges in
parallel, but overlapping ranges are unsupported.

`EMT_1PT_OUTPUT_MODE=monolithic` retains the legacy library/output path for
small comparisons.  The production shard path never overwrites a conflicting
or corrupt part automatically.
