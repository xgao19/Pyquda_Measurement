# EMT Disconnected One-Point Workflows

This application measures hadron-independent flowed one-point building blocks
used for EMT disconnected diagrams.
The Perlmutter entry points are:

```bash
bash perlmutter/run_quark_1pt.sh --config_num 1000
bash perlmutter/run_gluon_1pt.sh --config_num 1000
```

Quark production runs default to base-oriented shards.  A measurement job
writes only recoverable parts; after every requested base is complete, run:

```bash
bash perlmutter/run_finalize_quark_1pt.sh --config_num 1000
```

The finalizer validates every base/HP interval before atomically publishing one
canonical EMTc file, including its embedded ringed kinetic data.

The default smoke-test gauge is:

```text
Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

## Observables

The quark workflow estimates the stochastic loop

```text
L_q,munu(q,tau;t_f) = sum_x exp(i q.x) xi^dagger(x,t_f) Gamma_munu eta(x,t_f)
```

where `t_f` is the gradient-flow time.  The implementation stores the
symmetrized upper-triangle components `mu <= nu` of `Tmunu`; the missing lower
triangle is obtained by symmetry in analysis.

The identity scalar bilinear is stored once in the full local Gamma basis.  A
separate `flowed_noise_norm` dataset stores the flowed stochastic-source norm.
The same EMTc derives the ringed-fermion kinetic trace from the zero-momentum
diagonal derivative components under `derived/ringed`.  This reuses the exact EMT stochastic vectors
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

The memory-bounded loop readers and absolute-to-source-relative time alignment
used by `build_3pt` live in `application/analysis_helper`.  Application-level
data reading and observable assembly are kept there; production measurements,
operator contractions, and reusable computational infrastructure remain in
`pyquda_measurement_utils`.

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
  qext
  volume_norm
  emt_operator_schema_version
  gamma_basis_schema, gamma_basis_order
  mass, csw, tol, maxiter
  n_vec, n_base_noise, effective_n_inversions
  n_zn, config_num, noise_stream
  noise_generator, noise_counter_order
  noise_scheme, hp_num_vectors, hp_ordering

gamma_list, gamma_pyquda_ids, gamma_matrices
physical_gamma_list, physical_from_pyquda
derivative_directions

raw/local_bilinear_pervec
raw/derivative_bilinear_pervec
raw/flowed_noise_norm_pervec
raw/source_index
raw/base_noise_index
raw/hp_index

avg/flowed_noise_norm
avg/local_bilinear
avg/derivative_bilinear
avg/Tmunu/T11, T12, ..., T44
derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

The primitive shapes are

```text
raw/local_bilinear_pervec      [N_eff,16,Nq,Nflow,Nt]
raw/derivative_bilinear_pervec [N_eff,16,4,Nq,Nflow,Nt]
```

They contain the complete PyQUDA bit-mask basis in `gamma_list` order.  The
stored matrix `physical_from_pyquda` converts raw channels to a convention in
which every axial channel means `gamma_mu gamma5`; in particular raw `Y5` and
`T5` acquire a minus sign.  Raw tensor channels are
`[gamma_mu,gamma_nu]/2`; multiply them by `1j` for the Hermitian tensor
convention.  Primitive data are unsymmetrized and unrenormalized.

The historical EMT is a derived view.  Select raw vector channels in
`[X,Y,Z,T]` order and form

```text
B[nu,mu] = derivative_bilinear[gamma_nu,mu]
T[mu,nu] = (B[mu,nu] + B[nu,mu]) / 2
```

Only the ten upper-triangle averaged `Tmunu` datasets are duplicated for direct
EMT analysis; the large raw symmetric tensor is not stored.

Canonical quark-loop files are source independent and use
`EMTc/<lat>.EMTc.<cfg>.<ama>.<sm>.h5`.  A single full-time loop file is shared
by all hadron two-point source times on that configuration.
`build_3pt` converts both quark and gluon absolute-time loops to source-relative
time with `roll(time_axis, -source_t)` before constructing C3 or ratios.

The same EMTc contains the kinetic-only derived group:

```text
derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

It intentionally omits the ringed-field and ringed-bilinear factors.
Form the ensemble mean of `derived/ringed/kinetic_spacetime` first and apply the ringed
factor formula only afterward; averaging configuration-local inverse factors
would be biased.  The identity

```text
derived/ringed/kinetic_pervec = -2/Vs * sum_mu \
    raw/derivative_bilinear_pervec[:,gamma_mu,mu,q0,:,:]
```

is an exact file-level cross-check.  `qext` must contain exactly one zero
momentum for every EMT quark measurement.

The bookkeeping datasets mean:

```text
source_index      effective source index after HP expansion
base_noise_index  original stochastic base-noise index
hp_index          hierarchical-probing vector index for that base noise
```

Gluon output:

```text
EMTg/<lat>.EMTg.<cfg>.<ama>.<sm>.h5
attrs/
  measurement
  config_num
  flow_type, flow_epsilon, flow_steps, flow_times
  qext
  volume_norm
  upper_triangle_only

Tmunu/T11, T12, ..., T44
```

The gluon loop is source independent and is reused for every hadron source on
the same configuration.  Shared quark and gluon wrappers both read
`EMT_1PT_FLOW_EPSILON`, whose default is `0.207936`.

## Minimal HP Smoke Test

Example:

```bash
cd /global/cfs/cdirs/m4559/xgao/software_gradientflow/Pyquda_Measurement/application/EMT_disconnected_1pt/perlmutter

EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
EMT_1PT_N_VEC=1 \
EMT_1PT_NOISE_SCHEME=hierarchical_probing \
EMT_1PT_HP_NUM_VECTORS=2 \
EMT_1PT_HP_ORDERING=interleaved_xyz_binary_projected_to_evenodd \
bash run_quark_1pt.sh --config_num 1000
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
EMT_1PT_BASE_START=0
EMT_1PT_BASE_STOP=EMT_1PT_N_VEC
EMT_1PT_BLOCK_INTERVAL_SOLVES=64
EMT_1PT_SHARD_DIR=<data>/EMTc/shards
```

Each part is named with its base, part, and half-open HP interval, for example
`base000003.part0001.hp0064-0127.h5`. After every part of a base is atomically
written, rank 0 appends `base000003` to
`<data>/sample_log_disconnected/<canonical-stem>.log`. Resume reads only this
log and never probes HDF5, so logged shards may already have been transferred.
An unlogged base is recomputed from its first HP vector. Partial HP parts are
not independently resumable estimators. Jobs may process non-overlapping base
ranges in parallel, but overlapping ranges are unsupported.

There is no monolithic production mode.  Configuration identity is accepted
only through the required `--config_num` CLI option; it is never inferred from
an environment variable or silently defaulted to zero.  Small numerical references belong in
tests. The destination-side finalizer checks schema, metadata, HP coverage and
bookkeeping once while merging; it does not use the production sample log.
`config_num` is mandatory and HDF5 provenance stores only
`noise_stream`, not a duplicate `rand_seed` alias.
