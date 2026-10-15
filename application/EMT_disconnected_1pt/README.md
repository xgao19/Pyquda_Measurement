# EMT Disconnected Quark One-Point Examples

Platform entry points:

- [Aurora strange HP512/SC12 benchmark](Aurora/README.md): the a076 benchmark's
  dpnp/SYCL driver, eight-node launcher, source timing and output validator.
- `perlmutter/`: configurable CUDA smoke/production examples described below.

Both paths use the measurement checkout containing the launcher by default,
not an implicitly selected original `Pyquda_Measurement` checkout. The Aurora
example retains the benchmark's a076 strange settings; it is not a replacement
for the small-lattice Perlmutter defaults. Successful source execution is not
equivalent to complete-base output validation or an independent physics check.

For the current EMT HP2/32/512 implementation, multiplier plans and complete-base
statistics, use [PROBING.md](PROBING.md). Operational examples below use the
new format. The explicitly marked historical l64 record describes old HP
settings and requires the original checkout.

This application benchmarks stochastic estimators for the flowed quark
one-point functions used in disconnected energy-momentum-tensor calculations.
The primary numerical question is how quickly pure stochastic noise, complete 4D HP2/HP32/HP512,
and shape-validated multiplier probing converge at the same Dirac-solve cost.

Two observables from the same canonical `EMTquarkLoop` file are used for the comparison:

```text
the embedded ringed-fermion kinetic estimator K
one selected EMT component, such as T44
```

The same inversions, fermion flow, and covariant derivatives produce both
observables. Comparing them is useful because `K` sums four vector-diagonal
channels, whereas one `Tmunu` component probes a more specific part of the raw
bilinear data.

This is a one-gauge stochastic-estimator benchmark. It tests the measurement
pipeline, source bookkeeping, and convergence at fixed inversion cost. It is
not a gauge-ensemble determination of a ringed normalization or a physical
disconnected matrix element.

The main workflow is quark-only. A proton two-point function and a quark
disconnected three-point building block can be produced as an optional final
step. Gluon one-point production is not needed for this benchmark.

## Further Reading

This README is the main operational guide. The following repository documents
provide the physics and convention details used by the code:

```text
docs/EMT_disconnected_1pt/EMT_disconnected_1pt.pdf
  Derives the stochastic flowed quark loop, explains why the source remains
  full-volume under four-dimensional fermion flow, and documents HP and shard
  bookkeeping.

docs/EMT_gamma_and_raw_bilinears.md
  Lists the exact 16-Gamma PyQUDA basis, axial signs, tensor convention, raw
  HDF5 axes, and examples for reconstructing Tmunu and other bilinears.

docs/EMT_disconnected_1pt/EMT_proton_disconnected_guide.md
  Gives the proton C2 and disconnected C3 conventions, including
  source-relative time alignment and ensemble vacuum subtraction.

docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md
  Describes the dedicated standalone ringed-normalization workflow. That
  workflow remains useful for high-statistics kinetic measurements, but it is
  not required here because EMTquarkLoop already contains the kinetic estimator.
```

## Relation to the Standalone Ringed-Norm Exercise

If you previously ran the standalone flowed-quark ringed-norm benchmark, much
of the stochastic workflow here will look familiar. Both calculations use:

```text
full-volume counter-based Z4 noise
the same base-noise and 4D HP bookkeeping
eta = D^{-1} xi
the same fermion flow applied to xi and eta
the same four covariant-derivative directions
base/HP-part shards and a base-level sample log
a separate destination-side finalizer
```

The important difference is the observable being measured. The standalone
workflow is specialized to the one spacetime-averaged kinetic trace needed for
ringed-field normalization,

```text
K(t_f) = (1 / V4) sum_x
         <bar_chi(t_f,x) overleftrightarrow_Dslash chi(t_f,x)>.
```

The EMT workflow instead keeps a complete time- and momentum-resolved local
bilinear basis,

```text
L_A(q,tau;t_f)       for 16 Gamma_A,
L^D_A,mu(q,tau;t_f)  for 16 Gamma_A and mu = X,Y,Z,T.
```

It uses the vector derivative channels to construct every symmetric `Tmunu`.
The same stored diagonal channels also give

```text
K_r(t_f,tau) = -2 / Vs * sum_mu
               L^D_[gamma_mu,mu],r(q=0,t_f,tau),
```

so the spacetime-averaged kinetic estimator is included without another solve,
flow, or derivative application. With identical action, flow, source, HP, and
normalization conventions, this embedded result is a direct cross-check of the
standalone kinetic contraction.

The practical differences are summarized below.

| aspect | standalone ringed norm | disconnected EMT quark 1pt |
|---|---|---|
| primary physics target | kinetic expectation `K`, followed by ensemble-level ringed normalization | local and one-derivative quark bilinears, including all `Tmunu` components |
| resolved variables | flow time, with absolute time retained in raw data and averaged for `K` | Gamma, derivative direction, momentum, flow time, and absolute insertion time |
| contraction per flow time | four vector-diagonal derivative terms summed directly | 16 local plus `16x4` derivative channels; `Tmunu` and `K` are derived from them |
| inversion, flow, and derivative count | one inversion and one common flow per source; four derivative directions | the same counts per source; extra work is Gamma contraction, momentum projection, and I/O |
| spin-color dilution | not used | optional complete point dilution (`spin_color_dilution=point`) |
| canonical output | kinetic-only `FlowedQuarkRinged` file | full primitive `EMTquarkLoop` file with averaged `Tmunu` and `derived/ringed` kinetic data |
| typical use | dedicated high-statistics normalization study | reuse one quark loop for several hadron source times and study EMT or other bilinears |

The EMT file is therefore much larger. Its 64 derivative channels dominate
the storage, while the embedded kinetic data are a small derived view. The
standalone calculation is preferable when only `K` is needed, particularly
for dedicated high statistics. The EMT workflow
is preferable when the inversions should also support `Tmunu`, axial
one-derivative operators, tensor currents, or later three-point analyses.

The optional EMT point spin-color dilution uses all 12 exact spin-color
projectors for every spatial probing vector. The estimator sums the 12
projector contributions and averages only over spatial probing vectors. One
complete spatial basis plus all 12 projectors remains one independent base;
individual projectors are not independent samples. The default `none` mode
preserves full spin-color stochastic noise and all existing output behavior.

There is also one historical interface difference. The earlier standalone
exercise described fixed-interval `.block*.h5` files and ringed factors formed
from individual block averages. Those files are no longer the current
production format. The present standalone and EMT workflows both use
base/HP-part shards and fingerprinted text sample logs. Their canonical
per-configuration files contain kinetic measurements, while physical ringed
factors must be computed only after averaging `K` over gauge configurations:

```text
average K over configurations first
then evaluate the nonlinear ringed-normalization formula
never average configuration- or block-local values of 1/K
```

This EMT benchmark still compares convergence using the embedded `K`, but it
does not publish a ringed factor and does not turn a one-gauge stochastic test
into a normalization measurement.

The current standalone implementation is a kinetic-only subclass of the EMT
shared production runner. It shares noise, base/HP scheduling, inversions,
fermion flow, resume, and shard I/O with EMT, while its independent contraction
computes only the four vector-diagonal terms and never allocates the full EMT
primitive basis.

## Physics Targets

The quark measurement saves the primitive local and one-derivative bilinears

```text
L_A(q,tau;t_f)       = sum_x exp(i q.x) bar_chi Gamma_A chi
L^D_A,mu(q,tau;t_f)  = sum_x exp(i q.x)
                       bar_chi Gamma_A overleftrightarrow_D_mu chi
```

for all 16 Gamma matrices and four derivative directions. The symmetric quark
EMT building block is derived from the four vector Gamma channels:

```text
B[nu,mu] = L^D_[gamma_nu,mu]
T[mu,nu] = 0.5 * (B[mu,nu] + B[nu,mu])
```

The benchmark usually examines Euclidean `T44`, often called `T00` after the
appropriate Euclidean-to-Minkowski interpretation. All stored primitives and
derived EMT components are bare, unringed, and unrenormalized.

The same EMTquarkLoop file also contains the ringed-fermion kinetic estimator

```text
K_r(t_f,tau) = -2 / Vs * sum_mu
               L^D_[gamma_mu,mu],r(q=0,t_f,tau)
```

under:

```text
derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

The file deliberately does not contain a per-configuration inverse ringed
factor. A physical factor must be formed from the configuration-averaged
kinetic expectation value, not by averaging configuration-local values of
`1/K`.

For disconnected loops the derivative primitive is not a one-sided
`xi_f^dag Gamma D_mu eta_f` contraction. With

```text
D_mu = (D_+mu - D_-mu) / 2
S_f  = K D^{-1} K^dag
P_qtau = fixed-time projector times the spatial Fourier phase
```

the stored closed-loop building block is

```text
L_A,mu(q,tau) = -1/2 Tr[
    P_qtau Gamma_A D_mu S_f + D_mu P_qtau Gamma_A S_f]
```

The leading minus is the closed-fermion-loop Wick sign. Production reconstructs
the second term without extra `covDev` calls by using
`Gamma_sharp = gamma5 Gamma_A^dag gamma5`:

```text
L_A,mu(q,tau) = -1/2 [
    A_A,mu(q,tau) - A_Asharp,mu(-q,tau).conj()]
```

The internal momentum list therefore includes any missing `-q`; the canonical
`qext` axis still contains only the momenta requested by the user. The old
one-sided shortcut fails for a spatial derivative when the corresponding
momentum component is nonzero. It also fails for `D_4` at fixed insertion time,
even at zero spatial momentum.

There is a useful but subtle validation alternative. One may calculate the
left term explicitly by applying `covDev` to the flowed noise:

```text
L_direct = -1/2 * [
    xi_f^dag P_qtau Gamma D_mu eta_f
  - (D_mu xi_f)^dag P_qtau Gamma eta_f]
```

Production instead evaluates

```text
L_gamma5 = -1/2 * [A_Gamma(q) - A_Gamma_sharp(-q).conj()]
```

Both are unbiased estimators of the same trace, but they are not the same
quadratic form for an individual noise vector. Their equivalence uses cyclic
permutation inside a trace, which cannot be applied inside a fixed-source
quantity `xi^dag M xi`. A numerical check must therefore compare the paired
ensemble difference with its SEM rather than demand source-by-source equality.
The S8T8 validation used 256 counter-Z4 sources, all 16 Gamma channels, four
derivative directions, every time slice, and `q=0,+/-x,+/-y,+/-z`. The direct
and reconstructed ensemble results differed globally by only `0.981` paired
standard errors at solver tolerance `1e-15`. One-rank and four-rank means
agreed at relative L2 approximately `2.0e-16`.

## Mathematical Problem

For one absolute insertion time `tau`, the flowed stochastic loop has the form

```text
xi_f  = K(t_f) xi
eta_f = K(t_f) D^{-1} xi

L_hat(tau,t_f) = xi^dag K^dag P_tau Gamma K D^{-1} xi
```

and its noise expectation is

```text
E[L_hat] = Tr[P_tau Gamma K D^{-1} K^dag].
```

`P_tau` keeps the output resolved in insertion time, so the physical observable
contains a spatial sum rather than a time trace. The initial noise is still
nonzero on the full four-dimensional lattice. The fermion-flow kernel spreads
in all four Euclidean directions with characteristic radius approximately
`sqrt(8*t_f)`, so restricting the initial source to one time slice would not
give the same finite-flow estimator.

The default noise is counter-based `Z4`. Each phase is a deterministic function
of:

```text
global x,y,z,t
spin and color
configuration number
base-noise index
noise-stream salt
```

This makes the global source independent of the MPI decomposition. Do not
replace it with identical calls to a backend RNG on every rank. Equal local
lattice shapes and equal seeds can produce repeated rank-local arrays and
incorrect cross-rank noise correlations. Adding the rank to a conventional
seed avoids literal repetition for one geometry but still changes the source
when the MPI decomposition changes.

## Code Estimator

For every effective source, the production code performs:

```text
1. Build one full-volume counter Z4 source xi.
2. Solve eta = D^{-1} xi.
3. Flow xi and eta through the same four-dimensional fermion-flow schedule.
4. Construct each covariant derivative direction once.
5. Contract all 16 local and 16x4 right-derivative Gamma channels.
6. Project onto the requested momenta and any internally required `-q`.
7. Complete the two-sided derivative with gamma5 hermiticity.
8. Write only the completed primitive data to an atomic shard part.
```

The default flow schedule is:

```text
flow_type    = wilson
flow_steps   = 1
flow_epsilon = 0.207936
flow_times   = [0, 0.207936]
```

Thus `flow_index=0` is unflowed and `flow_index=1` is the fixed positive-flow
entry used in the example convergence analysis.

The measurement requires exactly one `q=0` entry because the embedded kinetic
estimator is derived from that channel. The examples below use only `q=0` to
keep the output small.

## Hierarchical Probing

The current develop contract is documented in [PROBING.md](PROBING.md).
The shared Python ordering is `nested_xyzt_checkerboard_v1`; shell wrappers
forward an ordering only when explicitly supplied. Old orderings and HP16/HP256
are not supported by this checkout.

| method | probes per base | cancellation distance | required extent divisor |
|---|---:|---:|---:|
| Z_N | 1 | none | none |
| HP2 | 2 | 1 | 2 |
| HP32 | 32 | 3 | 4 |
| HP512 | 512 | 7 | 8 |

Each complete probing basis multiplies the same full-volume random base; this
is not time dilution. `EMT_1PT_N_VEC` counts independent bases. Multiplier plans
specify the number of Fourier probes instead of an HP count. Point spin-color
dilution, available through the EMT Python API, sums 12 channels per probe.

```text
base estimate = average over probes (sum over dilution channels)
solves per base = vectors_per_base * dilution factor
```

Only complete bases are independent statistical units. One base has undefined
SEM; a partial probing prefix must not be treated as another independent base.

## Shards, Sample Log, and Finalization

Use fresh output/setup identities for the new shard schema
`disconnected_1pt_base_parts_v4` and sample-log schema
`disconnected_sample_log_v2`, even for Z_N. Old files/logs are rejected; the
original checkout remains the reader for old experiments. Do not include the
planned base count in the identity, so additional bases can extend a run.

Part names use `.baseXXXXXX.partXXXX.probeSTART-END.h5`. Every EMT part stores
complex128 means in `shard_mean/local_bilinear`,
`shard_mean/derivative_bilinear`, and `shard_mean/flowed_noise_norm`.
`--save-raw-per-vector` additionally stores per-source diagnostics. The base
finalizer requires the means, validates all complete bases and weights each
part by its actual probe count. It does not reconstruct missing means from raw.

After writing every part of a base, production appends that base to the
fingerprinted sample log. Resume trusts this log without probing HDF5 files;
unlogged bases are recomputed. Assign only nonoverlapping base ranges to jobs.
Finalization independently validates destination-side files, not the log.

## Canonical HDF5 Layout

Default finalization writes `<canonical_tag>.bases.h5`:

```text
base/base_noise_index
base/local_bilinear
base/derivative_bilinear
base/Tmunu/T44
base/ringed/kinetic_spacetime
avg/...
sem_real/...
sem_imag/...
```

Observables are already volume normalized. Metadata stores one probe count,
`vectors_per_base`, plus scheme, dilution and the applicable HP ordering or
multiplier plan. Do not normalize these base values by volume again.

Calling the Python finalizer with `--raw-output` additionally writes the raw
canonical `<canonical_tag>.h5`, requiring measurement with
`--save-raw-per-vector`. Raw bookkeeping is `base_noise_index`, `probe_index`
and, for SC12, `spin_index` and `color_index`. This optional raw path is used
by per-source diagnostics and the disconnected 3pt builder; the new statistical
analyses use `.bases.h5` only.

## Quick Start

On an allocated GPU with the appropriate environment activated, explicitly
select the develop checkout and an unused output directory. The following uses
the Perlmutter wrapper; the [Aurora example](Aurora/README.md) supplies its own
environment, PBS allocation and GPU-tile binding.

```bash
export MEASUREMENT_ROOT=/path/to/Pyquda_Measurement_develop
export PYTHONPATH="$MEASUREMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ROOT=$MEASUREMENT_ROOT
APP=$ROOT/application/EMT_disconnected_1pt/perlmutter
WORK=/path/to/new_emt_probing_run
export EMT_1PT_DATA_DIR="$WORK/data"
DATA=$EMT_1PT_DATA_DIR
export EMT_1PT_GAUGE_PATH="$ROOT/test_gauge/S8T8_wilson_b6.0"
export EMT_1PT_LAT_TAG=S8T8
export EMT_1PT_SETUP_TAG=smoke_hp2_nested_v4
export EMT_1PT_QMAX=0
export EMT_1PT_QZ_MAX=0
export EMT_1PT_FLOW_STEPS=1
export EMT_1PT_N_VEC=1
export EMT_1PT_N_ZN=4
export EMT_1PT_NOISE_SCHEME=hierarchical_probing
export EMT_1PT_HP_NUM_VECTORS=2
export EMT_1PT_SPIN_COLOR_DILUTION=none
bash "$APP/run_quark_1pt.sh" --config_num 0 --mg-block 8.8.4.4
bash "$APP/run_finalize_quark_1pt.sh" --config_num 0
```

`config_num` is part of the counter-noise identity, even for a test gauge.
This run performs two inversions and produces:

```text
data/EMTquarkLoop/S8T8.EMTquarkLoop.0.0.smoke_hp2_nested_v4.bases.h5
```

It should contain `n_base_noise=1`, `vectors_per_base=2`,
`effective_n_inversions=2` and `base/base_noise_index=[0]`. Its SEM is NaN.
Rerunning the same measurement skips the completed base according to its log.
This checks the data path, not stochastic convergence. Actual GPU/MPI validation
of the new probing implementation remains pending.

### Full spin-color dilution

The Perlmutter driver now forwards `EMT_1PT_SPIN_COLOR_DILUTION` to the shared
measurement class (`none` by default, or `point` for all 12 spin-color
projectors). `EMT_1PT_N_VEC` counts independent bases and defaults to one in
both the driver and launcher. To use the benchmark's estimator on a separately
configured compatible lattice:

```bash
export EMT_1PT_NOISE_SCHEME=hierarchical_probing
export EMT_1PT_HP_NUM_VECTORS=512
export EMT_1PT_SPIN_COLOR_DILUTION=point
export EMT_1PT_N_VEC=1
export EMT_1PT_BLOCK_INTERVAL_SOLVES=768
export EMT_1PT_SETUP_TAG=strange_hp512_sc12_nested
```

This is 6144 solves per independent base, split into eight 768-solve shards.
The 12 projector contributions are summed; HP vectors are averaged. These
settings alone do not select the a076 gauge, mass, clover coefficient or flow
parameters: configure those separately, or use the Aurora example. For point
dilution use a shard interval divisible by 12. Leave `EMT_1PT_HP_ORDERING`
unset for the corrected library default. Use a new output/setup identity when
changing the estimator or action. `--save-raw-per-vector` remains opt-in;
default measurement shards and `.bases.h5` finalization are mean-only.

Perlmutter keeps finalization separate: after all requested bases finish, run
`run_finalize_quark_1pt.sh` with the same data/tag/base-count settings. Do not
finalize each partial base range independently against the full requested count.

## Optional Fermion-Flow Source Batching

The quark wrapper can flow several already-inverted stochastic sources in one
QUDA call:

```bash
bash "$APP/run_quark_1pt.sh" \
  --config_num 1050 \
  --mg-block '4.4.4.4;4.4.4.4' \
  --flow-batch-size 8
```

The default is `--flow-batch-size 1`, which preserves the lowest-memory
execution path. A batch of size `B` packs the fields in the order

```text
xi_1, eta_1, xi_2, eta_2, ..., xi_B, eta_B
```

and evolves all `2B` fields with one double-precision fermion-flow call. The
sources still use sequential inversions. The original gauge is restored once
at the start of the batch, and all sources at one flow time share the same
resident flowed-gauge context for their covariant derivatives.

Use the largest batch that leaves a safe GPU-memory margin. The current
`l64c64a076`, resident-multigrid test found these practical starting points:

```text
80 GB Perlmutter GPU: B=8
40 GB Perlmutter GPU: B=1
```

`B=16` exhausted an 80 GB GPU for that particular `64^4` setup with
double-precision flow and resident multigrid. These are measured settings for
one ensemble, not universal limits; a new lattice, multigrid hierarchy, or GPU
should be tested from `B=1` upward. QUDA out-of-memory errors normally terminate
the MPI job, so the production code deliberately has no automatic fallback.

Batching changes only scheduling. It is not stored as physics provenance, does
not enter the sample-log fingerprint, and may be changed when resuming or
between nonoverlapping base jobs. Plain `Z4` batching may span pending bases.
Hierarchical probing is batched only within one base and one shard part, so a
partial HP prefix never becomes a completed estimator. Shards and the sample
log are published at the same completion boundaries as for `B=1`.

## Fixed-Cost Benchmark

Keep the gauge, configuration, counter stream, Dirac parameters, flow schedule
and momenta identical. Use a distinct fresh setup tag for each scheme. At 2048
solves without spin-color dilution, the complete-base counts are:

| method | `EMT_1PT_HP_NUM_VECTORS` | `EMT_1PT_N_VEC` |
|---|---:|---:|
| Z_N | 1 (unused) | 2048 |
| HP2 | 2 | 1024 |
| HP32 | 32 | 64 |
| HP512 | 512 | 4 |

Four HP512 bases are too few for a firm variance-efficiency claim. Multiplier
and SC12 comparisons must also use actual solves per base, not just probe
count. Only complete bases contribute to uncertainty estimates.

For multiple jobs, keep `EMT_1PT_N_VEC` equal to the full count and assign
nonoverlapping half-open `EMT_1PT_BASE_START/STOP` ranges. For HP32 above,
`[0,32)` and `[32,64)` share one measurement identity and sample log. Finalize
only after all requested bases are available, using the same setup tag and
full base count.

## Convergence Analysis

The analysis reads finalized `.bases.h5` observables directly. It does not
regroup raw vectors or recognize old HP16/HP256 modes. Labels are arbitrary;
solve costs and dilution factors come from each file's metadata.

```bash
python "$ROOT/application/analysis_helper/emt_quark_1pt_convergence.py" \
  --input pure="$DATA/EMTquarkLoop/S8T8.EMTquarkLoop.0.0.benchmark_pure_v4.bases.h5" \
  --input HP32="$DATA/EMTquarkLoop/S8T8.EMTquarkLoop.0.0.benchmark_hp32_v4.bases.h5" \
  --input HP512="$DATA/EMTquarkLoop/S8T8.EMTquarkLoop.0.0.benchmark_hp512_v4.bases.h5" \
  --flow_index 1 --component T44 --output_dir "$WORK/analysis"
```

The output includes cumulative/endpoint CSV tables and PNG/PDF convergence
plots for ringed kinetic and the requested EMT component. Compare real means,
imaginary parts, SEM and SEM-squared ratios at matched solve counts. A changing
SEM with few bases is not by itself a code error. These are fixed-gauge
stochastic uncertainties, not gauge-ensemble errors or renormalized results.

## From S8T8 to the l64 Ensemble

The following is a historical record from the old implementation. Its HP256
shards and measurements are not new nested-HP results; inspect or resume them
only with the original checkout and its matching environment.

The S8T8 workflow above is the safest place to learn the source bookkeeping,
shard layout, finalizer, and complete-base convergence analysis. After it
works, change the ensemble-dependent inputs rather than copying the S8T8
physics parameters into a production-size run.

For the current `l64c64a076` ensemble and configuration 1050, the two test
masses are:

| flavor label | bare mass `am_q` | run directory |
|---|---:|---|
| strange | `-0.015` | `EMT_disconnected_1pt_cfg1050_strange_am_m0p015` |
| light | `-0.049` | `EMT_disconnected_1pt_cfg1050_light_am_m0p049` |

These values are specific to this ensemble. They are not general strange- and
light-quark defaults and must not be reused for another ensemble without
checking its tuned action parameters.

The fixed gauge used by both tests is:

```text
/global/cfs/cdirs/m5208/xgao/ensembles/
l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/
l6464f21b7130m00119m0322a.1050.coulomb.1e-14
```

The prepared run roots are:

```text
/global/cfs/cdirs/m5208/xgao/runs/l64c64a076/
  EMT_disconnected_1pt_cfg1050_strange_am_m0p015/
  EMT_disconnected_1pt_cfg1050_light_am_m0p049/
```

The strange directory preserves the previous `am_q=-0.015` benchmark, tuning
cache, logs, and partial HP256 shards. Its base zero is not complete: the
sample log has no completed-base line, so a resume recomputes that entire base.
The light directory deliberately starts without HDF5 data, a sample log, MG
results, or a tuning cache.

Before changing from S8T8 to either l64 test, check every item in this list:

```text
gauge path and configuration number
lattice tag and lattice dimensions
quark flavor label and bare mass
clover coefficient, tolerance, and maximum iterations
MPI geometry and local lattice dimensions
run, data, shard, sample-log, log, and tuning-cache roots
multigrid hierarchy
counter stream, total base count, and scheduled base range
flow schedule, momenta, HYP preprocessing, and operator schema
```

On four Perlmutter GPU nodes, request and verify the allocation with:

```bash
salloc -N 4 -q interactive -t 04:00:00 \
  -C gpu --gpus-per-node=4 -A m5208_g

cd /global/cfs/cdirs/m5208/xgao/runs/l64c64a076/\
EMT_disconnected_1pt_cfg1050_light_am_m0p049
cp local_software.example.sh local_software.sh
# Edit local_software.sh to activate your environment and point to your
# Pyquda_Measurement checkout and QUDA install.
./preflight.sh
```

The intended layout is 16 MPI ranks, four ranks per node, one A100 per rank,
MPI geometry `2.2.2.2`, and a local lattice of `32^4` per GPU.

It is also valid to skip the S8T8 exercise and begin directly with the l64
test. In that case, do not skip the following staged checks:

1. Run the 16-rank GPU-binding and local-lattice preflight.
2. Benchmark `8.8.4.4`, `4.4.4.4`, and `4.4.4.4;4.4.4.4` at the selected mass.
3. Exclude the first autotuning solve and inspect the later solve times.
4. Complete one real pure-Z4 base.
5. For a new-format run, complete one nested HP32 base.
6. For a new-format run, complete one nested HP512 base.
7. Inspect timing, convergence, peak memory, shards, and sample logs before
   starting the full 2048-solve comparison.

Changing the mass changes the Dirac operator and multigrid behavior. Never
reuse the strange MG result or QUDA tuning cache as the official light
benchmark. Use completely separate data, sample-log, cache, and result
directories even though the gauge and configuration are the same.

The first solve includes QUDA autotuning and is a warmup. For subsequent
performance work, record at least:

```text
gauge read and HYP preprocessing
multigrid setup
Dirac gauge load or restore
inversion
fermion flow
local and derivative bilinear contractions
HDF5 shard write
```

The inversion time alone is not the EMT per-source cost. In the initial l64
strange test, repeated gauge restoration after fermion flow and the 64
derivative channels were important enough to time separately. Optimize and
retest this breakdown before committing a long light-quark production run.

## Reconstructing the Embedded Kinetic Estimator

This raw diagnostic requires measurement with `--save-raw-per-vector` and
Python finalization with `--raw-output`. New base analyses do not need it.

The following file-level check rebuilds the embedded kinetic data from the raw
vector-diagonal derivative channels:

```python
import h5py
import numpy as np

with h5py.File(path, "r") as h5:
    labels = [value.decode() for value in h5["gamma_list"][...]]
    vector = [labels.index(label) for label in ("X", "Y", "Z", "T")]
    qext = np.asarray(h5.attrs["qext"])
    q0 = np.flatnonzero(np.all(qext[:, :3] == 0, axis=1)).item()
    derivative = h5["raw/derivative_bilinear_pervec"]
    diagonal = sum(
        derivative[:, vector[mu], mu, q0, :, :]
        for mu in range(4)
    )
    expected = -2 * diagonal / h5.attrs["volume_norm"]
    np.testing.assert_allclose(
        expected,
        h5["derived/ringed/kinetic_pervec"],
    )
```

To reconstruct every averaged `Tmunu` without loading the source axis:

```python
with h5py.File(path, "r") as h5:
    labels = [value.decode() for value in h5["gamma_list"][...]]
    vector = [labels.index(label) for label in ("X", "Y", "Z", "T")]
    derivative = h5["avg/derivative_bilinear"][...]
    B = np.take(derivative, vector, axis=0)  # [nu,mu,q,flow,t_abs]
    T = 0.5 * (B + np.swapaxes(B, 0, 1))
    np.testing.assert_allclose(T[3, 3], h5["avg/Tmunu/T44"][...])
```

`avg/derivative_bilinear` is already averaged over effective sources and
divided by the spatial volume. Error analysis must instead use the raw source
axis and form complete base averages first.

## Optional Proton C2 and Quark Disconnected C3

The optional builder below uses undiluted raw canonical files. Produce those
with `--save-raw-per-vector` followed by Python finalization with `--raw-output`;
it is not a reader for `.bases.h5` or SC12 raw files.

The canonical quark loop is independent of the hadron source position and
source time. One full-time EMTquarkLoop file can therefore be reused for several C2
sources on the same gauge configuration.

Produce one proton C2 with the same gauge and momentum setup:

```bash
EMT_1PT_DATA_DIR="$DATA" \
EMT_1PT_GAUGE_PATH="$ROOT/test_gauge/S8T8_wilson_b6.0" \
EMT_1PT_LAT_TAG=S8T8 \
EMT_1PT_QMAX=0 \
EMT_1PT_SRC_POS=0.0.0 \
EMT_1PT_SRC_T=0 \
EMT_DISC_P2PT_QMAX=0 \
bash "$APP/run_proton_2pt.sh" --config_num 0
```

Then combine the C2 with one canonical quark loop. The builder is quark-only by
default and does not require a gluon file:

```bash
EMT_1PT_DATA_DIR="$DATA" \
EMT_1PT_LAT_TAG=S8T8 \
EMT_1PT_SETUP_TAG=benchmark_pure \
EMT_1PT_SRC_POS=0.0.0 \
EMT_1PT_SRC_T=0 \
bash "$APP/run_build_disconnected_3pt.sh" --configs 0 \
  --t_separations 2
```

EMTquarkLoop stores loops at absolute lattice time. Before forming the product, the
builder reads the full source position from the C2 file and converts both the
spatial Fourier origin and time coordinate:

```text
L_x0(q,tau_rel) = exp[-2 pi i q.(x0-o)/L]
                  * L_o(q,(source_t + tau_rel) mod Nt)
L_rel(tau_rel) = L_abs((source_t + tau_rel) mod Nt)
```

New quark and gluon loop files record the global lattice size, Fourier origin,
phase convention, and absolute-time convention. Analysis rejects old loops
without this provenance instead of guessing a phase.

Connected pion/proton runs use the same explicit multigrid option:

```bash
--mg-block 8.8.4.4
--mg-block '4.4.4.4;4.4.4.4'
--mg-block none
```

The shared default is one level `8.8.4.4`. The chosen hierarchy is written to
the connected C2/C3 attributes.

For one configuration, the output is only an unsubtracted `C2 * loop`
diagnostic. The physical disconnected building block requires a gauge ensemble
covariance:

```text
C3_disc = <C2 L>_cfg - <C2>_cfg <L>_cfg.
```

Run the builder with an explicit configuration list after producing matching
C2 and EMTquarkLoop files for every configuration:

```bash
bash "$APP/run_build_disconnected_3pt.sh" --configs 100,102,104
```

Stochastic bases on one configuration are not substitutes for independent
gauge configurations in the vacuum subtraction.

Gluon data can be included later with:

```bash
bash "$APP/run_build_disconnected_3pt.sh" \
  --configs 100,102,104 \
  --include_gluon
```

That advanced path requires matching canonical gluon files and matching
momentum and flow-time axes. It is outside the quark estimator benchmark.

## Code Map

The Perlmutter application layer contains:

```text
run_quark_1pt.sh
  Activates the current PyQUDA/QUDA environment, supplies stable defaults, and
  calls the quark production driver.

Pyquda_EMT_disconnected_quark_1pt.py
  Parses the explicit configuration, constructs qext/flow/noise parameters,
  reads and preprocesses the gauge, and starts base-range production.

run_finalize_quark_1pt.sh
Pyquda_EMT_disconnected_finalize_quark_1pt.py
  Stream shards into one canonical EMTquarkLoop file.

run_proton_2pt.sh
Pyquda_EMT_disconnected_proton_2pt.py
  Optional proton C2 measurement.

run_build_disconnected_3pt.sh
Pyquda_EMT_disconnected_build_3pt.py
  Optional quark-only C2/loop combination; gluon is explicitly opt-in.
```

Shared production implementation:

```text
pyquda_measurement_utils/Disconnected_1pt_EMT_vibe_develop.py
  Flowed quark contraction, shard production, and EMTquarkLoop finalizer.

pyquda_measurement_utils/Disconnected_utils_vibe_develop.py
  Counter noise, HP sign patterns, base/HP-part paths, atomic shard writes,
  sample-log resume, and streaming finalizer validation.

pyquda_measurement_utils/fermion_bilinear_basis.py
  Shared 16-Gamma basis and derived-operator helpers.
```

Analysis implementation:

```text
application/analysis_helper/emt_quark_1pt_convergence.py
  Complete-base ringed/Tmunu convergence tables and figures.

application/analysis_helper/emt_disconnected_analysis.py
  Memory-bounded quark/gluon loop readers and source-time alignment for C3.
```

The main data flow is:

```text
shell wrapper
  -> quark Python driver
  -> EMTDisconnectedQuark1pt.flowed_fermionic_1pt(...)
  -> atomic base/HP shards + sample log
  -> quark finalizer
  -> canonical EMTquarkLoop
  -> convergence helper
  -> CSV + PNG/PDF
```

## Reading Exercises

Read the code in this order:

1. `perlmutter/run_quark_1pt.sh`
2. `perlmutter/Pyquda_EMT_disconnected_quark_1pt.py`
3. `EMTDisconnectedQuark1pt.flowed_fermionic_1pt()`
4. the counter-noise and HP iterators in `Disconnected_utils_vibe_develop.py`
5. the primitive contraction methods in `Disconnected_1pt_EMT_vibe_develop.py`
6. `aggregate_emt_quark_1pt_bases()` (optional raw: `finalize_emt_quark_1pt_shards()`)
7. `application/analysis_helper/emt_quark_1pt_convergence.py`

Useful checks:

- Change the smoke from HP2 to HP32 with a new setup tag; confirm 32 solves
  and one independent base. Optional raw `probe_index` spans `0..31`.
- Run pure noise with `N_VEC=4`; confirm `base/base_noise_index=[0,1,2,3]`
  and `vectors_per_base=1`.
- Verify that changing `config_num`, `base_noise_index`, or `noise_stream`
  changes the counter source while rerunning the same tuple reproduces it.
- Reconstruct `T44` and the ringed kinetic from the raw derivative primitive.
- Compare ringed and `T44` relative SEM at the same solve count.
- Repeat one small run with a different valid MPI geometry and verify that the
  global counter source is unchanged after reassembly.
- Change the source time in the optional C2 workflow and verify the builder's
  absolute-to-relative time roll.

## Troubleshooting

If Python cannot import `pyquda`, activate the repository environment and check
the core packages:

```bash
source /path/to/your/python-environment/bin/activate
export MEASUREMENT_ROOT=/path/to/your/Pyquda_Measurement
python3 -c "import pyquda, cupy, h5py, mpi4py, matplotlib"
```

If no CUDA device is detected, verify that the process is running on a GPU node
or GPU-equipped login node and that one MPI rank is assigned to each GPU.

If finalization reports a missing part, compare:

```text
EMT_1PT_N_VEC
EMT_1PT_SETUP_TAG
EMT_1PT_SHARD_DIR
EMT_1PT_BLOCK_INTERVAL_SOLVES
the expected base and HP ranges
```

The finalizer intentionally does not trust the production sample log as proof
that destination-side shard files exist.

If a production rerun skips a base after its shards were moved, that is the
expected sample-log behavior. The log is the production-side resume record.
Use a new setup tag for a genuinely new run, or deliberately repair the log and
base range only after identifying where the transferred shards are stored.

If a sample-log fingerprint changes, do not mix the runs. Check the
configuration, counter stream, `Z_n`, HP ordering, flow schedule, physical
Dirac-action parameters (mass and `csw`), gauge preprocessing, momenta,
operator schema, and part interval. Multigrid blocks, solver tolerance and
maximum iterations are runtime controls rather than measurement identity, so
they may differ between resumed base ranges.

If HP512 seems worse than pure noise, first verify that:

```text
the ordering is the nested_xyzt_checkerboard_v1
every analyzed HP512 unit contains all 512 vectors
the x axis is solve count rather than source-row count
the comparison uses the same gauge, configuration, stream, flow, and momenta
there are enough complete HP512 bases to estimate a variance
```

The fixed-gauge benchmark measures stochastic convergence only. It does not
include gauge-ensemble fluctuations, operator mixing, ringed normalization,
matching, or renormalization.
