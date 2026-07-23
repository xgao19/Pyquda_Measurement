# EMT Disconnected Quark One-Point Benchmark on Perlmutter

This application benchmarks stochastic estimators for the flowed quark
one-point functions used in disconnected energy-momentum-tensor calculations.
The primary numerical question is how quickly pure stochastic noise, 4D HP16,
and 4D HP256 converge at the same Dirac-solve cost.

Two observables from the same canonical `EMTc` file are used for the comparison:

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
  not required here because EMTc already contains the kinetic estimator.
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
| spin-color dilution | not used | not used |
| canonical output | kinetic-only `FlowedQuarkRinged` file | full primitive `EMTc` file with averaged `Tmunu` and `derived/ringed` kinetic data |
| typical use | dedicated high-statistics normalization study | reuse one quark loop for several hadron source times and study EMT or other bilinears |

The EMT file is therefore much larger. Its 64 derivative channels dominate
the storage, while the embedded kinetic data are a small derived view. The
standalone calculation is preferable when only `K` is needed, particularly
for dedicated high statistics. The EMT workflow
is preferable when the inversions should also support `Tmunu`, axial
one-derivative operators, tensor currents, or later three-point analyses.

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

The same EMTc file also contains the ringed-fermion kinetic estimator

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

The workflow supports:

```text
EMT_1PT_NOISE_SCHEME=zn
EMT_1PT_NOISE_SCHEME=hierarchical_probing
```

One full-volume random `Z4` source is called a `base`. Pure stochastic noise
uses one effective vector per base. Hierarchical probing multiplies the same
random base by a deterministic sequence of HP sign vectors:

```text
L_base = (1 / N_HP) sum_h L_[base,h]
```

The solve count is:

```text
effective solves = N_VEC * HP_NUM_VECTORS
```

where `N_VEC` is the number of randomized bases, not the number of effective
sources.

The three comparison cases are:

| case | noise scheme | `N_HP` | solves per complete base |
|---|---|---:|---:|
| pure Z4 | `zn` | 1 | 1 |
| Z4 + HP16 | `hierarchical_probing` | 16 | 16 |
| Z4 + HP256 | `hierarchical_probing` | 256 | 256 |

The default ordering is:

```text
interleaved_xyzt_binary_projected_to_evenodd
```

It resolves nearby sites in all four Euclidean directions. The HP signs are
multiplied by a full-volume random base; HP16 and HP256 are not time-diluted
sources.

Only complete HP bases are independent stochastic estimator units. An HP256
prefix containing 16, 64, or 128 vectors is a checkpoint, not an additional
unbiased sample for the uncertainty estimate. Consequently, a 256-solve HP256
run with one base can test the data path but cannot provide a base-level SEM.

Useful fixed-cost choices are:

| total solves | pure bases | HP16 bases | HP256 bases | interpretation |
|---:|---:|---:|---:|---|
| 512 | 512 | 32 | 2 | pipeline and preliminary comparison |
| 2048 | 2048 | 128 | 8 | initial fixed-gauge comparison |
| 8192 | 8192 | 512 | 32 | more stable HP256 variance estimate |

## Shards, Sample Log, and Finalization

Production writes base/HP-part shards rather than one monolithic HDF5 file.
With the default 64-solve part interval, example names are:

```text
<canonical-stem>.base000003.part0000.hp0000-0063.h5
<canonical-stem>.base000003.part0001.hp0064-0127.h5
```

The canonical stem should describe the estimator and physics parameters, not
the planned base count. Do not add tags such as `N32`; the base and part indices
already support extending the same measurement with additional bases.

Every part stores the arithmetic mean over its HP interval in:

```text
shard_mean/local_bilinear
shard_mean/derivative_bilinear
shard_mean/flowed_noise_norm
```

This mean-only layout is the default. Pass `--save-raw-per-vector` to also
store the existing `raw/*_pervec` datasets and source bookkeeping. The mean is
accumulated in complex128 and `shard_mean_vector_count` records its weight.
Tmunu is not duplicated; derive it from `shard_mean/derivative_bilinear` with
`emt_tensor_from_derivative_bilinear()` after adding a length-one source axis.

Each complete part is first written to a temporary file and then atomically
renamed. After every part of one base has closed successfully, rank 0 appends
one exact line to:

```text
<data>/sample_log_disconnected/<canonical-stem>.log
```

For example:

```text
# disconnected_sample_log_v1 sha256=<run-fingerprint> canonical=<stem>
base000000
base000001
```

Production resume reads only this text log. A logged base is skipped without
checking whether its shard files are still present, so completed shards may be
transferred immediately. An unlogged base is recomputed from its first HP
vector. Part-level resume inside a base is intentionally not supported.

The finalizer is a separate destination-side operation for shards produced
with `--save-raw-per-vector`. It does not read the sample log. It validates the
complete base/HP layout and metadata while streaming all parts into one
canonical file. Mean-only shards intentionally fail with a clear unsupported
payload error in the current release:

```text
EMTc/<lat>.EMTc.<cfg>.<ama>.<setup-tag>.h5
```

Missing parts, incomplete HP intervals, mixed parameters, or an incompatible
operator schema cause finalization to fail before publication. An existing
canonical file is not replaced by an incomplete result.

Nonoverlapping base ranges can be processed by separate jobs. Two jobs must
not compute the same base range.

## Canonical HDF5 Layout

The finalized quark file contains:

```text
attrs/
  measurement, config_num
  mass, csw
  flow_type, flow_epsilon, flow_steps, flow_times
  qext, volume_norm
  loop_provenance_schema, global_lattice_size, momentum_phase_origin
  spatial_momentum_phase_convention, loop_time_convention
  n_zn, noise_stream, noise_generator, noise_counter_order
  noise_scheme, n_base_noise, hp_num_vectors, hp_ordering
  effective_n_inversions
  emt_operator_schema_version
  gamma_basis_schema, gamma_basis_order

gamma_list
gamma_pyquda_ids
gamma_matrices
physical_gamma_list
physical_from_pyquda
gamma5_hermiticity_partner
gamma5_hermiticity_sign
derivative_directions

raw/local_bilinear_pervec
raw/derivative_bilinear_pervec
raw/flowed_noise_norm_pervec
raw/base_noise_index
raw/hp_index

avg/local_bilinear
avg/derivative_bilinear
avg/flowed_noise_norm
avg/Tmunu/T11 ... T44

derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

The shard-mean shapes are:

```text
shard_mean/local_bilinear      [16,q,flow,t_abs]
shard_mean/derivative_bilinear [16,4,q,flow,t_abs]
shard_mean/flowed_noise_norm   [q,flow,t_abs]
```

When `--save-raw-per-vector` is enabled, the additional raw shapes are:

```text
raw/local_bilinear_pervec      [source,16,q,flow,t_abs]
raw/derivative_bilinear_pervec [source,16,4,q,flow,t_abs]
raw/flowed_noise_norm_pervec   [source,q,flow,t_abs]
```

The bookkeeping datasets mean:

```text
base_noise_index  randomized full-volume source index
hp_index          HP sign-vector index within that base
```

The effective source index is not stored. Reconstruct it as
`base_noise_index * hp_vectors_per_base + hp_index`, where
`hp_vectors_per_base` is 1 for plain noise and `hp_num_vectors` for HP.

The large per-source derivative primitive controls the file size. For the
current 8192-solve, nine-momentum S8T8 file, the measured complete-file shares
are approximately:

| part | file share |
|---|---:|
| `raw/derivative_bilinear_pervec` | 78.886% |
| `raw/local_bilinear_pervec` | 19.722% |
| `raw/flowed_noise_norm_pervec` | 1.233% |
| `derived/ringed/kinetic_pervec` | 0.137% |
| all averaged primitives and `avg/Tmunu` | 0.014% |
| metadata, bookkeeping, and HDF5 overhead | 0.008% |

## Quick Start

First run a two-solve HP smoke test on the repository S8T8 gauge:

```bash
ROOT=${MEASUREMENT_ROOT:?Activate your PyQUDA environment and set MEASUREMENT_ROOT}
APP=$ROOT/application/EMT_disconnected_1pt/perlmutter
WORK=/global/cfs/cdirs/m5208/xgao/runs/TEST/emt_disconnected_quark_demo
DATA=$WORK/data
mkdir -p "$DATA"

EMT_1PT_DATA_DIR="$DATA" \
EMT_1PT_GAUGE_PATH="$ROOT/test_gauge/S8T8_wilson_b6.0" \
EMT_1PT_LAT_TAG=S8T8 \
EMT_1PT_SETUP_TAG=smoke_hp2 \
EMT_1PT_QMAX=0 \
EMT_1PT_QZ_MAX=0 \
EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_N_VEC=1 \
EMT_1PT_N_ZN=4 \
EMT_1PT_NOISE_SCHEME=hierarchical_probing \
EMT_1PT_HP_NUM_VECTORS=2 \
bash "$APP/run_quark_1pt.sh" \
  --config_num 0 \
  --mg-block 8.8.4.4 \
  --flow-batch-size 1 \
  --save-raw-per-vector
```

`--config_num` is required even for the test gauge. It is part of the
counter-noise identity and is never inferred from an environment variable.

Expected production files are:

```text
$DATA/EMTc/shards/
  S8T8.EMTc.0.0.smoke_hp2.base000000.part0000.hp0000-0001.h5

$DATA/sample_log_disconnected/
  S8T8.EMTc.0.0.smoke_hp2.log
```

The log should contain `base000000`. Running the same command again should skip
that base without opening its shard HDF5 file.

Finalize the smoke output with the same data directory, lattice tag, setup tag,
configuration, and total base count:

```bash
EMT_1PT_DATA_DIR="$DATA" \
EMT_1PT_LAT_TAG=S8T8 \
EMT_1PT_SETUP_TAG=smoke_hp2 \
EMT_1PT_N_VEC=1 \
bash "$APP/run_finalize_quark_1pt.sh" --config_num 0
```

Expected canonical output:

```text
$DATA/EMTc/S8T8.EMTc.0.0.smoke_hp2.h5
```

Inspect the main metadata and bookkeeping:

```bash
H5="$DATA/EMTc/S8T8.EMTc.0.0.smoke_hp2.h5"
PY=${PYTHON:-python3}

"$PY" - "$H5" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as h5:
    for key in (
        "config_num", "n_zn", "noise_scheme", "hp_num_vectors",
        "n_base_noise", "effective_n_inversions", "hp_ordering",
        "emt_operator_schema_version", "flow_times", "qext",
    ):
        print(f"{key}: {h5.attrs[key]}")
    print("base_noise_index:", h5["raw/base_noise_index"][...])
    print("hp_index:        ", h5["raw/hp_index"][...])
    print("kinetic shape:   ", h5["derived/ringed/kinetic_pervec"].shape)
    print("T44 shape:       ", h5["avg/Tmunu/T44"].shape)
PY
```

The smoke output should report:

```text
n_base_noise           = 1
hp_num_vectors         = 2
effective_n_inversions = 2
base_noise_index       = [0, 0]
hp_index               = [0, 1]
```

This smoke test checks the environment, source bookkeeping, shard writer, and
finalizer. It is too small to compare stochastic convergence.

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

The following shell function runs one method through the normal production
wrapper. All methods use the same gauge, configuration, counter stream, Dirac
parameters, flow schedule, momentum grid, and 4D HP ordering.

```bash
run_method () {
  label=$1
  scheme=$2
  hp=$3
  bases=$4
  batch=${5:-1}
  EMT_1PT_DATA_DIR="$DATA" \
  EMT_1PT_GAUGE_PATH="$ROOT/test_gauge/S8T8_wilson_b6.0" \
  EMT_1PT_LAT_TAG=S8T8 \
  EMT_1PT_SETUP_TAG="$label" \
  EMT_1PT_QMAX=0 \
  EMT_1PT_QZ_MAX=0 \
  EMT_1PT_FLOW_STEPS=1 \
  EMT_1PT_N_VEC="$bases" \
  EMT_1PT_N_ZN=4 \
  EMT_1PT_RAND_SEED=0 \
  EMT_1PT_NOISE_SCHEME="$scheme" \
  EMT_1PT_HP_NUM_VECTORS="$hp" \
  EMT_1PT_HP_ORDERING=interleaved_xyzt_binary_projected_to_evenodd \
  bash "$APP/run_quark_1pt.sh" \
    --config_num 0 \
    --mg-block 8.8.4.4 \
    --flow-batch-size "$batch" \
    --save-raw-per-vector
}
```

For a 512-solve pilot comparison:

```bash
run_method benchmark_pure  zn                       1   512
run_method benchmark_hp16  hierarchical_probing   16    32
run_method benchmark_hp256 hierarchical_probing  256     2
```

For a more useful 2048-solve comparison, change the base counts to:

```text
pure:   2048 bases
HP16:    128 bases
HP256:     8 bases
```

If several GPU jobs share one method, keep `EMT_1PT_N_VEC` equal to the full
base count in every job and assign nonoverlapping half-open intervals:

```text
EMT_1PT_BASE_START=<first base>
EMT_1PT_BASE_STOP=<one past the last base>
```

For example, two HP16 jobs for 128 total bases may use `[0,64)` and `[64,128)`.
They may share the same fingerprinted sample log. Overlapping ranges are not a
supported scheduling mode.

After all bases are complete, finalize each method:

```bash
finalize_method () {
  label=$1
  bases=$2
  EMT_1PT_DATA_DIR="$DATA" \
  EMT_1PT_LAT_TAG=S8T8 \
  EMT_1PT_SETUP_TAG="$label" \
  EMT_1PT_N_VEC="$bases" \
  bash "$APP/run_finalize_quark_1pt.sh" --config_num 0
}

finalize_method benchmark_pure  512
finalize_method benchmark_hp16   32
finalize_method benchmark_hp256   2
```

For the 2048-solve case, use the corresponding base counts instead.

## Convergence Analysis

The analysis helper reads finalized EMTc files and groups all HP vectors of one
randomized base before computing cumulative statistics. It reads only the one
or two vector-derivative channels needed for the selected symmetric `Tmunu`;
it does not load the full 16x4 primitive into memory.

Run:

```bash
PY=${PYTHON:-python3}

"$PY" "$ROOT/application/analysis_helper/emt_quark_1pt_convergence.py" \
  --input pure="$DATA/EMTc/S8T8.EMTc.0.0.benchmark_pure.h5" \
  --input HP16="$DATA/EMTc/S8T8.EMTc.0.0.benchmark_hp16.h5" \
  --input HP256="$DATA/EMTc/S8T8.EMTc.0.0.benchmark_hp256.h5" \
  --flow_index 1 \
  --component T44 \
  --output_dir "$WORK/analysis"
```

The outputs are:

```text
analysis/cumulative_statistics.csv
analysis/endpoint_summary.csv
analysis/ringed_kinetic_convergence.png
analysis/ringed_kinetic_convergence.pdf
analysis/t44_convergence.png
analysis/t44_convergence.pdf
```

Each figure contains:

```text
top:    cumulative real mean with a one-SEM band
bottom: SEM / abs(real mean), on a logarithmic scale
x axis: Dirac solves at complete-base boundaries
```

Use the same solve cost when comparing methods. For example, HP256 appears only
at multiples of 256 solves. A smooth-looking partial HP prefix is not a valid
additional base-level estimator.

Useful diagnostics include:

- whether the cumulative means from all methods approach compatible values;
- whether the imaginary parts in the CSV are consistent with the expected
  symmetries and fixed-gauge fluctuations;
- the relative SEM or SEM-squared ratio at matched solve counts;
- whether conclusions from the ringed kinetic agree with those from `T44`;
- how the HP256 uncertainty changes as the number of complete randomized bases
  increases.

An SEM that temporarily increases does not by itself indicate a code error,
especially when HP256 has only a few complete bases. Fixed-cost efficiency is
assessed from variance or SEM squared, not only from how smooth a cumulative
curve appears.

## From S8T8 to the l64 Ensemble

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
5. Complete one real 4D-HP16 base.
6. Complete one real 4D-HP256 base.
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

The canonical quark loop is independent of the hadron source position and
source time. One full-time EMTc file can therefore be reused for several C2
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

EMTc stores loops at absolute lattice time. Before forming the product, the
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
C2 and EMTc files for every configuration:

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
  Stream shards into one canonical EMTc file.

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
  Flowed quark contraction, shard production, and EMTc finalizer.

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
  -> canonical EMTc
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
6. `finalize_emt_quark_1pt_shards()`
7. `application/analysis_helper/emt_quark_1pt_convergence.py`

Useful checks:

- Change the smoke from HP2 to HP4 and confirm that `hp_index` becomes
  `[0,1,2,3]` while `base_noise_index` remains `[0,0,0,0]`.
- Run pure noise with `N_VEC=4` and confirm that `base_noise_index` is
  `[0,1,2,3]`, with `hp_index=0`; reconstruct source indices `[0,1,2,3]`.
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

If HP256 seems worse than pure noise, first verify that:

```text
the ordering is the default four-dimensional xyzt ordering
every analyzed HP256 unit contains all 256 vectors
the x axis is solve count rather than source-row count
the comparison uses the same gauge, configuration, stream, flow, and momenta
there are enough complete HP256 bases to estimate a variance
```

The fixed-gauge benchmark measures stochastic convergence only. It does not
include gauge-ensemble fluctuations, operator mixing, ringed normalization,
matching, or renormalization.
