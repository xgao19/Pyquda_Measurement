# S8T8 Flowed-Quark Ringed-Norm HP Benchmark on Perlmutter

This application benchmarks stochastic trace estimators for one scalar quantity
needed by a lattice-QCD normalization workflow.  The physics motivation is
simple for this test: flowed quark fields need a normalization factor, and that
factor is computed from a kinetic expectation value called `K`.  Once that goal
is stated, the work here is mostly a numerical linear algebra and code
pipeline problem.

The benchmark compares three estimators on the same S8T8 test gauge:

```text
pure stochastic @ 1024 solves
stochastic HP16 @ 1024 solves
stochastic HP256 @ 1024 solves
```

This is a one-gauge estimator benchmark.  It is meant to help you understand
how stochastic hierarchical probing changes convergence at fixed solve count,
and how to inspect the resulting HDF5 files.  It is not a physics ensemble
result.

Gradient flow appears because the target physics use case needs flowed fields.
For this benchmark, treat the flow time `t_f` as a meaningful but fixed input
parameter.  You do not need to understand the physics of gradient flow to do
this test, and the benchmark does not study flow-time dependence.  It uses one
fixed positive flow time, `t_f=0.207936` in lattice units.

## Further Reading

This README is the main guide for the benchmark.  For extra context, use these
repo documents:

```text
docs/EMT_disconnected_1pt/EMT_disconnected_1pt.pdf
  Sections 3.2, 3.8, and 4 explain stochastic trace estimation,
  ringed-fermion kinetic normalization, and hierarchical probing/source
  bookkeeping in the older disconnected-EMT setting.

docs/proton_EMT/proton_EMT.pdf
  Sections 11 and 14.4 give broader gradient-flow and ringed-fermion
  normalization context for EMT applications.  This is optional background for
  this benchmark.

docs/flowed_quark_ringed_norm/flowed_quark_ringed_norm.md
  Short schema and convention note for the standalone ringed-normalization data
  product.
```

## Physics Target

The normalization factor is computed from the kinetic expectation value

```text
K_f(t_f) = (1 / V4) sum_x < bar_chi_f(t_f,x)
                             overleftrightarrow{Dslash}
                             chi_f(t_f,x) >
```

For the purposes of this benchmark, read this as:

```text
K = one complex scalar we want to estimate accurately
```

The stochastic source is also 4D: random phases live on all `x,y,z,t` sites,
spin, and color.  In the HDF5 output, this scalar is stored as:

```text
avg/kinetic_spacetime[flow]
```

The stored `K_f(t_f)` is converted into two normalization factors:

```text
Z_ring_bilinear(t_f) = -2 Nc / ((4 pi)^2 t_f^2 K_f(t_f))
Z_ring_field_sqrt(t_f) = sqrt(Z_ring_bilinear(t_f))
```

The `t_f=0` factors are set to `NaN`, because the conventional formula is only
used at positive flow time.  In this benchmark, `t_f=0` is kept as a
bookkeeping and schema check; the convergence comparison uses only the fixed
positive-flow entry `flow=1`.

## Mathematical Problem

The numerical task is trace estimation.  Abstractly, we want

```text
K = Tr(A) / volume
```

for a large implicit matrix/operator `A`.  We cannot build `A` explicitly.  We
can only apply the ingredients of `A` to vectors: a Dirac solve, field flow, a
covariant derivative, gamma matrices, and reductions over lattice sites.

A stochastic trace estimator uses random vectors `xi`:

```text
Tr(A) ~= average_i xi_i^dagger A xi_i
```

Hierarchical probing changes the vector set used in this estimator.  The
benchmark asks which vector set gives a more stable estimate of `K` for the
same number of expensive Dirac solves.

## Code Estimator

The code estimates the trace with 4D stochastic volume sources.  In
`make_zn_noise_fermion(...)`, random `Z_n` phases are assigned over the full
`LatticeFermion` data array, including the time direction.

For each source `xi`, the code solves the unflowed Dirac equation

```text
eta = D^{-1} xi
```

Then it flows both `xi` and `eta` along the same gauge-flow schedule.  At each
saved flow time it evaluates the local kinetic contraction

```text
sum_mu xi(t_f)^\dagger gamma_mu (Dplus_mu - Dminus_mu) eta(t_f)
```

using the flowed gauge field.  The implementation records this in
`raw/kinetic_pervec` and averages that dataset to obtain `K`.

The derivative convention written to HDF5 is:

```text
gamma_mu*(Dplus_mu-Dminus_mu)
```

The fixed flow schedule for this benchmark is:

```text
flow_type    = wilson
flow_steps   = 1
flow_epsilon = 0.207936
flow_times   = [0, 0.207936]
```

Thus there is only one positive flow time in the output.  The analyzer and PDF
use `raw/kinetic_pervec[:, 1, :]`, not a scan over several flow times.

## Hierarchical Probing

The workflow supports two noise schemes:

```text
FLOWED_RINGED_NOISE_SCHEME=zn
FLOWED_RINGED_NOISE_SCHEME=hierarchical_probing
```

For pure stochastic, each solve uses an independent random volume source.

For stochastic hierarchical probing, each base random source is multiplied by a
deterministic set of HP sign vectors.  These sign patterns cancel selected
nearby off-diagonal contributions in the trace estimator.  The estimator is
still stochastic because the base source is random.

The effective number of solves is

```text
N_eff = N_VEC * HP_NUM_VECTORS
```

The S8T8 benchmark uses the ordering

```text
interleaved_xyzt_binary_projected_to_evenodd
```

With `HP_NUM_VECTORS=16`, the first complete HP block covers the x/y/z/t parity
bits.  With `HP_NUM_VECTORS=256`, each complete block covers the next larger
HP level used in this test.  Only complete HP blocks should be used when
judging convergence.

The three benchmark cases are:

```text
case       noise_scheme            N_VEC   HP_NUM_VECTORS   effective solves
zn1024     zn                      1024    1                1024
hp64x16    hierarchical_probing    64      16               1024
hp4x256    hierarchical_probing    4       256              1024
```

For matched-cost analysis:

```text
zn1024:   blocks of 16 independent noises
hp64x16: complete 16-vector HP cycles
hp4x256: complete 256-vector HP cycles
```

That is why HP256 appears only at 256, 512, and 1024 solves in the summary and
plot.

## HDF5 Layout

Each run writes one file under:

```text
benchmark/s8t8_hp_convergence/data/<case>/FlowedQuarkRinged/
```

The core datasets are:

```text
flow_times

raw/kinetic_pervec
raw/source_index
raw/base_noise_index
raw/hp_index

avg/kinetic_spacetime
avg/Z_ring_field_sqrt
avg/Z_ring_bilinear
```

The `raw/` group keeps the per-effective-solve data.  This is what makes it
possible to redo block averages and convergence studies after the run.  The
`avg/` group keeps the full-sample averages and the ringed factors computed
from those averages.

For the 1024-solve benchmark, the expected shapes are:

```text
raw/kinetic_pervec      (1024, 2, 8)
avg/kinetic_spacetime   (2,)
avg/Z_ring_field_sqrt   (2,)
avg/Z_ring_bilinear     (2,)
flow_times              (2,)
```

The bookkeeping datasets mean:

```text
source_index      effective source index after HP expansion
base_noise_index  original stochastic base-noise index
hp_index          HP sign-vector index for that base noise
```

Important attrs include:

```text
measurement = flowed_quark_ringed_norm
normalization_scope = all_flowed_quark_fields
operator = bar_chi_overleftrightarrow_Dslash_chi
Nc = 3
flavor_convention = single_flavor_trace_for_this_dirac_operator
flow_type, flow_epsilon, flow_steps, flow_times
mass, csw, tol, maxiter
gauge_preprocessing
t_boundary
noise_scheme, n_vec, n_zn, hp_num_vectors, hp_ordering
effective_n_inversions
volume_average = spacetime_average_from_raw_kinetic_pervec
derivative_convention = gamma_mu*(Dplus_mu-Dminus_mu)
```

## Quick Start

First run a tiny smoke test on a Perlmutter login node:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/flowed_quark_ringed_norm/perlmutter
bash run_login_smoke.sh
```

The smoke test uses:

```text
script:      run_login_smoke.sh
gauge:       test_gauge/S8T8_wilson_b6.0
flow:        wilson, FLOW_STEPS=1, FLOW_EPSILON=0.207936
ranks:       1
geometry:    1.1.1.1
noise:       zn
N_VEC:       1
```

Expected smoke output:

```text
benchmark/login_smoke/data/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.0.0.x0y0z0t0.S8T8_login_smoke.h5
```

This is only an environment and schema check.  Do not use it to judge HP
convergence.

## Full Benchmark

Submit the full one-node, four-GPU benchmark from a Perlmutter login node:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/flowed_quark_ringed_norm/perlmutter
export NERSC_ACCOUNT=<your_gpu_project>
sbatch -A "$NERSC_ACCOUNT" submit_s8t8_hp_convergence.sh
```

The full benchmark uses:

```text
script:      run_s8t8_hp_convergence.sh
gauge:       test_gauge/S8T8_wilson_b6.0
flow:        wilson, FLOW_STEPS=1, FLOW_EPSILON=0.207936
ranks:       4
geometry:    1.1.1.4
GPUs:        4
```

If the job stops after one or two cases, rerun without deleting the output:

```bash
FLOWED_RINGED_SKIP_EXISTING=1 sbatch -A "$NERSC_ACCOUNT" submit_s8t8_hp_convergence.sh
```

You can also run from an interactive GPU allocation:

```bash
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account "$NERSC_ACCOUNT"
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/flowed_quark_ringed_norm/perlmutter
bash run_s8t8_hp_convergence.sh
```

For a single-GPU interactive test, override both the rank count and MPI
geometry:

```bash
export FLOWED_RINGED_NRANKS=1
export FLOWED_RINGED_MPI_GEOMETRY=1.1.1.1
bash run_s8t8_hp_convergence.sh
```

A single-GPU run should not use `ranks=4` or `geometry=1.1.1.4`.

## Analysis Outputs

After the full benchmark, these files are produced:

```text
benchmark/s8t8_hp_convergence/summary.csv
benchmark/s8t8_hp_convergence/summary.json
benchmark/s8t8_hp_convergence/s8t8_hp_convergence_results.pdf
```

The analyzer uses:

```text
K_spacetime(flow=1) = mean(raw/kinetic_pervec[:, 1, :])
```

and recomputes

```text
Z_ring_bilinear(flow=1) = -2 Nc / ((4 pi)^2 t^2 K_spacetime(flow=1)).
```

Expected matched solve rows:

```text
zn1024:   16, 32, 64, 128, 256, 512, 1024
hp64x16: 16, 32, 64, 128, 256, 512, 1024
hp4x256: 256, 512, 1024
```

The PDF has two panels:

```text
K_spacetime(flow=1), real part
Z_ring_bilinear(flow=1), real part
```

both plotted against matched solve count.

You can regenerate the summaries and plot with:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
python3 analyze_s8t8_hp_convergence.py
python3 plot_s8t8_hp_convergence_pdf.py
```

## Code Map

This directory is a thin Perlmutter application layer.  The reusable measurement
implementation lives in the main package.

Entry points:

```text
run_login_smoke.sh              login-node one-vector smoke test
run_flowed_quark_ringed_norm.sh single-measurement environment wrapper
run_s8t8_hp_convergence.sh      three-case benchmark runner
submit_s8t8_hp_convergence.sh   Slurm wrapper for one GPU node
```

Python files:

```text
Pyquda_flowed_quark_ringed_norm.py
analyze_s8t8_hp_convergence.py
plot_s8t8_hp_convergence_pdf.py
```

Shared implementation:

```text
pyquda_measurement_utils/flowed_quark_ringed_norm.py
pyquda_measurement_utils/io_corr.py
pyquda_measurement_utils/Disconnected_utils_vibe_develop.py
```

The data flow is:

```text
Slurm or login shell
  -> shell wrapper
  -> Pyquda_flowed_quark_ringed_norm.py
  -> FlowedQuarkRingedNorm.flowed_kinetic_norm(...)
  -> HDF5 file
  -> analyze_s8t8_hp_convergence.py
  -> summary.csv / summary.json
  -> plot_s8t8_hp_convergence_pdf.py
  -> PDF
```

Stable numerical defaults live in `Pyquda_flowed_quark_ringed_norm.py`: gauge
path, fixed flow schedule, mass, clover coefficient, tolerance, HP ordering,
and single-rank geometry.  The shell scripts only override values that differ
between the smoke test and the benchmark cases.

## Reading Exercises

Read the code in this order:

1. `run_login_smoke.sh`
2. `run_flowed_quark_ringed_norm.sh`
3. `Pyquda_flowed_quark_ringed_norm.py`
4. `pyquda_measurement_utils/flowed_quark_ringed_norm.py`
5. `run_s8t8_hp_convergence.sh`
6. `analyze_s8t8_hp_convergence.py`
7. `plot_s8t8_hp_convergence_pdf.py`

Useful checks:

- Change only `FLOWED_RINGED_N_VEC=2` in the login smoke and confirm that the
  first HDF5 dimension changes from 1 to 2.
- Run the smoke with `FLOWED_RINGED_NOISE_SCHEME=hierarchical_probing` and
  `FLOWED_RINGED_HP_NUM_VECTORS=16`; confirm that the first HDF5 dimension is
  16.
- Open the HDF5 attrs and identify the flow schedule, noise scheme, HP
  ordering, mass, clover coefficient, and flavor convention.
- In `summary.csv`, compare the relative standard error at 256, 512, and 1024
  solves for pure stochastic, HP16, and HP256.

## Troubleshooting

If the job says no CUDA-capable device was detected, check that the job was
submitted with `-C gpu` and `--gpus-per-node=4`, that the runner uses
`--gpus-per-task=1`, or that the interactive allocation used `--gpus 4`.

If `sbatch` rejects the job, confirm that the account is a GPU allocation.  At
NERSC, GPU projects often end in `_g`.

If Python cannot import `pyquda`, source the shared environment first:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
python3 -c "import pyquda, cupy, h5py, mpi4py"
```

If the PDF is missing but the HDF5 files exist, regenerate the summaries and
plot:

```bash
python3 analyze_s8t8_hp_convergence.py
python3 plot_s8t8_hp_convergence_pdf.py
```
