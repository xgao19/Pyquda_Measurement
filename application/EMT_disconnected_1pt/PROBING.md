# Complete HP and multiplier probing

The implementation changes only source construction, bookkeeping and statistics.
The Dirac, flow and EMT operators are unchanged. This document describes usage;
GPU/MPI validation is still pending.

## HP

Set `noise_scheme="hierarchical_probing"` and explicitly supply
`hp_num_vectors=2`, `32`, or `512`. The shared Python default is
`hp_ordering="nested_xyzt_checkerboard_v1"` for EMT, standalone ringed and qTMD.
Shell entries forward only explicit overrides. Old orderings and incomplete
counts (including HP16/HP256) are rejected; their generators have been removed
from this checkout. Standalone ringed and qTMD support Z_N and HP only.

For q=1,2,4, the number of colors/vectors is `2*q**4`. Complete bases cancel
site pairs through distances 1,3,7, respectively. Each global extent must be
divisible by `2*q`; power-of-two extents are not required. Use a new
output/setup identity for every new-format run, including Z_N.

The existing application reads:

```bash
export EMT_1PT_NOISE_SCHEME=hierarchical_probing
export EMT_1PT_HP_NUM_VECTORS=32
export EMT_1PT_SETUP_TAG=1HYP.hp32.nested-v1
```

## Multiplier

Set `noise_scheme="multiplier_probing"` and supply `multiplier_plan` as a
dictionary or JSON path. Rank zero reads/checks the plan and broadcasts either
the canonical plan or an error. The actual gauge shape must match the plan.

```json
{"version": 1, "shape": [8, 8, 8, 8], "distance": 2, "colors": 16, "sigma": [1, 2, 3, 4]}
```

The application accepts `EMT_1PT_MULTIPLIER_PLAN`:

```bash
export EMT_1PT_NOISE_SCHEME=multiplier_probing
export EMT_1PT_MULTIPLIER_PLAN="$PWD/application/EMT_disconnected_1pt/probing_plans/m2_8x8x8x8.json"
export EMT_1PT_SETUP_TAG=1HYP.m2.16colors.v1
```

The supplied plans cover M1/M2/M6/M8 on 8^4 and M4 on 64^4. They are not interchangeable
between shapes without verification. Distance-3/7 search is not implemented.
M2/HP32 and M4/HP512 are not equal-cancellation-radius comparisons.

The temporary 8^4 distance-6 test plan uses **411 colors**, sigma
`(1,8,64,102)`, and requires **411 inversions per base**. Its shortest same-color
distance is 7. An exhaustive search establishes the smallest color count within
the cyclic linear coloring family, not among arbitrary nonlinear colorings or
by minimum EMT wall time/variance. See [search evidence](probing_plans/M6_SEARCH.md).

```bash
export EMT_1PT_NOISE_SCHEME=multiplier_probing
export EMT_1PT_MULTIPLIER_PLAN="$PWD/application/EMT_disconnected_1pt/probing_plans/m6_8x8x8x8.json"
export EMT_1PT_SETUP_TAG=1HYP.m6.411colors.v1
```

Use an actual 8^4 gauge with this plan; the older application's default 8x8x8x32
gauge does not match it. GPU/MPI EMT validation of M6 remains pending.

The 8^4 distance-8 test plan uses **829 colors**, sigma `(1,8,246,310)`, and
requires **829 inversions per base**. Its shortest same-color distance is 9.
This is also color-count optimal within cyclic linear colorings only; it is not
a minimum-variance or minimum-wall-time claim. See [M8 search evidence](probing_plans/M8_SEARCH.md).

```bash
export EMT_1PT_NOISE_SCHEME=multiplier_probing
export EMT_1PT_MULTIPLIER_PLAN="$PWD/application/EMT_disconnected_1pt/probing_plans/m8_8x8x8x8.json"
export EMT_1PT_SETUP_TAG=1HYP.m8.829colors.v1
```

Use an actual 8^4 gauge and a separate output identity. GPU/MPI EMT validation
of M8 remains pending; adding this plan does not submit a job.

`colors` is the number of Fourier vectors per independent base. The same noise
is reused for all vectors and the sum is divided by `colors`. No mask-source
mode or runtime search is provided. `EMT_1PT_N_VEC` remains the number of bases.

## Outputs and resume

This checkout writes `disconnected_1pt_base_parts_v4` shards and
`disconnected_sample_log_v2` logs. Use fresh output/setup identities, including
for Z_N. Old logs/files are rejected, not inferred, converted or mixed. Use the
original `Pyquda_Measurement` checkout for old workflows and their data.

Part filenames use `.baseXXXXXX.partXXXX.probeSTART-END.h5`. Stored metadata
has one probe count, `vectors_per_base`, plus `noise_scheme` and explicit
`spin_color_dilution`. HP additionally stores `hp_ordering`; multiplier stores
`multiplier_plan_json`. `hp_num_vectors` remains an HP-only input option, not a
second stored count. Raw bookkeeping uses `base_noise_index`, `probe_index`,
and, for point dilution, `spin_index` and `color_index`. The effective source index is
`(base * vectors_per_base + probe) * factor + channel`, with factor 12 for
point dilution and 1 otherwise; `channel = 3 * spin_index + color_index`
for SC12 and zero for undiluted sources.

Resume is for a fixed plan/count/storage mode. Interrupted bases are recomputed;
completed bases are skipped according to the fingerprinted sample log. Changing
these settings requires a new output identity. There is no automatic HP layer
extension or old-file migration. Different jobs must own nonoverlapping bases.

EMT always writes complex128 part means. `--save-raw-per-vector` adds raw data
for diagnostics; it does not replace the means. The base finalizer requires
means and all requested complete bases, weights parts by their actual probe
counts, and checks raw/mean consistency when raw is present. SC12 contributions
are summed over the 12 channels and averaged only over probes.

The default finalizer writes `<canonical_tag>.bases.h5`, containing `base/`,
`avg/`, `sem_real/`, and `sem_imag/` with volume-normalized primitive loops,
Tmunu and linear ringed kinetic observables. `base/base_noise_index` identifies
independent samples. SEM is NaN for one base; nonlinear ringed factors and
ensemble gauge errors are not computed here.

```bash
python application/EMT_disconnected_1pt/perlmutter/Pyquda_EMT_disconnected_finalize_quark_1pt.py \
  --config_num 11
```

Adding `--raw-output` to this Python command also creates `<canonical_tag>.h5`
for per-source diagnostics and the optional raw-based disconnected 3pt builder.
This requires measurement with `--save-raw-per-vector`; the shell finalizer
wrapper performs only the default base finalization. Standalone ringed and
qTMD keep their raw finalization paths and always retain required raw data.

## Analysis

Ringed comparisons, convergence analysis and proton T44 analysis read the new
`.bases.h5` directly. They do not regroup raw vectors or accept old files.
Methods and inversion costs come from metadata, not HP16/HP256 lookup tables:

```text
solves per independent base = vectors_per_base * (12 if point dilution else 1)
total solves = n_base_noise * solves per independent base
```

For example HP32 costs 32 solves/base, multiplier with 16 colors costs 16,
and HP2 with SC12 costs 24. Comparisons must use the same physics and total
solve cost. A small number of bases cannot establish reliable variance gains.
`fixed_cost_job_manifest(mode_specs, total_solves=..., chunks=...)` takes an
explicit mapping of labels to `vectors_per_base` and optional
`spin_color_dilution`; only complete bases may be divided among jobs.

## Verification commands

From the develop repository root, with NumPy, h5py, opt_einsum, matplotlib and pytest:

```bash
python -m pytest tests/test_disconnected_probing_math.py -q
python tests/run_probing_cpu_tests.py
```

The second command explicitly installs import stubs for unavailable PyQUDA
GPU/MPI entry points; it runs real source, algebra, shard, resume and aggregation
code with CPU/mock fields. It loads gamma matrices from the sibling
`PyQUDA-develop` source (override with `--pyquda-root`). It is not GPU validation.
Run the same test files with ordinary pytest in the real PyQUDA environment to
avoid these stubs.

On an allocated Frontier GPU, activate the existing environment, then prepend
this develop repository to `PYTHONPATH`; the environment script otherwise points
to the original checkout. Keep the existing rank/GPU binding outside the test.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tests/check_disconnected_probing_gpu.py --output /tmp/probing-hp2-new --hp-count 2 --run-emt
```

The GPU entry defaults to CuPy HIP and requires an unused output directory. On
Aurora, activate the SYCL environment and add `--backend-target sycl` to use
dpnp. Prepend this develop repository to `PYTHONPATH` after activation on either
system. The entry creates its own
QUDA cache, checks M1/HP2 source identity, and writes a decomposition-independent
source SHA-256 to `smoke.json`. Repeat with `--hp-count 32` or `512`; use
`--scheme multiplier_probing --plan <JSON>` for multiplier. `--run-emt` adds one
complete base and one flow interval on the repository's 8^4 gauge, with MG off
for this small correctness test. It does not benchmark production performance.

For MPI verification, launch the same entry under the site's rank/GPU wrapper
with a matching `--mpi-geometry` and a new output directory. For the same scheme
and parameters, compare `global_source_sha256` between decompositions; compare
the generated base observables within solver/roundoff tolerance. Also compare
HP2 and M1 complete EMT outputs. No scheduler submission is performed by these
scripts.
