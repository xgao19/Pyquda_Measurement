# Validation status

Baseline: `5036ce0` (before the topological-charge addition).

## CPU

The complete CPU mock regression runner passed **291 tests** on 2026-10-20,
including the new topological math/workflow tests and existing EMT tests.
The runner stubs GPU/MPI operations and therefore does not establish GPU
correctness or performance. It checks the epsilon reduction, known analytic
normalization, color conjugation, reflection, clover projection, Fourier and
source alignment, immutable input, schedule, root ownership and HDF5 schema.

## GPU

Initial job **8948645** failed (`Exit_status=1`) on identity-field Wilson flow.
Unflowed loops were zero; flow produced NaN before any real-a050 comparison.
The installed QUDA `exponentiate_iQ` evaluates `acos(0/0)` for exactly zero
force. The revised test retains identity links at t=0 only and performs flow
on nontrivial fields. QUDA is unchanged and exact-zero-force flow is **not
validated or fixed**. Explicit finite checks reject NaN even if both references
contain it. Neither zero Fourier momentum nor zero total charge is excluded.

Replacement job **8948692 passed** on 2026-10-20, final PBS state **F**,
`Exit_status=0`, with `VALIDATION PASSED` and `comparison.json: passed=true`.
It used 10 nodes, 100 ranks, `debug-scaling`, requested `01:00:00` and recorded
`00:16:44` of wall time. Its output is isolated in
`runs/8948692.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov/`.
CPU control-flow mocks verified that identity cases use zero flow steps,
nontrivial cases retain their flow schedule, and matching NaNs fail explicitly.
Python and shell syntax checks passed; these are not GPU validation results.

The independent scripts, frozen code and numerical outputs remain outside
the repository under
`run/l80c80a050/develop/topological_charge_1pt/`.
The failed run's preserved output directory is
`runs/8948645.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov/`.
The submission runs a050 b.600 with no additional HYP, 27 momenta, both Wilson
and Symanzik flow, output times `0,.1,.2,.3,.4`, and integration steps
`.01,.005,.0025`. A separate EMT baseline comparison preserves its HYP1 input.
Small deterministic fields test identity links, gauge transformation, a full
epsilon contraction and decomposition independence (one rank and two
100-rank grids). Native scalar `qcharge()` is the total-charge reference.

### Numerical results

| Check | Maximum absolute difference / result |
|---|---:|
| EMT vs pre-extraction baseline, three repeats | 0 (datasets and metadata agree) |
| Nontrivial small field across MPI decompositions | 1.913e-16 |
| Small-field gauge transformation | 1.001e-15 |
| a050 total charge vs QUDA, Wilson | 2.665e-13 |
| a050 total charge vs QUDA, Symanzik | 1.741e-13 |
| Independent 24-term epsilon sum | Passed |
| Fourier conjugation, zero-mode total, repeatability | Passed |
| Identity field | Passed at t=0 only |

At common output times, halving the integrator step from `.01` to `.005`
and again to `.0025` gives loop L2 differences:

- Wilson: `8.545819e-10 -> 1.040088e-10` (8.22x reduction).
- Symanzik: `2.805449e-9 -> 3.354757e-10` (8.36x reduction).

These check numerical integration convergence, not a continuum or physical
zero-flow-time extrapolation. Both NaN and infinity are rejected.

### Timing results

One full a050 measurement contains 27 momenta and five output flow times,
including t=0, with 40 integration steps of `.01`. Two clean repeats exclude
warmup and fine-grained instrumentation:

| Measurement | Clean repeat 1 (s) | Clean repeat 2 (s) |
|---|---:|---:|
| Wilson topology loop | 39.2709 | 39.2777 |
| Symanzik topology loop | 41.0714 | 40.9714 |

Separate instrumented passes report maximum accumulated time over ranks:

| Component | Wilson (s) | Symanzik (s) |
|---|---:|---:|
| Gauge flow | 32.5628 | 34.2819 |
| Clover construction | 6.5630 | 6.5613 |
| Topological-density contraction | 0.0631 | 0.0633 |
| Momentum projection | 0.0161 | 0.0162 |
| Projected device-to-host copy | 0.0012 | 0.0011 |
| MPI reduction | 0.0071 | 0.0077 |
| HDF5 write | 0.0102 | 0.0109 |
| Total instrumented measurement | 39.3086 | 41.0289 |

Gauge input read and initial device transfer took 1.8693 seconds separately.
The loop timings exclude that initial input cost, initialization and tuning
warmup. This is one-configuration validation, not an ensemble throughput study.

Equivalent calculations use `rtol=1e-10, atol=1e-12`; near-zero global totals
use an additional roundoff bound proportional to the density L1 norm.
Step refinement tests decreasing differences, not exact equality.
The report separates warmup, two clean repeats and an instrumented pass.
Profile timings synchronize the owning device queues; component maxima need
not add exactly to a total wall time due to rank imbalance and overhead.

No production files are changed. No Perlmutter, continuum/RGI extrapolation,
physical nucleon ratio sign or final form-factor validation is claimed.
The LaTeX source is compiled with the local standalone Tectonic installation.
The repository tracks its final PDF, not compiler caches or intermediate files.
