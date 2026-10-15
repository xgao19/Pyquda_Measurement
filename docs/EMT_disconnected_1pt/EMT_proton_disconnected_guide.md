# EMT Proton Disconnected Diagram Workflow

This note gives the minimal end-to-end diagnostic workflow for combining EMT
disconnected one-point loops with a proton two-point function.

Use your own checkout path:

```bash
export MEASUREMENT_ROOT=/path/to/your/Pyquda_Measurement_develop
export PYQUDA_ROOT="$MEASUREMENT_ROOT"
```

The workflow assumes that `QUDA`, `PyQUDA`, and the Python environment are
already installed by the user. These instructions describe the current develop
format. Use a fresh output/setup identity, including for Z_N; old shards/logs
are not migrated and require the original checkout. See
[PROBING.md](../../application/EMT_disconnected_1pt/PROBING.md).

---

## 1. What This Workflow Computes

The quark disconnected one-point workflow estimates stochastic quark loops:

```text
L_munu(q,tau;t_f) ~= Tr[ Gamma_munu D^{-1} ]
```

with stochastic sources:

```text
eta = D^{-1} xi
Tr[Gamma D^{-1}] ~= (1/N) sum_r xi_r^\dagger Gamma eta_r
```

The one-point loop is hadron independent.  The proton disconnected three-point
building block is constructed in analysis from proton `C2` and loop `L`:

```text
C3_disc,munu(pf,pi;t,tau;t_f)
  = < C2_proton(pf,t) L_munu(q,tau;t_f) >_cfg
    - < C2_proton(pf,t) >_cfg < L_munu(q,tau;t_f) >_cfg
```

with:

```text
q = pf - pi
```

The basic ratio is:

```text
R_disc,munu(t,tau;t_f) = C3_disc,munu(t,tau;t_f) / C2_proton(t)
```

For one `S8T32` gauge, the workflow checks code paths, shapes, momentum/time
conventions, and stochastic-source convergence proxies.  A physical
vacuum-subtracted disconnected signal requires multiple gauge configurations.

---

## 2. Main Code Paths

Application directory:

```text
application/EMT_disconnected_1pt/perlmutter
```

Quark-only diagnostic scripts, including explicit finalization:

```text
Pyquda_EMT_disconnected_quark_1pt.py
run_quark_1pt.sh

Pyquda_EMT_disconnected_finalize_quark_1pt.py
run_finalize_quark_1pt.sh

Pyquda_EMT_disconnected_proton_2pt.py
run_proton_2pt.sh

Pyquda_EMT_disconnected_build_3pt.py
run_build_disconnected_3pt.sh
```

The builder is quark-only by default.  The separate gluon measurement is an
advanced optional step and is included only when the builder is called with
`--include_gluon`.

Shared source code:

```text
pyquda_measurement_utils/Disconnected_1pt_EMT_vibe_develop.py
pyquda_measurement_utils/Disconnected_utils_vibe_develop.py
pyquda_measurement_utils/proton_EMT_vibe_develop.py
```

Related docs:

```text
application/EMT_disconnected_1pt/README.md
docs/EMT_disconnected_1pt/EMT_disconnected_1pt.tex
docs/proton_EMT/proton_EMT.tex
```

---

## 3. Environment

Enter the repository:

```bash
cd "$MEASUREMENT_ROOT"
export PYQUDA_ROOT="$MEASUREMENT_ROOT"
source systems/perlmutter/activate-venv-quda.sh
export PYTHONPATH="$MEASUREMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

If this activation script does not match your own `QUDA` / `PyQUDA`
installation, replace it with your own environment setup. Select the develop
checkout after activation as above. `PYQUDA_ROOT` also prevents the Perlmutter
wrappers from prepending the original checkout when they reactivate the
installation. These are Perlmutter examples; Aurora uses its own environment
and MPI/GPU binding. Run GPU measurements only with suitable GPU resources.

Check the test gauge:

```bash
ls -lh test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

For Perlmutter login-GPU tests, use the appropriate GPU login node, for example:

```bash
ssh login32
cd "$MEASUREMENT_ROOT/application/EMT_disconnected_1pt/perlmutter"
```

---

## 4. Measurement and Finalization

Run commands from the application directory and keep the shared settings
exported for measurement, finalization and the optional builder:

```bash
cd "$MEASUREMENT_ROOT/application/EMT_disconnected_1pt/perlmutter"
export EMT_1PT_DATA_DIR=/path/to/new_disconnected_run/data
export EMT_1PT_LAT_TAG=S8T32
export EMT_1PT_SETUP_TAG=1HYP.disc_v4
export EMT_1PT_FLOW_STEPS=1
export EMT_1PT_QMAX=0
export EMT_1PT_QZ_MAX=0
export EMT_1PT_N_VEC=2
export EMT_1PT_N_ZN=4
export EMT_1PT_RAND_SEED=0
export EMT_1PT_NOISE_SCHEME=zn
```

Use the existing S8T32 gauge for this example, or explicitly set
`EMT_1PT_GAUGE_PATH` and a matching lattice tag. `N_VEC=2` gives two independent
bases, sufficient to exercise SEM bookkeeping but not to establish convergence.
Choose one of the two paths below **before** starting production. If changing
raw-storage mode after a run, use a new setup/output identity.

### A. Ordinary complete-base analysis

```bash
bash run_quark_1pt.sh --config_num 1000
bash run_finalize_quark_1pt.sh --config_num 1000
```

Measurement writes shard means and the sample log, not a canonical file.
Finalization writes:

```text
$EMT_1PT_DATA_DIR/EMTquarkLoop/
  S8T32.EMTquarkLoop.1000.0.1HYP.disc_v4.bases.h5

base/base_noise_index
base/local_bilinear
base/derivative_bilinear
base/flowed_noise_norm
base/Tmunu/T11 ... T44
base/ringed/kinetic
base/ringed/kinetic_spacetime
avg/...
sem_real/...
sem_imag/...
```

These observables are already volume normalized. New convergence, ringed and
proton T44 analyses read independent base values directly. A one-base run has
NaN SEM. This path does not require proton C2 or the raw-based 3pt builder.

### B. Raw-based proton disconnected diagnostic

The current builder requires undiluted raw canonical files: it does not accept
`.bases.h5` or SC12 raw files. For this path, perform these four steps.

### Step 1: Measure quark shards with raw diagnostics

```bash
bash run_quark_1pt.sh --config_num 1000 --save-raw-per-vector
```

### Step 2: Finalize both bases and raw canonical output

```bash
python Pyquda_EMT_disconnected_finalize_quark_1pt.py \
  --config_num 1000 --raw-output
```

`--raw-output` is a Python-finalizer option, not an option of the shell wrapper.
It adds the following file alongside `.bases.h5`:

```text
$EMT_1PT_DATA_DIR/EMTquarkLoop/
  S8T32.EMTquarkLoop.1000.0.1HYP.disc_v4.h5

raw/local_bilinear_pervec
raw/derivative_bilinear_pervec
raw/flowed_noise_norm_pervec
raw/base_noise_index
raw/probe_index
avg/Tmunu/T11 ... T44
avg/local_bilinear
avg/derivative_bilinear
avg/flowed_noise_norm
derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

Both modes require complete validated shard means; raw-only shards are not a
fallback. Finalization uses the same full `N_VEC`, configuration, data/shard
directory and setup tag as measurement. The shared format uses
`disconnected_1pt_base_parts_v4` and `disconnected_sample_log_v2`, with
`.baseXXXXXX.partXXXX.probeSTART-END.h5` part filenames.
Resume trusts the fingerprinted sample log; unlogged bases are recomputed.
Finalization independently checks destination-side shard completeness.

The quark loop file is source independent and is reused for every proton
two-point source time on the same configuration.  Its full-volume counter-based
noise is keyed by global coordinates, spin, color, configuration, base-noise
index, and the optional stream salt.

At flow time `t_f`, the measured fields are `xi_f=K(t_f)xi` and
`eta_f=K(t_f)D^{-1}xi`.  For an absolute insertion-time projector `P_tau`,

```text
L_hat(tau,t_f) = xi^dag K^dag P_tau Gamma K D^{-1} xi
E[L_hat]       = Tr[P_tau Gamma K D^{-1} K^dag].
```

The loop is a spatial trace at fixed `tau`, not a time-summed observable.
However, `K(t_f)` is a four-dimensional fermion-flow kernel and spreads in the
temporal direction.  Restricting the initial source to only one time slice is
therefore not the same finite-flow estimator.  Complete time dilution would
require summing all time projectors; this workflow instead keeps one
full-volume `Z4` source and stores every absolute `tau`.

For a two-point source time `t0`, downstream analysis selects

```text
tau_abs = (t0 + tau_rel) mod Nt.
```
The current builder implements this for the quark loop, and also for the
optional gluon loop when `--include_gluon` is requested, as
`roll(time_axis, -t0)` before any C3 product or ensemble subtraction.

The canonical EMTquarkLoop file has no source-position tag because the same loop is
reused for every `t0` on that configuration.

The saved quark loop is an unringed flowed bilinear. Optional raw canonical
output embeds kinetic data under `derived/ringed`; the default base output
stores `base/ringed` and `avg/ringed`. Neither stores a per-configuration ringed
factor: compute that factor only after averaging the configuration-level
kinetic observable over the ensemble.

### Optional: Gluon EMT 1pt

```bash
EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
EMT_1PT_QZ_MAX=0 \
bash run_gluon_1pt.sh --config_num 1000
```

Expected gluon output:

```text
data/EMTgluonLoop/<lat>.EMTgluonLoop.<cfg>.<ama>.<sm>.h5
Tmunu/T11 ... T44
```

There is no stochastic source axis or hadron source-position tag for the gluon
loop.  Run the shared `application/EMT_disconnected_1pt/perlmutter/run_gluon_1pt.sh`
entry; meson/proton-specific gluon wrappers are intentionally not maintained.
To include it in the final combination, append `--include_gluon` to
`run_build_disconnected_3pt.sh`. The quark-convergence benchmark does not need
this step.

### Step 3: Proton C2

```bash
EMT_1PT_QMAX=0 \
EMT_1PT_QZ_MAX=0 \
bash run_proton_2pt.sh --config_num 1000
```

Expected C2 output:

```text
$EMT_1PT_DATA_DIR/EMTproton2pt/*.h5
SS/I/PX0PY0PZ0
SS/T/PX0PY0PZ0
```

The default selected channel for the merger is:

```text
source interpolator = 5
sink projector = PpUnpol = (I + gamma_4) / 4
momentum = PX0PY0PZ0
```

### Step 4: Build the Disconnected 3pt Diagnostic

```bash
bash run_build_disconnected_3pt.sh --configs 1000 \
  --t_separations 2
```

Expected merger output:

```text
$EMT_1PT_DATA_DIR/EMTdisc3pt/*.h5
C2
quark/source_count
quark/loop_cumulative
quark/C3_unsubtracted_cumulative
gluon/loop                    # only with --include_gluon
gluon/C3_unsubtracted         # only with --include_gluon
summary/quark_source_count
summary/quark_T44_q0_flow0_loop_norm
summary/quark_T44_q0_flow0_unsub_ratio_proxy
```

For a single configuration, `quark/C3_disc_cumulative` and
`quark/R_disc_cumulative` are intentionally absent.  They are written only when
multiple configurations are provided.

---

## 5. Random-Source Convergence Check

For ordinary stochastic-error analysis, use `.bases.h5` and the current
`emt_quark_1pt_convergence.py` / `emt_ringed_stochastic_comparison.py` helpers.
They derive solve cost as `vectors_per_base * dilution_factor`, not from a
hardcoded HP-mode table. SC12 has dilution factor 12; the 12 channels are
summed, not treated as independent samples.

The optional undiluted raw merger instead builds cumulative diagnostic averages:

```text
B_N[nu,mu] = mean(
    raw/derivative_bilinear_pervec[:N,gamma_nu,mu], axis=0
) / volume_norm
Lbar_N[mu,nu] = (B_N[mu,nu] + B_N[nu,mu]) / 2
N = 1, ..., effective_n_inversions
```

The current raw Gamma order is stored in `gamma_list`; do not assume numerical
PyQUDA-ID order.  The vector positions are currently `[X,Y,Z,T]=[3,5,7,1]`.
This complete averaged reconstruction is a useful schema check:

```python
import h5py
import numpy as np

with h5py.File(loop_file, "r") as h5:
    labels = [x.decode() for x in h5["gamma_list"][...]]
    vector = [labels.index(x) for x in ("X", "Y", "Z", "T")]
    D = h5["avg/derivative_bilinear"][...]
    B = np.take(D, vector, axis=0)  # [nu,mu,q,flow,t_abs]
    T = 0.5 * (B + np.swapaxes(B, 0, 1))
    np.testing.assert_allclose(T[3, 3], h5["avg/Tmunu/T44"][...])
```

The explicit matrices, physical axial transform, tensor convention, and
connected pion/proton axis examples are in
[`docs/EMT_gamma_and_raw_bilinears.md`](../EMT_gamma_and_raw_bilinears.md).

For ringed-fermion normalization, read `avg/ringed/kinetic_spacetime` from
`.bases.h5`, or `derived/ringed/kinetic_spacetime` from optional raw canonical
output.  The exact
reconstruction below remains a useful cross-check:

```text
K_code(flow) = -2 * mean_tau_cfg[
    T11(q=0,flow,tau) + T22(q=0,flow,tau)
  + T33(q=0,flow,tau) + T44(q=0,flow,tau)
]
```

First average this kinetic value over configurations, then apply the resulting
ringed bilinear factor to quark connected and disconnected EMT observables at
the same flow step.  Do not average configuration-local inverse factors, and
do not use the unflowed `flow=0` step.

The single-configuration proxy is:

```text
C3_unsubtracted_N(tsep,mu,nu,q,flow,tau)
  = C2(tsep) * Lbar_N(mu,nu,q,flow,tau)
```

The quick convergence datasets are:

```text
summary/quark_source_count
summary/quark_T44_q0_flow0_loop_norm
summary/quark_T44_q0_flow0_unsub_ratio_proxy
```

These are useful for checking whether the stochastic estimator changes
smoothly as `N` increases.  They do not replace a real multi-configuration
error analysis. HP/multiplier prefixes within a base are correlated diagnostics,
not extra independent samples for SEM.

---

## 6. Multi-Configuration Physical Combination

To build the physical disconnected building block, provide matching lists:

```bash
export EMT_DISC_C2_FILES=/path/c2_1000.h5,/path/c2_1008.h5,/path/c2_1016.h5
export EMT_DISC_QUARK_1PT_FILES=/path/q_1000.h5,/path/q_1008.h5,/path/q_1016.h5
bash run_build_disconnected_3pt.sh --configs 1000,1008,1016
```

The quark paths above must be the undiluted raw canonical files, not `.bases.h5`.
For gluon output, also export `EMT_DISC_GLUON_1PT_FILES` with a matching list
and add `--include_gluon`. For `Ncfg >= 2`, the builder writes:

```text
quark/C3_disc_cumulative
quark/R_disc_cumulative
gluon/C3_disc                 # only with --include_gluon
gluon/R_disc                  # only with --include_gluon
```

using:

```text
C3_disc = mean_cfg(C2 * L) - mean_cfg(C2) * mean_cfg(L)
R_disc = C3_disc / mean_cfg(C2)
```

---

## 7. Useful Environment Variables

Shared input/output controls:

```text
EMT_1PT_DATA_DIR
EMT_1PT_GAUGE_PATH
EMT_1PT_MPI_GEOMETRY
EMT_1PT_QMAX
EMT_1PT_QZ_MAX
EMT_1PT_FLOW_STEPS
EMT_1PT_SETUP_TAG
```

Configuration identity is deliberately not an environment variable.  Pass
`--config_num CFG` to measurement/finalize wrappers and `--configs
CFG[,CFG...]` to the disconnected 3pt builder.

Quark stochastic controls:

```text
EMT_1PT_N_VEC
EMT_1PT_N_ZN
EMT_1PT_RAND_SEED
EMT_1PT_TOL
EMT_1PT_NOISE_SCHEME
EMT_1PT_HP_NUM_VECTORS
EMT_1PT_HP_ORDERING
EMT_1PT_MULTIPLIER_PLAN
EMT_1PT_BASE_START
EMT_1PT_BASE_STOP
EMT_1PT_BLOCK_INTERVAL_SOLVES
EMT_1PT_SHARD_DIR
```

The shared Python HP default is `nested_xyzt_checkerboard_v1`; normally omit
`EMT_1PT_HP_ORDERING`. HP requires a complete count 2, 32 or 512 and global
extents divisible by 2, 4 or 8 respectively. Old orderings and HP16/HP256 are
not supported. Multiplier uses a validated shape-specific plan, with all
Fourier probes averaged within each base. `hp_num_vectors` is an HP input, not
a stored generic probe count; output stores `vectors_per_base`.

SC12 is available through the EMT Python measurement API only; the CLI does
not currently expose it. Neither standalone ringed nor qTMD supports SC12 or
multiplier. The raw-based 3pt builder in this guide remains undiluted only.

Proton C2 and merger controls:

```text
EMT_DISC_INTERPOLATOR
EMT_DISC_WIDTH
EMT_DISC_BOOST_IN
EMT_DISC_BOOST_OUT
--t_separations
EMT_DISC_C2_MOMENTUM
EMT_DISC_C2_FILES
EMT_DISC_QUARK_1PT_FILES
EMT_DISC_GLUON_1PT_FILES
EMT_DISC_3PT_OUT
```

For the default `PpUnpol` proton projection, the merger constructs

```text
C2_PpUnpol = 0.25 * (C2_I + C2_T)
```

from the two stored sink-gamma channels.  The source interpolator label `5`
must not be reused as a sink projector.

Default loop smearing tag:

```text
EMT_1PT_SETUP_TAG = 1HYP
```

For new-format runs, explicitly override this default with a fresh identity,
as in `1HYP.disc_v4` above; do not reuse an old run's log or shards.

Default proton C2 smearing tag:

```text
EMT_DISC_SM_TAG = 1HYP_GSRC_W1_k0_5
```

---

## 8. Validation Scope

The new probing CPU/mock suite needs NumPy, h5py, opt_einsum, matplotlib and
pytest. It uses explicit GPU/MPI import stubs and the real PyQUDA gamma matrices:

```bash
cd "$MEASUREMENT_ROOT"
python tests/run_probing_cpu_tests.py --pyquda-root /path/to/PyQUDA
```

This verifies source algebra, SC12 accounting, resume, aggregation and analysis,
not actual QUDA/GPU/MPI execution. New probing GPU/MPI validation remains
pending. Historical operator checks do not validate the new probing schemes.
Use the backend-specific GPU entry and fresh output directories described in
[PROBING.md](../../application/EMT_disconnected_1pt/PROBING.md) when GPU resources
are allocated. This guide does not submit validation jobs automatically.

Useful tests to read:

```text
tests/test_disconnected_probing_math.py
tests/test_disconnected_probing_workflow.py
tests/test_disconnected_noise_bookkeeping.py
tests/test_disconnected_emt_flow_bookkeeping.py
tests/test_emt_hdf5_schema.py
```
