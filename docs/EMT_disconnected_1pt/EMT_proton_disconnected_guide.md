# EMT Proton Disconnected Diagram Workflow

This note gives the minimal end-to-end diagnostic workflow for combining EMT
disconnected one-point loops with a proton two-point function.

Use your own checkout path:

```bash
export MEASUREMENT_ROOT=/path/to/your/Pyquda_Measurement
```

The workflow assumes that `QUDA`, `PyQUDA`, and the Python environment are
already installed by the user.

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

Three-step quark-only diagnostic scripts:

```text
Pyquda_EMT_disconnected_quark_1pt.py
run_quark_1pt.sh

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
source systems/perlmutter/activate-venv-quda.sh
```

If this activation script does not match your own `QUDA` / `PyQUDA`
installation, replace it with your own environment setup.

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

## 4. Four-Step Smoke Workflow

Run all commands from:

```bash
cd "$MEASUREMENT_ROOT/application/EMT_disconnected_1pt/perlmutter"
```

### Step 1: Quark EMT 1pt

Use `N_VEC=2` for the smallest stochastic-source convergence check:

```bash
EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
EMT_1PT_QZ_MAX=0 \
EMT_1PT_N_VEC=2 \
EMT_1PT_N_ZN=4 \
EMT_1PT_RAND_SEED=0 \
EMT_1PT_NOISE_SCHEME=zn \
bash run_quark_1pt.sh --config_num 1000
```

Expected quark output:

```text
data/EMTquarkLoop/<lat>.EMTquarkLoop.<cfg>.<ama>.<sm>.h5
raw/local_bilinear_pervec
raw/derivative_bilinear_pervec
raw/flowed_noise_norm_pervec
raw/base_noise_index
raw/hp_index
avg/Tmunu/T11 ... T44
avg/local_bilinear
avg/derivative_bilinear
avg/flowed_noise_norm
derived/ringed/kinetic_pervec
derived/ringed/kinetic_spacetime
```

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

The saved quark loop is an unringed flowed bilinear.  The same EMTquarkLoop embeds
kinetic data under `derived/ringed`, extracted from the identical raw EMT
vectors.  It contains no per-configuration ringed factor: compute that factor
only after averaging `derived/ringed/kinetic_spacetime` over the ensemble.

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

### Step 2: Proton C2

```bash
EMT_1PT_QMAX=0 \
EMT_1PT_QZ_MAX=0 \
bash run_proton_2pt.sh --config_num 1000
```

Expected C2 output:

```text
data/EMTproton2pt/*.h5
SS/5/PX0PY0PZ0
```

The default selected channel for the merger is:

```text
gamma = 5
momentum = PX0PY0PZ0
```

### Step 3: Build the Disconnected 3pt Diagnostic

```bash
bash run_build_disconnected_3pt.sh --configs 1000 \
  --t_separations 2
```

Expected merger output:

```text
data/EMTdisc3pt/*.h5
C2
quark/source_count
quark/loop_cumulative
quark/C3_unsubtracted_cumulative
gluon/loop
gluon/C3_unsubtracted
summary/quark_source_count
summary/quark_T44_q0_flow0_loop_norm
summary/quark_T44_q0_flow0_unsub_ratio_proxy
```

For a single configuration, `quark/C3_disc_cumulative` and
`quark/R_disc_cumulative` are intentionally absent.  They are written only when
multiple configurations are provided.

---

## 5. Random-Source Convergence Check

The quark merger builds cumulative loop averages:

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

For ringed-fermion normalization, read
`derived/ringed/kinetic_spacetime` from the same canonical EMTquarkLoop.  The exact
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
error analysis.

---

## 6. Multi-Configuration Physical Combination

To build the physical disconnected building block, provide matching lists:

```bash
EMT_DISC_C2_FILES=/path/c2_1000.h5,/path/c2_1008.h5,/path/c2_1016.h5
EMT_DISC_QUARK_1PT_FILES=/path/q_1000.h5,/path/q_1008.h5,/path/q_1016.h5
EMT_DISC_GLUON_1PT_FILES=/path/g_1000.h5,/path/g_1008.h5,/path/g_1016.h5
bash run_build_disconnected_3pt.sh --configs 1000,1008,1016
```

For `Ncfg >= 2`, the builder writes:

```text
quark/C3_disc_cumulative
quark/R_disc_cumulative
gluon/C3_disc
gluon/R_disc
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
```

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

Default proton C2 smearing tag:

```text
EMT_DISC_SM_TAG = 1HYP_GSRC_W1_k0_5
```

---

## 8. Lightweight Repository Tests

Run:

```bash
cd "$MEASUREMENT_ROOT"
source systems/perlmutter/activate-venv-quda.sh
python tests/run_smoke_tests.py
```

Useful tests to read:

```text
tests/test_disconnected_noise_bookkeeping.py
tests/test_disconnected_emt_flow_bookkeeping.py
tests/test_emt_hdf5_schema.py
```
