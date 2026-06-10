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

Four-step diagnostic scripts:

```text
Pyquda_EMT_disconnected_quark_1pt.py
run_quark_1pt.sh

Pyquda_EMT_disconnected_gluon_1pt.py
run_gluon_1pt.sh

Pyquda_EMT_disconnected_proton_2pt.py
run_proton_2pt.sh

Pyquda_EMT_disconnected_build_3pt.py
run_build_disconnected_3pt.sh
```

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
EMT_1PT_N_VEC=2 \
EMT_1PT_NOISE_SCHEME=zn \
bash run_quark_1pt.sh
```

Expected quark output:

```text
data/EMTc/*.h5
raw/Tmunu_pervec
raw/CHI_pervec
raw/source_index
raw/base_noise_index
raw/hp_index
avg/Tmunu/T11 ... T44
avg/CHI
```

### Step 2: Gluon EMT 1pt

```bash
EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
bash run_gluon_1pt.sh
```

Expected gluon output:

```text
data/EMTg/*.h5
Tmunu/T11 ... T44
```

There is no stochastic source axis for the gluon loop.

### Step 3: Proton C2

```bash
EMT_1PT_QMAX=0 \
bash run_proton_2pt.sh
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

### Step 4: Build the Disconnected 3pt Diagnostic

```bash
EMT_DISC_T_SEPS=2 \
bash run_build_disconnected_3pt.sh
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
Lbar_N = mean(raw/Tmunu_pervec[:N], axis=0) / volume_norm
N = 1, ..., effective_n_inversions
```

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
EMT_DISC_CONFIGS=1000,1008,1016
EMT_DISC_C2_FILES=/path/c2_1000.h5,/path/c2_1008.h5,/path/c2_1016.h5
EMT_DISC_QUARK_1PT_FILES=/path/q_1000.h5,/path/q_1008.h5,/path/q_1016.h5
EMT_DISC_GLUON_1PT_FILES=/path/g_1000.h5,/path/g_1008.h5,/path/g_1016.h5
bash run_build_disconnected_3pt.sh
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
EMT_1PT_CONFIG_NUM
EMT_1PT_MPI_GEOMETRY
EMT_1PT_QMAX
EMT_1PT_FLOW_STEPS
EMT_1PT_SM_TAG
```

Quark stochastic controls:

```text
EMT_1PT_N_VEC
EMT_1PT_N_ZN
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
EMT_DISC_T_SEPS
EMT_DISC_C2_GAMMA
EMT_DISC_C2_MOMENTUM
EMT_DISC_C2_FILES
EMT_DISC_QUARK_1PT_FILES
EMT_DISC_GLUON_1PT_FILES
EMT_DISC_3PT_OUT
```

Default loop smearing tag:

```text
EMT_1PT_SM_TAG = 1HYP_GSRC_W1_k0
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
