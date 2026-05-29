# PyQUDA Measurement Session Memory

Last updated: 2026-05-21

This file is for reusable knowledge, stable run tips, repeated pitfalls, and
validated cluster/code/test facts.  Historical commit-style progress should go
to `log.md`.

## Memory Policy

- Read this file first when resuming work in this repository.
- Keep this file concise and reusable.  Do not use it as a full work log.
- Before each commit, update both `SESSION_MEMORY.md` and `log.md`.
  `log.md` should include the intended commit title and the main changes.
- Only update this file when the user asks or when a new reusable pitfall,
  environment fact, or validation baseline should be preserved.

## Repository And Paths

- Repository root:
  `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement`
- Software root:
  `/global/cfs/cdirs/m3760/xgao/software`
- PyQUDA checkout:
  `/global/cfs/cdirs/m3760/xgao/software/PyQUDA`
- QUDA source:
  `/global/cfs/cdirs/m3760/xgao/software/quda`
- QUDA install:
  `/global/cfs/cdirs/m3760/xgao/software/quda/install`
- Shared Python venv:
  `/global/cfs/cdirs/m3760/xgao/software/venv`

## Perlmutter Environment

- Use the repo helper by default:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
export QUDA_PATH=/global/cfs/cdirs/m3760/xgao/software/quda/install
```

- Important runtime modules/settings come from the helper:
  - `cray-mpich-abi/9.0.1`
  - `cray-hdf5-parallel/1.14.3.7`
  - `MPICH_GPU_SUPPORT_ENABLED=1`
  - HDF5/MPI library preload settings that avoid h5py/HDF5 mismatch issues
- Known-good GPU login node: `login32`.
- `ssh login32` has been used successfully for small GPU smoke tests.

## Validated QUDA / PyQUDA Facts

- QUDA build is RELEASE, `sm_80`, MPI enabled.
- Important QUDA options:
  - `QUDA_DIRAC_COVDEV=ON`
  - `QUDA_MULTIGRID=ON`
  - `QUDA_DIRAC_DEFAULT_OFF=ON`
  - `QUDA_DIRAC_WILSON=ON`
  - `QUDA_DIRAC_CLOVER=ON`
  - `QUDA_DIRAC_STAGGERED=ON`
  - `QUDA_DIRAC_LAPLACE=ON`
  - `QUDA_CLOVER_DYNAMIC=OFF`
  - `QUDA_CLOVER_RECONSTRUCT=OFF`
- PyQUDA branch: `develop`.
- Editable installs completed for `PyQUDA` and `PyQUDA-Utils`.
- Validated Python environment includes:
  - Python `3.13.11`
  - `Cython==3.2.4`
  - `cupy-cuda12x==14.0.1`
  - `h5py==3.16.0`
  - `mpi4py==4.1.1`
  - `numpy==2.4.4`
  - `opt-einsum==3.4.0`

## Common Smoke Test Gauge Files

- Gradient-flow smoke gauge:
  `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T8_wilson_b6.0`
- Main measurement smoke gauge:
  `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0`

## Gradient-Flow Smoke Test

- Script:
  `systems/perlmutter/check-gradient-flow.sh`
- Python entry:
  `systems/perlmutter/check-gradient-flow.py`
- Validated on `login32`.
- Reference outputs:
  - Wilson plaquette before flow:
    `[0.5919862407536087, 0.5911411986845838, 0.5928312828226336]`
  - Wilson plaquette after one step:
    `[0.6228197128397271, 0.6220381245881715, 0.6236013010912828]`
  - Symanzik plaquette after one step:
    `[0.6374540708371997, 0.6367304972845912, 0.6381776443898083]`
  - Fermion flow:
    `fermion_norm2=1.7070053377544578`
    and `fermion_sample=(0.9234826602037387+0.9234826601680953j)`

## Git / Data Hygiene

- Never add generated HDF5 data, QUDA cache, CuPy cache, profiles, logs, or
  `__pycache__`.
- Measurement application `.gitignore` files should ignore:
  - `data/`
  - `data_smoke/`
  - `logs/`
  - `sample_log*/`
  - `.quda-cache/`
  - `.cupy-cache/`
  - `profile_*`
- `docs/` should contain only `.tex`, compiled `.pdf`, and `.gitignore`.
  Compile LaTeX with intermediates in `/tmp` when possible.
- NERSC has `texlive/2024`; use:

```bash
module load texlive/2024
```

## Repeated Pitfalls And Fixes

- Do not revert user or unrelated changes in a dirty worktree.
- Use `apply_patch` for manual edits.
- Use `rg`/`rg --files` for search.
- In code comments, write English only.
- On CuPy, avoid implicit CuPy-to-NumPy conversion.  Use `.get()` before
  wrapping PyQUDA gamma arrays into another backend array.
- Clean `__pycache__` after `py_compile` if it appears in new application dirs.
- Some QUDA memory-leak warnings at program shutdown have appeared in successful
  smoke tests; check the process exit code and physics output before treating
  them as fatal.
- If using `py_compile` in production directories with restrictive filesystems,
  it may fail only because Python cannot write `__pycache__`.  Use `ast.parse`
  if a write-free syntax check is needed.
- The lightweight test runner is:

```bash
source systems/perlmutter/activate-venv-quda.sh
python tests/run_smoke_tests.py
```

  The default runner skips tests marked with `TEST_REQUIRES = "gpu"` or
  `TEST_REQUIRES = "external_hdf5"`.
- Optional tiny-gauge GPU smoke tests can be run on `login32` with:

```bash
PYQUDA_RUN_TINY_GAUGE_SMOKE=1 python -c "from tests.test_tiny_gauge_smoke_workflows import test_optional_pion_soft_factor_tiny_gauge_smoke; test_optional_pion_soft_factor_tiny_gauge_smoke()"
```

  The pion soft-factor tiny smoke has been validated on `login32`.

## EMT Conventions And Baselines

- Active pion/meson EMT source:
  `pyquda_measurement_utils/pion_EMT_vibe_develop.py`
- Active proton EMT source:
  `pyquda_measurement_utils/proton_EMT_vibe_develop.py`
- Legacy `pyquda_measurement_utils/EMT_meson.py` was removed after migration;
  recover from Git history if needed.
- EMT meson Perlmutter application:
  `application/EMT_meson/perlmutter`
- Proton EMT applications:
  - `application/EMT_proton/perlmutter` contains connected quark 3pt plus
    proton 2pt, quark 1pt, and gluon 1pt workflows.
  - `application/EMT_proton/Aurora` is intentionally connected-only: proton
    2pt plus connected quark EMT 3pt.  It does not include quark/gluon 1pt.
- Runtime working directory used earlier:
  `/global/cfs/cdirs/m3760/xgao/software/EMT_meson`
- EMT output is HDF5-only in vibe paths; `.npy` was removed.
- HDF5 tags include `lat`, `cfg`, `ama`, `src`, and `sm`.
- `flow_epsion` was standardized to `flow_epsilon`.
- `GEN_SIMD_WIDTH` was removed from EMT entry scripts.
- Gradient-flow schedule convention:
  - measure first, then flow
  - `step=0` is unflowed
  - first interval is subdivided into 10 small steps
  - output index `step` corresponds to `step * flow_epsilon`
- Meson/proton EMT 3pt uses convention B:
  - fixed-sink sequential source
  - `dst2 = gamma5 * seq_bw_prop^dagger * gamma5`
  - sequential source uses `gamma5 * Gamma_sink^dagger * gamma5`
  - `meson_sign = 1`
- EMT connected validation reached roundoff-level agreement after refactors:
  - B-only vs previous B sanity:
    `C2 max_rel ~ 1.5e-17`,
    `C3_chi max_rel ~ 8.4e-16`,
    `C3_Tmunu max_rel ~ 4.8e-16`
  - q=0 explicit-q output matched old no-q baseline at the true q=0 index.
- Ringed fermion normalization data:
  - reconstruct from quark 1pt diagonal `Tmunu` at q=0:
    `avg/Tmunu/T11`, `T22`, `T33`, `T44`
  - `CHI` is a scalar trace/noise diagnostic, not the standard ringed-fermion
    normalization by itself.
- Gluon 1pt code saves the full gluonic building block, not a traceless EMT
  projection.  `_F_clover_traceless` only projects field-strength matrices onto
  the su(3) algebra.

## Pion Sequential-Source Smearing

- Important fix: meson sequential source needs active-line sink smearing before
  sequential inversion, matching the known-good proton logic.
- `create_meson_bw_seq_pyquda(...)` accepts optional `sm_width` and `sm_boost`.
- EMFF now passes `width` and `pos_boost_sink`.
- Pion EMFF C2 must not use mixed source/sink boost smearing.  Even when the
  3pt uses independent `pos_boost_src`, `pos_boost_sink`, `neg_boost_src`, and
  `neg_boost_sink`, the 2pt has no momentum transfer and uses source-side
  boosts on both ends: `pos_boost_src` and `neg_boost_src`.
- Pion EMFF C2 tags are intentionally shorter than 3pt tags:
  `posSrc..._negSrc...` only.  The 3pt tag keeps all four boost labels.
- pion qTMD CG now passes `width` and `pos_boost`.
- pion EMT passes the appropriate active-line boost when `CG_GaussSmear` is on.
- S8T32 local-limit check passed after this fix:
  - EMFF local current
  - local qTMD CG
  - GI_PDF local
  - CG_PDF local
  all agree at q=0, `bT=bz=eta=0`, `gamma_4/T`, source/sink gamma5.
  Max absolute difference: `8.401198274707147e-14`;
  max relative difference: `7.016753011768688e-16`.

## Test Suite Baselines

- Current default lightweight test baseline:
  `89 passed, 5 skipped, 0 failed`.
- The skipped tests are optional GPU or external-HDF5 checks.
- The test suite now includes guards for:
  - PyQUDA gamma label/order and composite gamma signs
  - pion/meson sequential-source phase and smearing placement
  - pion EMFF free-field momentum flow
  - pion soft-factor contraction order, phase convention, HDF5 schema, and
    prop/contract time-slice bookkeeping
  - connected/disconnected qTMD local/PDF/GI limits and Wilson-link helpers
  - EMT 1pt/3pt HDF5 schema, flow bookkeeping, and connected toy contraction
  - boosted-smearing kernel phase/symmetry
  - disconnected noise and hierarchical-probing bookkeeping
  - pion current background-response phase/formula/HDF5 summary schema
- Optional pion current background-response tiny-gauge GPU smoke passed on
  `login32` using S8T32 with `qext=[0,0,0]` and `[0,0,1]`, `tsep=2,4`,
  restricted tau window, and current `T`.
  Explicit summed C3 and response C2-like contractions agreed at
  `relative_difference ~ 5e-18` to `2e-16`.
- Pion current background-response HDF5 schema v2 includes a `summary/` group
  with table-like datasets for `relative_difference`, `response_R_sum`,
  `explicit_R_sum`, `pf`, `qext`, `pi`, `tsep`, gamma labels, and window labels.
- Pion current-background response code is now in the more generic module:
  `pyquda_measurement_utils/pion_current_background_response_vibe_develop.py`.
  The former `pion_EMFF_background_response_vibe_develop.py` name should not be
  used in new code.
- Pion current-background response docs live under:
  `docs/pion_current_background_response/pion_current_background_response.tex`.
  The old `docs/pion_EMFF_background_response` path should not be used.
- Current-current response diagnostic uses the nested first-order construction
  without caching per-tau response propagators:
  `S_resp^(2) = D^{-1} O_2 D^{-1} O_1 S`.
- Current-current response HDF5 output uses measurement
  `pion_current_current_response`, schema version `1`, and stores a `summary/`
  table with `response_R_sum`, `pf`, `pi`, `first_qext`, `second_qext`,
  `total_qext`, tau windows, and gamma labels.
- Optional pion current-current response tiny-gauge GPU smoke passed on
  `login32` using S8T32, `pf=[0,0,0]`, `q1=[0,0,1]`, `q2=[0,0,-1]`,
  `tsep=2`, and current gamma `T/T`.
- Pion soft-factor prop generation should usually cover all time slices
  (`PION_SOFT_T_COUNT=0`) because contract time `t0` with separation `tsep`
  needs wall propagators at both `t0` and `(t0 + tsep) % Lt`.

## Connected qTMD / GI qTMD

- Original connected CG-only applications remain:
  - `application/pion_TMD_CG/perlmutter`
  - `application/nucleon_TMD_CG/Aurora`
- New connected applications with GI qTMD support:
  - `application/pion_TMD/perlmutter`
  - `application/nucleon_TMD/Aurora`
  - `application/nucleon_TMD/perlmutter`
- Fixed-length GI qTMD staple convention:
  - Wilson index: `[b_T, b_z, eta, transverse_direction]`
  - path:
    `x -> x + (eta + b_z/2) zhat
       -> x + (eta + b_z/2) zhat + b_T e_perp
       -> x + b_z zhat + b_T e_perp`
  - staple length: `2*eta + b_T`
  - constraints: `b_z` even, `eta >= abs(b_z)/2`, `b_T >= 0`
- Reusable implementation pattern:
  - build gauge-only staple transporters once
  - apply each transporter to endpoint-shifted fermion/propagator
  - use `link_cache` mode by default when available
- Output tags:
  - pion connected GI qTMD: `GI_qTMD.ex`
  - proton connected GI qTMD: `GI_qTMD.U.ex`, `GI_qTMD.D.ex`
- Pion connected GI qTMD S8T32 smoke passed on `login32`.
- Nucleon connected GI qTMD Perlmutter smoke passed on `login32` with
  `qmax=0`, `bz=0`, `bT=0`, `eta=0`, GI-only enabled.
- Connected GI qTMD nonzero-staple cache/direct consistency passed on
  `login32` with `qmax=0`, `bz=2`, `bT=1`, `eta=1`, GI-only enabled:
  `link_cache` and `direct_covdev` agree within `1e-12` for pion and nucleon.
- Optional regression test:
  `tests/test_connected_gi_qtmd_link_cache_consistency.py`.
  It compares paired HDF5 outputs under
  `/tmp/pyquda_connected_gi_qtmd_consistency`.
- Useful connected TMD toggles:
  - pion: `PION_TMD_RUN_CG_QTMD`, `PION_TMD_RUN_GI_QTMD`,
    `PION_TMD_RUN_PDF`, `PION_TMD_GI_STAPLE_MODE`
  - nucleon: `NUCLEON_TMD_RUN_CG_QTMD`, `NUCLEON_TMD_RUN_GI_QTMD`,
    `NUCLEON_TMD_RUN_PDF`, `NUCLEON_TMD_GI_STAPLE_MODE`

## Disconnected qTMD 1pt

- Source:
  `pyquda_measurement_utils/Disconnected_1pt_qTMD_vibe_develop.py`
- Shared stochastic helpers:
  `pyquda_measurement_utils/Disconnected_utils_vibe_develop.py`
- Supported operator kinds:
  - `CG_qTMD`
  - `CG_PDF`
  - `GI_PDF`
  - `GI_qTMD`
- Hierarchical probing exists, with source bookkeeping datasets.
- Short-term plan: no spin-color dilution and no time dilution.
- The local/PDF limit sanity test is codified:
  `tests/test_qtmd_disconnected_local_pdf_limit.py`
- The nonzero-bz CG sanity test is codified:
  `tests/test_qtmd_disconnected_nonzero_bz.py`
- Known fixed bug: `CG_qTMD` must reset shifted source when transverse direction
  changes; otherwise `b_Y` continues from the final `b_X` shift.
- Local-limit expected identity:
  `GI_PDF(bz=0) = CG_PDF(bz=0) = CG_qTMD(bT=0,bz=0)`.

## Documentation

- qTMD connected docs:
  - `docs/pion_qTMD/pion_qTMD.tex`
  - `docs/proton_qTMD/proton_qTMD.tex`
- Disconnected qTMD docs:
  - `docs/qTMD_disconnected_1pt/qTMD_disconnected_1pt.tex`
- Pion smearing docs updated previously in:
  - `docs/pion_EMFF/pion_EMFF.tex`
  - `docs/pion_qTMD/pion_qTMD.tex`
  - `docs/pion_EMT/pion_EMT.tex`
  - `docs/pion_qTMDWF/pion_qTMDWF.tex`
- Smearing kernel:

```tex
K_{\mathbf{k}}(\mathbf{r})
=
\exp\left[-\frac{r_x^2+r_y^2+r_z^2}{2w^2}\right]
\exp\left[
  2\pi i
  \left(
    \frac{k_x r_x}{L_x}
    + \frac{k_y r_y}{L_y}
    + \frac{k_z r_z}{L_z}
  \right)
\right].
```

## Current Worktree Reminder

At the time this memory was split, the working tree had uncommitted connected
GI qTMD/application/docs/PDF changes.  Check `git status --short` before any
commit, and update `log.md` with the intended commit title and summary first.
