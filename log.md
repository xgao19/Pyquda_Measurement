# PyQUDA Measurement Work Log

This file records commit-oriented history.  Before each commit, add a short
entry with the intended commit title and the main changes.  Keep reusable tips,
cluster facts, and repeated pitfalls in `SESSION_MEMORY.md` instead.

## Pending Commit: Validate connected GI qTMD cache/direct consistency

- Added connected GI qTMD bookkeeping and HDF5 cache/direct consistency tests.
- Validated pion and nucleon nonzero-staple GI-only smoke outputs on `login32`
  with `qmax=0`, `bz=2`, `bT=1`, `eta=1`.
- Added README files for the new connected pion/nucleon TMD applications.
- Updated reusable session memory with connected TMD toggles and validation
  notes.

## 2026-05-19: Add connected GI qTMD workflows and docs

- Split `SESSION_MEMORY.md` into reusable tips/baselines and this work log.
- Added connected `GI_qTMD` support for pion and proton source modules.
- Added new connected applications:
  - `application/pion_TMD/perlmutter`
  - `application/nucleon_TMD/Aurora`
  - `application/nucleon_TMD/perlmutter`
- Updated qTMD docs for connected fixed-length GI staple convention.
- Compiled all docs with `texlive/2024`, keeping only `.tex` and `.pdf` under
  `docs`.
- Added repo-local `AGENTS.md` and made commit memory updates explicit.

## 2026-05-19: Compile all docs

- Used NERSC `texlive/2024` to compile every `docs/**/*.tex`.
- Wrote LaTeX intermediates under `/tmp` and copied only final PDFs back.
- New PDFs were produced for:
  - `docs/EMT_disconnected_1pt/EMT_disconnected_1pt.pdf`
  - `docs/qTMD_disconnected_1pt/qTMD_disconnected_1pt.pdf`
- Verified `docs` contains no `.aux`, `.log`, `.out`, `.toc`, or other
  intermediate files.

## 2026-05-19: Connected pion/proton GI qTMD applications

- Added connected fixed-length `GI_qTMD` helpers to:
  - `pyquda_measurement_utils/pion_qTMD_vibe_develop.py`
  - `pyquda_measurement_utils/proton_qTMD_pyquda.py`
- Kept existing `*_TMD_CG` applications as the known-good CG-only workflows.
- Added `application/pion_TMD/perlmutter` with CG/GI/PDF toggles.
- Added `application/nucleon_TMD/Aurora` by extending the Aurora nucleon TMD
  workflow with GI staple contractions.
- Added `application/nucleon_TMD/perlmutter`, a CUDA/CuPy smoke-test-friendly
  nucleon connected qTMD workflow.
- Validated pion connected GI qTMD on `login32` with S8T32 smoke settings.
- Validated nucleon connected GI qTMD on `login32` with GI-only minimal smoke:
  `qmax=0`, `bz=0`, `bT=0`, `eta=0`.

## 2026-05-19: Disconnected GI qTMD fixed-length staple and link cache

- Added fixed-length GI qTMD convention:
  `[b_T, b_z, eta, transverse_direction]`,
  with even `b_z` and `eta >= abs(b_z)/2`.
- Added direct covariant-shift and cached gauge-only transporter paths.
- Defaulted production-style GI qTMD to link-cache transporters.
- Added and ran S8T32 tests for:
  - staple design
  - identity gauge behavior
  - link-cache vs direct covariant transport
  - HDF5 direct/cache consistency
  - GI qTMD vs GI PDF local/PDF limit
- Relevant commits:
  - `5edb024 Use fixed-length GI qTMD staple convention`
  - `ad0388c Add GI qTMD covariant staple helpers`
  - `afa600d Add GI qTMD fixed-length staple support`
  - `1fb8380 Cache GI qTMD staple transporters`

## 2026-05-18: Pion sequential-source smearing fix

- Diagnosed pion EMFF `C3/C2` ratio suppression as missing active-line
  sequential-source smearing.
- Updated `create_meson_bw_seq_pyquda(...)` with optional `sm_width` and
  `sm_boost`.
- Patched pion EMFF, pion qTMD CG, and pion EMT callers to pass the correct
  active-line sink-smearing boost.
- Validated EMFF, qTMD, and EMT S8T32 smoke tests on `login32`.
- Verified local-limit identity:
  EMFF local current equals local qTMD CG, GI_PDF, and CG_PDF at q=0 and
  zero displacement to machine precision.
- Updated pion docs with the smearing kernel and sequential-source smearing
  formulas.

## 2026-05-18: Disconnected qTMD 1pt sanity checks

- Added local/PDF limit sanity helper:
  `tests/test_qtmd_disconnected_local_pdf_limit.py`.
- Added nonzero-bz CG sanity helper:
  `tests/test_qtmd_disconnected_nonzero_bz.py`.
- Found and fixed `CG_qTMD` branch-reset bug where `b_Y` continued from the
  final shifted source of `b_X`.
- Verified `CG_PDF` and `CG_qTMD` agree at `bT=0` for nonzero `bz` after the
  reset fix.

## 2026-05-04: EMT meson baseline and convention B

- Migrated legacy meson EMT development into `pion_EMT_vibe_develop.py`.
- Removed old branchy pre-B 3pt contraction after convention-B validation.
- Standardized on:
  - fixed-sink meson sequential source
  - `meson_sign = 1`
  - `flow_epsilon`
  - HDF5-only output
- Regenerated EMT meson baseline data under:
  `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data`.
- Baseline included gluon 1pt, quark 1pt, EMT 2pt, and EMT 3pt outputs.
- Verified roundoff-level consistency against previous B sanity outputs.

## Earlier Setup: Perlmutter QUDA / PyQUDA baseline

- Built and installed QUDA under:
  `/global/cfs/cdirs/m3760/xgao/software/quda/install`.
- Installed PyQUDA and PyQUDA-Utils editable from:
  `/global/cfs/cdirs/m3760/xgao/software/PyQUDA`.
- Added Perlmutter helper scripts under:
  `systems/perlmutter`.
- Validated gradient-flow smoke test on `login32`.
