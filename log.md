# PyQUDA Measurement Work Log

This file records commit-oriented history.  Before each commit, add a short
entry with the intended commit date, title, and main changes.  Keep reusable
tips, cluster facts, and repeated pitfalls in `SESSION_MEMORY.md` instead.

## 2026-07-10: Trim session memory to durable guidance

- Reduced `SESSION_MEMORY.md` to reusable environment facts, stable physics
  and implementation conventions, and recurring pitfalls.
- Moved completed-work history and one-off validation details out of session
  memory; those remain available in this log and Git history.
- Kept the current full-volume `Z4`, four-dimensional fermion-flow, canonical
  EMT-loop, HP, qTMD, and ringed-normalization conventions.

## 2026-07-10: Add full-volume Z4 EMT loop workflow

- Replaced rank-local stochastic EMT sources with decomposition-independent
  counter-based full-volume `Z4` noise keyed by global coordinates, spin,
  color, configuration, base-noise index, and stream salt.
- Made EMT quark-loop files source independent, changed quark 1pt defaults to
  `Z4` and `tol=1e-10`, restricted quark/gluon HDF5 writes to rank 0, and
  removed redundant gauge loads.
- Preserved spatial and 4D hierarchical-probing choices while adding source
  bookkeeping, canonical tag, HDF5 metadata, and partition-invariance tests.
- Documented four-dimensional fermion-flow spreading, fixed-time spatial
  traces, full-volume sources, and the distinction between one time projector
  and a complete time-dilution basis across the disconnected, pion, and proton
  EMT notes; rebuilt all three PDFs.
- Verified `112 passed, 12 skipped`, Python compilation, LaTeX cross-references,
  PDF text, and `git diff --check`.  A real multi-rank GPU smoke still requires
  a valid Slurm allocation.

## 2026-07-02: Add HP256 ringed-norm sample resume

- Tightened standalone flowed-quark ringed normalization to fixed-interval
  block HDF5 output as the only persistent data product.
- Added deterministic HP256 sample-log resume support so complete base-noise
  samples can be skipped while partial interval blocks remain available for
  diagnostics.
- Added base-range controls for l64 production-style runs and per-base
  deterministic stochastic seeds independent of skipped samples.
- Updated Aurora and Perlmutter S8T8 analyzers, drivers, docs, and tests for
  interval block files, removed obsolete block-output metadata assumptions,
  and kept the ringed normalization dataset layout unchanged.
- Added and validated tests for fixed interval metadata, HP256 sample logs,
  base-range selection, deterministic sample seeds, and updated tag naming.

## 2026-06-29: Add flowed ringed-norm block checkpoints

- Added optional complete-block HDF5 checkpoint output for standalone
  flowed-quark ringed normalization, preserving the existing monolithic output
  path by default.
- Added block sizing and filename helpers so pure stochastic, HP16, HP256, and
  HP16 plus spin-color point dilution write complete estimator blocks at or
  above the configured minimum solve count.
- Exposed `FLOWED_RINGED_BLOCK_WRITE`, `FLOWED_RINGED_BLOCK_MIN_SOLVES`, and
  `FLOWED_RINGED_SAVE_FULL` in the Aurora and Perlmutter ringed-norm drivers.
- Added unit/schema tests for block sizing, block file naming, and block HDF5
  metadata.
- Updated the Aurora ringed-norm README with the l64c64a076 prod256 benchmark
  setup, partial-data conclusions, and archived convergence PDF.
- Added a `.gitattributes` rule so archived PDFs are treated as binary files
  during Git diff checks.
- Verified Python compilation, shell syntax for the prod256 helpers, focused
  flowed ringed-norm tests in the Aurora PyQUDA develop environment, and a
  synthetic block-file analyzer smoke.

## 2026-06-23: Add spin-color dilution for flowed ringed norm

- Added `spin_color_dilution=point` to the flowed-quark ringed-normalization
  source bookkeeping while preserving the default full spin-color stochastic
  noise behavior.
- Kept HP site-only and applied it before broadcasting into exact spin/color
  basis channels.
- Extended effective inversion counting, HDF5 raw bookkeeping, and metadata
  with spin/color indices and spin-color dilution factors.
- Corrected point-diluted kinetic normalization to use a spin-color trace
  factor of `12` in the spacetime kinetic average and in convergence analysis.
- Added the `hp6x16sc12` S8T8 convergence benchmark case to Aurora and
  Perlmutter helpers and updated the PDF/CSV/JSON analysis.
- Updated flowed ringed-norm documentation and Perlmutter intern README to
  describe full spin-color noise, site-only HP, point spin-color dilution, and
  the benchmark interpretation.
- Verified Python compilation, shell syntax, and focused tests for flowed
  ringed norm plus disconnected noise bookkeeping.

## 2026-06-23: Add proton EMT source completion callback

- Added an optional `on_source_done` callback to
  `ProtonQuarkEMT.connected_3pt(...)`.
- The callback runs after each source finishes successfully, allowing external
  run drivers to record per-source completion without waiting for the full
  source batch to return.
- Preserved the existing batch API behavior when no callback is supplied, so
  active Aurora and Perlmutter template callers remain compatible.
- Verified the modified proton EMT module with Python compilation.

## 2026-06-22: Remove embedded C2 from EMT quark 3pt outputs

- Changed `save_emt_quark_3pt_hdf5(...)` to save only `C3_chi`,
  `C3_Tmunu`, and optional `momentum_transfer_list`; quark EMT 3pt files no
  longer embed selected two-point data.
- Updated proton and pion EMT quark 3pt callers to use the lighter writer
  interface and removed `c2_selected_*` attrs from 3pt outputs.
- Updated proton EMT documentation and schema tests to make the no-embedded-C2
  convention explicit.
- Validated syntax for touched Python files and ran
  `tests/test_emt_hdf5_schema.py` successfully.

## 2026-06-22: Add Perlmutter ringed-norm HP benchmark

- Added `application/flowed_quark_ringed_norm/perlmutter` with login smoke,
  single-measurement runner, one-node S8T8 HP convergence benchmark, Slurm
  wrapper, analyzer, PDF plotter, and intern-facing README.
- Extended S8T8 HP convergence support to include pure stochastic, HP16, and
  HP256 matched at 1024 solves.
- Simplified standalone ringed-normalization HDF5 schema by removing
  `avg/kinetic_timeslice`; consumers use `raw/kinetic_pervec` for convergence
  analysis and `avg/kinetic_spacetime` for ringed factors.
- Validated Python/shell syntax and `tests/test_flowed_quark_ringed_norm.py`
  under the Aurora PyQUDA develop environment.

## 2026-06-18: Encode proton EMT 3pt sink kinematics in file names

- Changed `get_emt_proton_quark_3pt_file_tag(...)` to require `pf` and one
  `t_sep`.
- Appended `PX<px>PY<py>PZ<pz>dt<tsep>` to connected proton EMT 3pt tags,
  matching the established nucleon TMD naming convention.
- Wrote multiple source-sink separations as separate HDF5 files while keeping a
  length-one `tsep` axis in each file.
- Added the single-value `t_separations` HDF5 attribute so it explicitly labels
  the corresponding `C3_chi` and `C3_Tmunu` axis.
- Updated the Aurora and Perlmutter proton EMT application callers.
- Kept proton 2pt file names unchanged because they are independent of
  source-sink separation.
- Updated the output-convention documentation and tag-helper regression test.
- Verified Python compilation, all three tag-helper tests, and HDF5 metadata
  round-trip for a length-one `tsep` axis.

## 2026-06-10: Add EMT disconnected full-workflow diagnostic

- Added proton `C2` generation and disconnected `C2 x 1pt` merger scripts under
  `application/EMT_disconnected_1pt/perlmutter`.
- Added run wrappers, a concise workflow guide, and a detailed LaTeX diagnostic
  section with regenerated PDF.
- Validated syntax, `login32` S8T32 quark/gluon/C2/build smoke workflow, and
  one-config plus fake two-config HDF5 schema behavior.

## 2026-06-02: Fix proton EMT left derivative and raw-only sequential builder

- Fixed proton connected EMT `C3_Tmunu` left-acting derivative by differentiating
  the raw sequential propagator before the final gamma5-hermiticity/index
  transform, matching the meson EMT construction and standard proton GFF EMT
  convention.
- Added `create_bw_seq_raw_pyquda(...)` for proton EMT so the workflow no longer
  constructs both finalized `dst_seq` and raw sequential propagators; restored
  `create_bw_seq_pyquda(...)` public behavior for qTMD/nucleon TMD callers.
- Updated `docs/proton_EMT/proton_EMT.tex` with the raw-sequential derivative
  convention and MIT/Fermilab proton GFF reference.
- Validated on Aurora S8T32 point-source gauge-covariance tests: local
  `C3_Tmunu` relative diff improved from `0.4599556563267348` to
  `9.244013511964703e-12`; `C2` and `C3_chi` stayed unchanged.

## 2026-05-29: Add Aurora proton EMT connected workflow

- Added `application/EMT_proton/Aurora` for connected quark EMT 3pt plus proton
  2pt only.
- Added Aurora/SYCL driver, run script, PBS submit script, README, and ignore
  rules for data/cache/log outputs.
- Validated Python and shell syntax locally; no Aurora runtime test was run.

## 2026-05-21: Generalize pion current-response docs

- Renamed the background-response documentation from
  `pion_EMFF_background_response` to `pion_current_background_response`.
- Reworded the document as a general pion current-response note with EMFF as
  the first concrete application and current-current response as an extension.
- Regenerated the PDF and removed LaTeX intermediate files.

## 2026-05-21: Add pion current-current response diagnostics

- Renamed the pion EMFF background-response module to the more generic pion
  current-background response module.
- Added current-current response helpers using
  `D^{-1} O_2 D^{-1} O_1 S`, plus HDF5 summary schema and a minimal
  Perlmutter GPU diagnostic application.
- Updated background-response documentation and regenerated the PDF.
- Validated default smoke tests and optional `login32` tiny-gauge GPU smoke for
  both first-order current response and second-order current-current response.

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
