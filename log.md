# PyQUDA Measurement Work Log

This file records commit-oriented history.  Before each commit, add a short
entry with the intended commit date, title, and main changes.  Keep reusable
tips, cluster facts, and repeated pitfalls in `SESSION_MEMORY.md` instead.

## 2026-07-18: Consolidate connected qTMD/PDF output

- Replaced the per-Gamma connected qTMD/PDF files and their deeply nested
  HDF5 groups with one dense
  `corr[wilson,momentum,gamma,time]` dataset per operator.
- Pion production now writes one file per enabled operator. Proton production
  writes one file per operator, flavor, and polarization while retaining one
  polarization-independent C2 file per source.
- Added polarization to the proton sample-log identity so independently
  produced polarization channels cannot skip one another.
- Stored the canonical 16-Gamma basis, momentum list, Wilson-index list, and
  explicit axis metadata in every connected qTMD/PDF file. Old hierarchical
  files are not read or migrated.
- The S8T8 numerical gate used solver tolerance `1e-15`, one/four MPI ranks,
  nonzero momenta, all CG/GI qTMD/PDF paths, and proton `PpUnpol`/`PpSzp`.
  Reconstructed legacy arrays and dense output were bitwise identical in both
  layouts. The one/four-rank maximum absolute and relative-L2 differences were
  `1.46e-14` and `6.85e-14`; output counts were five pion files and seventeen
  proton files for two polarizations.

## 2026-07-18: Decouple connected qTMD operator utilities

- Moved the unchanged CG/GI qTMD and PDF Wilson-index, displacement, staple
  construction, and cached transport operations into the neutral
  `qtmd_operator_utils.py` module.
- Rewired connected pion/proton and disconnected qTMD production to the same
  pure operator functions, removing connected imports of disconnected
  production modules without changing execution order or output schema.
- Consolidated NumPy/CuPy/dpnp host conversion in `tools.array_to_numpy` and
  removed the disconnected and pion duplicate implementations.
- S8T8 reference/candidate runs covered connected pion/proton and disconnected
  CG/GI qTMD/PDF with one-rank `1.1.1.1` and four-rank `2.2.1.1` layouts.
  All 27,655 same-layout datasets were bitwise identical at both solver
  tolerances `1e-15` and `1e-10`; the full regression suite passes with
  279 tests and 12 skips.

## 2026-07-18: Unify sink-separation interfaces

- Replaced the remaining insertion-time production arguments with the common
  `t_separations` list interface; low-level sequential contractions now use
  the scalar name `t_sep`.
- Kept pion/proton qTMD at one separation per invocation while retaining
  pion EMFF multi-separation production and partial resume.
- Removed unused EMFF measurement state and the obsolete disconnected-proton
  C2 fallback, so that wrapper now supplies the `t_separations` required by
  `ProtonQuarkEMT`.
- Removed the old CLI and environment names without a compatibility alias;
  output tags, HDF5 data, and sample-log entries remain unchanged.
- Removed sink-separation environment variables altogether.  Production
  entrypoints and shell wrappers now accept `--t_separations` directly, while
  platform/example defaults are ordinary hard-coded defaults.
- S8T8 reference/candidate runs at solver tolerance `1e-15` covered pion
  qTMD, proton qTMD, pion EMFF, pion/proton EMT, and disconnected-workflow
  proton C2 with one-rank `1.1.1.1` and four-rank `2.2.1.1` layouts.
  Same-layout HDF5 datasets were bitwise identical; the largest cross-layout
  absolute and relative-L2 differences were `7.27e-14` and `6.23e-13`,
  respectively.  qTMD/EMFF sample-log files were unchanged.
- The final regression suite passes with 275 tests and 12 skips; Python and
  shell syntax checks, the two-pass pion EMFF PDF build, and
  `git diff --check` also pass.

## 2026-07-18: Fix GI qTMD staple composition order

- Reversed `covDev` traversal of the unchanged fixed-length geometric staple
  segments and synchronized the direct test reference and fake-path
  expectations.
- Added a fixed-seed, position-dependent, noncommuting SU(3) ordered-product
  CPU test covering positive and negative `b_z` and both transverse
  directions.
- Validated the HYP-smeared S8T8 gauge in one-rank and four-rank Perlmutter
  layouts, including cached/direct agreement, local gauge covariance,
  gauge-invariant contractions, straight-PDF limits, and a minimal connected
  pion GI-qTMD workflow.
- Added the staple geometry and operator-order TikZ figure, recorded numerical
  validation results in the pion TMD README, and rebuilt the tracked PDF.
- Verified `274 passed, 12 skipped`, two-pass LaTeX compilation, visual figure
  layout, and `git diff --check`.

## 2026-07-17: Unify proton qTMD production and resume

- Replaced the duplicated Perlmutter, Aurora GI, and Aurora CG-only drivers
  with one backend-independent application runner plus two thin platform
  entrypoints.
- Added exact-line source-level resume, mandatory CLI configuration, and an
  explicit runtime-only `--mg-block` interface defaulting to `8.8.4.4`.
- Removed unused proton qTMD/EMT state, duplicate Gamma aliases, the unused GI
  gauge argument, and the old `application/nucleon_TMD_CG` workflow.
- Kept MG, tolerance, and maxiter out of HDF5 and sample-log identity.
- S8T8 reference/candidate runs at solver tolerance `1e-15` covered two
  sources, C2, CG/GI qTMD, CG/GI PDF, zero/nonzero momenta, all 16 Gamma
  channels, and every Wilson index.  Same-layout relative L2 differences were
  at most `6.41e-13`; one-rank/four-rank differences were at most `4.36e-13`.
  A repeated candidate run skipped both sources before inversion.
- The final CPU regression suite passes with 273 tests and 12 skips.

## 2026-07-16: Fix the unequal-boost pion EMT active line

- Kept the positive-boost propagator as the fixed-sink spectator used to build
  the sequential source, including its positive sink smearing and the outer
  negative active-line sink smearing.
- Corrected the direct EMT insertion field to use the independently inverted
  negative-boost propagator.  The previous code incorrectly reused the
  positive spectator when the two boosts differed.
- Replaced ambiguous pion `source_boost`/`sink_boost` HDF5 attrs with
  `pos_boost`/`neg_boost` and recorded the negative-active line convention.
- Added structural line-identity and provenance tests.  S8T8 tests at solver
  tolerance `1e-15`, for both 1-rank and 4-rank layouts, found exact
  reference/candidate equality for equal boosts and unchanged C2 for opposite
  boosts.  The corrected unequal-boost C3 is distinctly different from the
  old positive-active result, while its 1-rank/4-rank relative L2 difference
  is below `2e-15`.
- Added explicit `--pos-boost` and `--neg-boost` options to the pion EMT
  Python, run, and submit entrypoints.  The default smearing tag now encodes
  both line boosts, preventing unequal-boost outputs from sharing the old
  `k0` identity; a user-supplied `EMT_SM_TAG` remains an explicit override.

## 2026-07-13: Extend quark EMT to the complete Dirac bilinear basis

- Added one canonical 16-element PyQUDA Gamma definition shared by EMT, qTMD,
  and two-point helpers, together with explicit matrices and a stored
  `physical_from_pyquda` transform.  This records the `Y5`/`T5` axial signs and
  distinguishes raw `[gamma_mu,gamma_nu]/2` tensors from the Hermitian
  `i[gamma_mu,gamma_nu]/2` convention.
- Extended disconnected quark shards to save 16 local and `16x4`
  unsymmetrized derivative bilinears.  Removed the redundant raw symmetric
  `Tmunu_pervec`; the finalizer derives `avg/Tmunu`, ringed kinetic data, and
  the existing disconnected build products from vector primitive channels.
- Extended pion and proton connected EMT contractions and HDF5 files with the
  same primitive basis.  Existing `C3_chi` and `C3_Tmunu` remain derived
  datasets with their previous shapes and conventions.
- Batched all Gamma contractions and momentum projections after constructing
  each derivative once.  No inversion, stochastic source, fermion-flow step,
  sequential inversion, or covariant-derivative count was added.
- Added algebra, schema, shard/finalizer, and old-vector-EMT regression tests.
  A real A100 smoke also verified the CuPy metadata path, the full primitive
  shard/finalizer schema, exact identity-to-CHI, vector-to-EMT, and
  vector-diagonal-to-ringed reconstruction.  Final verification passed with
  138 tests and 8 skips; all three EMT LaTeX documents compiled twice without
  undefined references and their PDFs were rebuilt.

## 2026-07-13: Fix disconnected qTMD trace direction

- Replaced the reversed `eta^dagger Gamma O_b xi` contraction with the unbiased
  estimator `xi^dagger Gamma O_b eta`, applying every qTMD/PDF operator to
  `eta=D^{-1}xi` for the target trace `Tr[P_qtau Gamma O_b Dinv]`.
- Added qTMD schema-version and trace-convention provenance, and made the
  finalizer reject old reversed-trace shards without replacing canonical data.
- Replaced regression checks that depended on invalid old HDF5 outputs with
  complete-basis trace-direction and current-operator tests.
- Updated the qTMD workflow documentation and derivation; old disconnected
  qTMD shards and canonical files must be discarded and regenerated.
- Rebuilt the qTMD PDF and verified `121 passed, 8 skipped`, Python syntax,
  LaTeX references, and `git diff --check` with the `software_gradientflow`
  development environment.  The nontrivial-gauge cached-link/direct and
  staple/PDF-limit GPU regression also passed on `login32`.

## 2026-07-13: Simplify disconnected production workflows

- Replaced skip-set stochastic iteration with direct deterministic generation
  of one base and HP interval; EMT, qTMD, and standalone ringed now share the
  same part layout, strict resume validator, and completion markers.
- Removed EMT/qTMD monolithic production, configuration fallbacks, duplicate
  `rand_seed` metadata, source-tag loop helpers, and obsolete meson quark-1pt
  platform wrappers.
- Migrated standalone ringed away from `.block*.h5` and HP256 text sample logs;
  its explicit finalizer writes kinetic-only configuration files, and a new
  ensemble analyzer averages K before computing nonlinear ringed factors.
- Made qTMD GI production link-cache-only and moved direct covariant transport
  to a test reference.  Removed stale platform convergence scripts.
- Unified shared quark/gluon flow epsilon defaults at `0.207936`, changed gluon
  loops to one source-independent canonical file per configuration, and removed
  duplicate hadron-specific gluon wrappers.

## 2026-07-13: Restore lightweight disconnected sample-log resume

- Replaced EMT, qTMD, and standalone ringed JSON completion markers and HDF5
  resume probes with one fingerprinted text log and one exact line per complete base.
- Production now trusts the log even after shards have been transferred; an
  unlogged base is recomputed in full and atomically replaces its part files.
- Finalizers no longer prevalidate every shard. They infer layout from base 0
  and validate each expected part once while streaming canonical output.
- Required explicit CLI configuration identity throughout disconnected EMT,
  disconnected qTMD, and standalone ringed production and analysis entry points.
- Updated the durable Perlmutter memory to use the `software_gradientflow`
  venv, QUDA install, and editable PyQUDA checkout instead of the old shared
  `m3760/software` stack.
- Updated workflow documentation and regression tests for the intentionally
  incompatible production schema and rebuilt the four affected PDFs.
- Verified `113 passed, 12 skipped`, Python and shell syntax, and
  `git diff --check` using the `software_gradientflow/venv-quda-develop`
  environment.

## 2026-07-11: Remove rank-local RNG from disconnected sources

- Removed backend RNG generation from the shared disconnected source utility
  and made global-coordinate SplitMix64 counter noise mandatory.
- Added decomposition-independent site-only counter noise for exact point
  spin-color dilution and migrated standalone ringed normalization, including
  HP256 sample identity, to configuration/stream/base counter keys.
- Changed standalone defaults to `Z4` with stream salt zero and documented the
  repeated-local-noise failure caused by identical per-rank RNG seeds.
- Removed the invalid legacy Aurora l64 convergence plot, rebuilt the EMT,
  pion, proton, and qTMD PDFs, and verified `125 passed, 12 skipped`.

## 2026-07-10: Add base-resumable disconnected qTMD loops

- Reused the disconnected base/HP shard infrastructure for qTMD/PDF loops,
  with explicit streaming finalize and source-independent canonical tags.
- Changed production defaults to decomposition-independent counter-based `Z4`
  and `tol=1e-10`, restricted serial HDF5 to rank 0, and removed sequential RNG
  and redundant per-vector gauge loads.
- Preserved all operator, Wilson-line, link-cache, raw, and averaged schemas;
  retained the legacy source-tag helper and monolithic library mode.
- Verified `121 passed, 12 skipped`, rebuilt the qTMD PDF, and obtained exact
  S8T8 raw/average agreement between two independent base jobs plus finalize
  and the matched monolithic run.

## 2026-07-10: Add base-resumable EMT quark loop shards

- Added base/HP-interval atomic shards, resume validation, complete-base
  markers, and an explicit streaming finalizer for canonical EMTc and kinetic
  companion files.
- Kept monolithic library output for compatibility while making production
  wrappers default to shard mode with independently schedulable base ranges.
- Fixed stochastic runs after positive fermion flow by restoring the original
  QUDA resident gauge before every new inversion.
- Verified `120 passed, 12 skipped`, rebuilt three EMT PDFs, and compared two
  independent S8T8 base jobs plus finalize against monolithic GPU output.

## 2026-07-10: Integrate ringed kinetic into EMT quark loops

- Derived the flowed-quark kinetic trace directly from the zero-momentum
  diagonal raw EMT tensor without additional inversions, flow updates,
  derivative contractions, or MPI gathers.
- Made every in-repository quark EMT 1pt entry point write a source-matched,
  kinetic-only `FlowedQuarkRinged` companion with counter-noise and source
  bookkeeping metadata; retained the standalone block/resume workflow.
- Documented that the nonlinear ringed factor must be evaluated after the
  kinetic expectation value is averaged over gauge configurations, and rebuilt
  the disconnected, pion, and proton EMT PDFs.
- Verified `117 passed, 12 skipped`; an S8T8 single-rank GPU smoke gave zero
  EMT-to-kinetic identity and averaging error.  A two-rank smoke still requires
  a valid Slurm allocation.

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
## 2026-07-13: EMT operator schema v3 and production cleanup

- Fixed pion connected serial HDF5 writes so only MPI rank 0 opens the file.
- Made disconnected quark/gluon loop times source-relative in `build_3pt` and
  changed quark-loop reading to source chunks containing only X/Y/Z/T Gamma
  channels.
- Replaced disconnected `CHI` duplication with the identity local primitive
  plus `flowed_noise_norm`, and embedded ringed kinetic data under
  `EMTc/derived/ringed`; finalization now publishes one file atomically.
- Removed broken Frontier pion, duplicate proton quark-1pt, hadron gluon alias,
  and unused connected helper entry points.
- Removed the singleton proton `tsep` data axis, made connected configuration
  CLI-only, expanded connected provenance, and updated Perlmutter defaults to
  the current `software_gradientflow` runtime.
- EMT schema v3 is intentionally incompatible with previous shards, sample
  logs, canonical loops, and connected outputs; all must be regenerated.
- Verified one S8T32, one-source, zero-flow shard and finalization on a
  `login32` A100: schema 3 produced derivative shape `[1,16,4,1,1,32]`,
  flowed-noise norm, and embedded kinetic shape `[1,1,32]`.

## 2026-07-13: Centralized application analysis helpers

- Moved the disconnected EMT loop readers and source-relative time alignment
  from `pyquda_measurement_utils` to
  `application/analysis_helper/emt_disconnected_analysis.py`.
- Reserved `pyquda_measurement_utils` for production measurements, operators,
  and reusable computational infrastructure; application data combination and
  post-processing now have a shared application-level home.
- Kept the reader formulas, HDF5 schema, chunking, and output axes unchanged.

## 2026-07-14: S8T8 EMT Z2/Z4 and hierarchical-probing benchmark

- Completed six fixed-gauge disconnected EMT measurements at 2048 solves each:
  Z2/Z4 pure stochastic, HP16, and HP256, using the production spatial HP
  ordering and the embedded ringed kinetic at `t_f/a^2=0.207936`.
- Distributed 24 non-overlapping 512-solve chunks over idle login-node GPUs;
  all jobs, shard validation, and six canonical finalizations completed.
- Pure Z2 and Z4 had essentially equal endpoint stochastic variance.  At this
  fixed cost, HP16/pure SEM-squared ratios were 1.49 (Z2) and 1.73 (Z4), while
  HP256/pure ratios were 16.49 and 4.29; HP256 has only eight randomized bases
  and therefore a comparatively uncertain variance estimate.
- Stored the five comparisons, cumulative tables, endpoint summary, and caveat
  that this one-gauge test is not a physical ensemble determination of Z_chi
  under `/global/cfs/cdirs/m4559/xgao/runs/TEST/emt_disconnected_z2_z4/results`.
- The EMT-derived kinetic is positive (`Re K` about 0.436) with the current
  covariant-derivative convention.  Since the existing standalone factor uses
  `-2*Nc/((4*pi)^2*t^2*K)`, its sign convention needs a dedicated check before
  applying a physical ringed normalization; no factor was formed in this test.

## 2026-07-13: S8T8 8192-solve EMT and proton T44 diagnostic

- Added reusable analysis helpers for deterministic time-stratified point
  sources, the proton `PpUnpol=(I+T)/4` C2 projection, optimized nonzero-q
  ratios, T44-only disconnected reads, source-translation covariance,
  jackknife, and two-way source/base bootstrap errors.
- Corrected the disconnected proton builder to use `(I+T)/4` rather than the
  source interpolator gamma label as its C2 sink projection, and added an
  independent `EMT_1PT_QZ_MAX` control for planar momentum grids.
- Completed 96/96 disconnected chunks for six 8192-solve measurements with
  nine `q=(qx,qy,0)` momenta and 32/32 connected chunks for 128 point sources
  at `tsep=2,3,4`; all pre-production smokes and finalizers passed.
- At fixed 8192-solve cost, pure Z2 and Z4 have equal stochastic efficiency
  within bootstrap uncertainty.  HP16 remains worse than pure noise, while
  HP256/pure SEM-squared ratios are about 17.5 (Z2) and 24.4 (Z4), even after
  increasing HP256 from 8 to 32 complete randomized bases.
- Wrote nine PNG/PDF figures, complex-valued T44 HDF5/CSV results, and reports
  under `/global/cfs/cdirs/m4559/xgao/runs/TEST/emt_s8t8_t00_qxy1_8192`.
  These outputs are bare, unringed, unrenormalized fixed-gauge diagnostics and
  do not include gauge-ensemble fluctuations.

## 2026-07-14: Diagnose spatial versus four-dimensional HP

- Repeated the Z4 HP16 and HP256 measurements with the isotropic
  `interleaved_xyzt_binary_projected_to_evenodd` ordering at 8192 solves,
  matched configuration, stream, base indices, action, flow schedule, and
  nine-momentum grid to the spatial-ordering benchmark.
- Verified the standalone ringed and EMT-derived kinetic contractions on the
  same base and all 16 HP vectors: all `16x2x8` values agree to maximum absolute
  difference `3.11e-15`, excluding a contraction, sign, or normalization bug.
- At `t_f/a^2=0.207936`, fixed-cost variance relative to pure Z4 is `0.311`
  for 4D HP16 and `0.378` for 4D HP256, versus `1.625` and `24.356` for the
  corresponding spatial orderings.  Paired base bootstraps confirm that the
  ordering change, not different random sources, drives the improvement.
- 4D HP16 improves every absolute time slice in this test, with variance ratios
  `0.382--0.462`.  The spacetime-averaged 4D HP256 result also benefits from
  cross-time covariance and is not uniformly better than pure Z4 per slice.
- Added HP-geometry tests showing that isotropic 4D HP16 cancels nearest
  neighbors in all four directions and 4D HP256 cancels displacements one
  through three, whereas spatial HP leaves purely temporal displacements
  completely uncancelled.
- Stored canonical data, bootstrap tables, figures, implementation crosscheck,
  and the technical report under
  `/global/cfs/cdirs/m4559/xgao/runs/TEST/emt_s8t8_t00_qxy1_8192/results/hp_ordering_4d`.

## 2026-07-14: Make isotropic 4D HP the flowed EMT default

- Changed the disconnected EMT library, Perlmutter Python entry point, and
  shell wrapper default from the time-independent spatial ordering to
  `interleaved_xyzt_binary_projected_to_evenodd`.
- Kept spatial and alternative 4D orderings available through explicit
  `EMT_1PT_HP_ORDERING` selection; qTMD and standalone ringed defaults were
  already four dimensional and were left unchanged.
- Updated root/workflow documentation and the HP production example, and added
  regression coverage for the library, Python, and shell defaults.

## 2026-07-14: Document EMT Gamma basis and raw-bilinear analysis

- Added one analysis-facing reference for the explicit DeGrand--Rossi Gamma
  matrices, the 16-label PyQUDA bitmask order, axial sign transform, and raw
  tensor convention shared by connected pion/proton and disconnected EMT.
- Added executable NumPy/HDF5 examples that reconstruct every symmetric
  `Tmunu` component from derivative primitives, including the distinct pion,
  proton, and disconnected dataset-axis orders and spatial-volume convention.
- Documented how the same primitives produce symmetric axial one-derivative
  operators and the six local tensor-current channels.
- Measured current single-file storage weights.  The S8T8 8192-solve EMTc is
  78.886% raw derivative and 19.722% raw local bilinears; all averaged
  primitives and `Tmunu` copies together are only 0.014%.  Connected scientific
  arrays approach the logical `64:16:16:1` derivative/local/Tmunu/chi ratio.

## 2026-07-14: Add the guided disconnected quark EMT workflow

- Reorganized the English application README around an executable sequence: S8T8
  smoke, shard/finalize checks, equal-solve pure/4D-HP16/4D-HP256 production,
  complete-base convergence analysis, and optional proton C2/3pt assembly.
- Added `emt_quark_1pt_convergence.py`, which compares embedded ringed kinetic
  and one selected symmetric EMT component without loading the full 16x4
  derivative basis.  It writes cumulative/endpoint CSV tables and headless
  PNG/PDF figures, and never treats a partial HP prefix as an estimator.
- Made the disconnected 3pt application quark-only by default so the primary
  benchmark does not require gluon production. `--include_gluon` retains the
  explicit advanced quark+gluon build.
- Recorded the distinction between stochastic fixed-gauge convergence and a
  gauge-ensemble vacuum-subtracted disconnected observable, together with the
  sample-log, base-range, time-alignment, and counter-noise pitfalls most
  likely to affect a first production test.

## 2026-07-14: Add l64 multigrid benchmark controls

- Made the disconnected EMT quark-one-point QUDA multigrid hierarchy an
  explicit `--mg-block` run parameter.  One or more coarsening levels are
  stored in shard/canonical provenance and therefore participate in the
  base-level sample-log fingerprint.
- Added the external l64c64a076 cfg1050 fixed-gauge test workflow: 16-GPU
  preflight, three-way
  MG timing with the first solve excluded, 2048-solve pure/4D-HP16/4D-HP256
  resume workflows, finalization, and ringed-kinetic/T44 convergence analysis.

## 2026-07-14: Separate l64 strange and light EMT tests

- Cancelled Perlmutter allocation `55900532` before changing the active run
  tree. Archived the existing `am_q=-0.015` MG results, tuning caches, logs,
  and three partial HP256 shard parts as the strange-quark test.
- Recorded that the archived HP256 base is incomplete: its sample log contains
  no completed-base line, so a future resume recomputes the full base rather
  than treating the 192-vector prefix as an estimator.
- Created an independent `am_q=-0.049` light-quark run tree with no inherited
  HDF5 data, sample log, MG result, runtime log, or QUDA tuning cache.
- Extended the application guide with ensemble-specific strange/light labels,
  the transition from S8T8 to `64^4`, a direct-to-l64 path, and the required
  preflight, fresh MG benchmark, smoke, and timing checks.

## 2026-07-14: Relate the EMT guide to the standalone ringed exercise

- Added an answer-first comparison of the standalone kinetic-only measurement
  and the full disconnected EMT primitive basis, covering their physics
  targets, resolved axes, computational work, dilution options, outputs, and
  intended downstream use.
- Clarified that the embedded EMT kinetic reuses the same inversion, fermion
  flow, and derivative applications, while the additional cost comes from the
  full Gamma contractions, momentum projection, and substantially larger I/O.
- Marked the historical standalone `.block*.h5` and per-block ringed-factor
  description as obsolete: current workflows share base/HP-part shards and
  form physical ringed factors only after the configuration average of `K`.

## 2026-07-14: Add m5208-local EMT run templates

- Created clean strange (`am_q=-0.015`) and light (`am_q=-0.049`) run-script
  trees under `m5208`, with independent empty data, log, analysis, MG-result,
  and tuning-cache directories.
- Added one local software configuration file per template so an existing
  personal QUDA/PyQUDA environment can be connected without depending on or
  modifying the repository's `m4559` environment helper.
- Updated the application examples to use `MEASUREMENT_ROOT`, `m5208` gauge
  and run paths, and the `m5208_g` GPU account. The allocation requests generic
  Perlmutter GPU nodes without a memory-capacity constraint.

## 2026-07-14: Avoid repeated EMT multigrid and covDev gauge setup

- Kept the initial full disconnected-EMT multigrid construction, but restored
  the unchanged original gauge before subsequent stochastic inversions with a
  thin MG update instead of refreshing near-null vectors and coarse operators.
- Reused one flowed-gauge context for all eight forward/backward covariant
  derivatives at each flow time.
- Validated pure Z4 and complete 4D-HP16 S8T8 outputs over all primitive and
  derived channels, including a two-rank decomposition test. Tight-solve
  reference/candidate results passed `rtol=1e-9, atol=1e-11`; the warmed HP16
  S8T8 total QUDA time decreased from 3.648 to 1.233 seconds.

## 2026-07-15: Add optional disconnected-EMT source-batched fermion flow

- Added the explicit `--flow-batch-size` production option, with a low-memory
  default of one and no environment-variable alias or automatic OOM fallback.
- Plain counter-Z4 sources may batch across pending bases; hierarchical-probing
  sources batch only inside one base and one HP-part interval. Existing shard,
  complete-base sample-log, and canonical schemas are unchanged.
- Each batch restores the unchanged original gauge once, performs sequential
  inversions, flows the interleaved `xi,eta` fields in one double-precision
  multi-field call, and shares one flowed-gauge context across all sources at
  each flow time.
- The preceding cfg1050 light-quark l64 benchmark found exact agreement for
  B=1,2,4,8. B=8 reduced the measured total to 3.19 s/source at about 59.6 GiB
  peak device memory; B=16 exhausted an 80 GB GPU. The application guide thus
  recommends B=8 as the measured 80 GB starting point and B=1 for 40 GB nodes.
- Formal-path S8T8 regression compared pure B=1/2/4 and complete HP16 B=1/4.
  All 27 canonical datasets passed `rtol=1e-9, atol=1e-11`; the largest raw
  difference was `1.96e-10` with relative L2 difference below `7.38e-12`.
- A 16-rank cfg1050 light-quark smoke then compared the same 16 counter sources
  through the formal B=1 and B=8 production/finalization paths. All 27 datasets
  were bitwise identical, both maximum true residuals were `9.286115e-11`, and
  sampled peak device memory was 33.8 GiB versus 60.7 GiB.
- In that short smoke, variable gauge/HYP and full resident-MG startup dominated
  total wall time, so its setup-inclusive 322 s versus 387 s is not a
  steady-state source-throughput comparison. The two directly targeted QUDA
  stages improved from 37.47 s to 14.44 s (`loadGauge+gFlow`), while total QUDA
  time decreased from 226.41 s to 216.42 s. The longer source-only benchmark
  remains the basis for the 3.19 s/source B=8 recommendation.

## 2026-07-15: Lighten disconnected-EMT batching internals

- Removed the unused single-source flow wrapper, an unused shard-method
  argument, and indirect raw-shape bookkeeping while preserving the source
  scheduler and HDF5 schemas.
- Made the primitive contraction require a caller-owned resident flowed-gauge
  context, eliminating its optional context branch and making the one-context
  per-flow-time invariant explicit.
- Consolidated the batching tests around small shared lattice, communicator,
  measurement, and output fixtures. A fresh S8T8 pure-B4 and complete-HP16-B4
  comparison passed for all 27 canonical datasets; HP16 was bitwise identical,
  while pure had maximum absolute difference `1.93e-10` and relative L2
  difference `3.76e-12`.

## 2026-07-15: Upgrade connected EMT and standalone ringed flow setup

- Connected pion and proton EMT now keep the initial full gauge/MG setup, use
  thin updates before later source and sequential inversions, and reuse one
  caller-owned flowed-gauge context for all eight covariant derivatives at a
  given flow time. Their existing one-branch `[forward,sequential]` flow is
  unchanged; no connected branch batching or new public option was added.
- Standalone ringed normalization gained the CLI/library `flow_batch_size`
  option with default one. Plain undiluted sources may batch across bases;
  HP and point spin-color diluted sources remain inside one base/part. Each
  batch performs one thin restore, sequential inversions, one interleaved
  double-precision multi-field flow, and one flowed-gauge context per output
  time. The performance-only batch size is absent from HDF5 provenance and the
  sample-log fingerprint.
- Shard metadata comparison now treats matching floating-point NaNs as equal;
  this is required for standalone multi-part finalization because the
  undefined flow-zero normalization factor is intentionally stored as NaN.
- S8T8 reference/candidate comparisons passed for pion C2 and all primitive/
  derived C3 data, proton `t_sep=2,3,4` with both flavors and two
  polarizations, and standalone plain, HP16, and point spin-color dilution at
  `B=1,2,4`. The largest connected relative L2 difference was `1.88e-10`;
  standalone relative L2 differences stayed below `4.30e-12`. Pion and
  standalone two-rank checks also passed.
- The cfg1050 light-quark l64 standalone scan found warmed median costs of
  5.31, 4.32, 3.88, and 3.64 seconds/source for `B=1,2,4,8`, with measured
  device memory of 26.6, 28.7, 34.0, and 44.8 GiB/GPU. This supports `B=8`
  on 80-GB GPUs and `B=4` as a 34.0-GiB starting point that still requires a
  local smoke on 40-GB GPUs; the default remains one.
- A persistent gluon gauge-flow context reproduced Wilson/Symanzik outputs
  exactly for one and ten output steps. After prewarming both paths and five
  interleaved repetitions, its end-to-end change ranged from a small slowdown
  to about a 3% speedup and was order dependent. It missed the required 10%
  threshold, so production gluon code was intentionally left unchanged.
- The planned l64 connected timing could not be completed at 16 ranks: even
  after omitting the unchanged C2 contraction, the existing 24-field
  forward/sequential flow exhausted an 80-GB GPU in both the reference and
  optimized pion paths. This is an independent connected-memory limitation;
  S8T8 remains the numerical validation for this change.

## 2026-07-15: Share the EMT runner with kinetic-only standalone ringed

- Refactored `EMTDisconnectedQuark1pt` so counter noise, base/HP-part
  scheduling, inversion, batched fermion flow, flowed-gauge contexts, sample
  logs, and shard writes are inherited production infrastructure.
- Replaced the standalone implementation with `RingedQuark1pt`, which emits
  only the direct four-vector-diagonal kinetic contraction and never allocates
  the full 16-local/64-derivative EMT primitive output.
- Removed standalone spin-color dilution, stored/computed ringed factors, the
  ensemble analyzer, and the old public class/function APIs. The canonical
  file now contains only kinetic data and source/base/HP bookkeeping.
- Detailed production timers remain available with
  `PYQUDA_MEASUREMENT_TIMERS=1` but are disabled by default.
- The S8T8 reference/candidate matrix covered full EMT and kinetic-only
  standalone pure, HP16, and HP256 measurements through batch sizes up to
  eight, plus pure/HP16 two-rank checks. Full EMT was bitwise identical; the
  largest standalone/embedded-ringed difference was `8.92e-16`, and the
  largest recorded true residual was `2.54e-16`.
- A 16-rank cfg1050 light-quark l64 validation on four 80-GiB A100 nodes then
  passed full EMT pure/HP16 at `B=8` and standalone pure `B=1/8` plus HP16
  `B=8`. Full EMT remained bitwise identical; standalone maximum absolute and
  relative-L2 differences were `8.88e-16` and `5.23e-17`, and the maximum true
  residual was `9.85e-13` at solver tolerance `1e-12`.
- Standalone warmed costs were statistically unchanged by the refactor:
  reference/candidate values were `3.586/3.573 s/source` for pure `B=1`,
  `2.109/2.120 s/source` for pure `B=8`, and `2.053/2.063 s/source` for HP16
  `B=8`. The sampled B=8 peak was `60705 MiB/GPU` on 80-GiB A100s. An otherwise
  identical 40-GiB run failed in `performGFlowQuda` after all inversions, so
  B=8 is not safe for this l64/two-level-MG setup on 40-GiB nodes.
# 2026-07-15: consolidate disconnected shard utilities

- Merged the base/HP-part shard, atomic HDF5, sample-log, and finalizer
  validation helpers into `Disconnected_utils_vibe_develop.py` alongside the
  counter-noise and HP bookkeeping they consume.
- Removed the old `disconnected_shards.py` module and updated EMT, qTMD,
  standalone ringed, application entrypoints, and tests to use the consolidated
  utility API without an import compatibility layer.
- S8T8 reference/candidate validation used solver tolerance `1e-15`.  The
  single-rank matrix covered EMT/ringed pure and HP16 plus GI-PDF qTMD: all
  127 numerical datasets were bitwise identical across 42 solves per runtime,
  with maximum true residual `6.30e-16`.  A two-rank, two-GPU smoke covered all
  three workflows: all 84 numerical datasets were bitwise identical and the
  maximum true residual was `2.15e-16`.
## 2026-07-15: EMT correctness and focused runtime cleanup

- Made proton `CG_GaussSmear=False` a genuine point workflow: source, C2 sink,
  sequential propagator sink, and sequential-source smearing are all disabled.
  Added separate source/sink/sequential smearing provenance and `POINT` default
  tags for this mode.
- Extracted the calculation-only proton C2 contraction into
  `proton_utils_vibe_develop.py`; proton EMT and qTMD now call the same kernel.
- Unified connected pion/proton multigrid selection as the CLI-only
  `--mg-block` interface, with default `8.8.4.4` and explicit `none` support.
- Added instance-local backend/dtype/device-or-queue Gamma caches for EMT and
  standalone ringed contractions.
- Removed the redundant thin restore before the first inversion after the
  initial full gauge/MG load. Later inversions still restore the original gauge
  after any flowed-gauge context.
- Added strict disconnected-loop Fourier provenance. Analysis now converts an
  origin-based absolute-time loop to each C2 source using the spatial phase and
  periodic time roll; old loops without the new provenance are rejected.
- Classified multigrid hierarchy, solver tolerance and maximum iterations as
  runtime solver controls rather than measurement identity. Resume/finalize
  may mix them across completed bases. They are supplied only when the solver
  is created and are absent from shards, sample-log identity and canonical
  disconnected files.
- S8T8 reference/candidate checks at solver tolerance `1e-15` found bitwise
  identical disconnected EMT, standalone ringed, pion C2/C3, proton C2/C3,
  and all primitive/derived channels.  The new point-proton path completed a
  real GPU smoke with all three smearing flags false.  Two-rank checks agreed
  with single-rank results at relative L2 differences of order `1e-15` or less.
- On l64 cfg1050 at light mass `-0.049` and tolerance `1e-12`, the same eight
  counter sources gave bitwise-identical reference/candidate disconnected EMT
  and standalone ringed files.  Full connected l64 validation is presently
  memory-blocked independently of this refactor: the existing pion C2 einsum
  OOMs even on 80-GiB A100s, and skipping C2 then reaches an 80-GiB OOM in the
  24-field connected fermion flow.  The complete connected numerical gate is
  therefore supplied by S8T8, while the l64 resource failure is retained in
  the external validation logs.
## 2026-07-16: Simplify EMT interfaces and disconnected bookkeeping

- Removed the pion EMT spin-projection argument, `.spin*` output suffix and
  HDF5 attribute. Pion channel identity now comes from the source and sink
  interpolators encoded in the canonical connected tags.
- Routed pion C2 and gluon EMT host transfers through the shared
  backend-independent array conversion helper and removed stale copied pion
  module documentation.
- Removed unused proton EMT propagator-saving state and duplicate boost names;
  connected proton EMT now uses only `boost_in` and `boost_out`.
- Upgraded disconnected shards to base/HP-only bookkeeping. EMT, qTMD and
  standalone ringed canonical files no longer persist the exactly
  reconstructible source index, and analysis reconstructs it as
  `base_noise_index * hp_vectors_per_base + hp_index`.
- Removed obsolete shard fields for per-HP solve counts and spin-color
  dilution, and moved the qTMD Wilson-index generator out of the shared
  disconnected noise/shard utility.
- Fixed an additional gluon backend boundary exposed by the GPU regression:
  an `opt_einsum` host result is converted to the active accumulator backend
  before addition, while the gathered result uses `array_to_numpy`.
- Completed the requested S8T8 matrix at solver tolerance `1e-15` with both
  one rank (`1.1.1.1`) and four ranks (`2.2.1.1`). Reference/candidate outputs
  were bitwise identical in both layouts across EMT, ringed, qTMD, pion and
  gluon workflows. The one/four-rank maximum relative L2 difference was
  `8.31e-15`, and all 80 HDF5 files opened successfully with no per-rank
  duplicate outputs.

## 2026-07-16: Make pion channel provenance explicit

- Removed the unused proton qTMD `pos_boost`/`boost_out` measurement aliases
  and `save_propagators` state.  Proton C2 now consumes `boost_in` directly;
  application-owned sequential smearing continues to use `boost_out`.
- Separated pion qTMD/PDF and EMFF smearing identity from channel identity.
  C2 tags now encode the source interpolator, while three-point tags and
  sample logs encode both source and sink interpolators.  HDF5 attrs record
  source, sink and operator/current Gamma conventions explicitly.
- Routed the active proton drivers through the centralized 16-Gamma basis and
  made the Perlmutter proton qTMD/PDF serial HDF5 publication rank-0-only.  The
  previous rank-distributed writer omitted one flavor in a one-rank run.
- Rebuilt pion qTMD, pion EMFF and proton qTMD PDFs.  The S8T8 regression used
  tolerance `1e-15` and both one rank (`1.1.1.1`) and four ranks (`2.2.1.1`).
  Reference/candidate datasets were bitwise identical at fixed layout; the
  largest one/four-rank relative L2 difference was `1.06e-13`, the maximum
  absolute difference was `5.69e-14`, and the largest parsed true residual was
  `9.78e-16`.

## 2026-07-16: Unify pion C2 and soft-factor channel conventions

- Replaced the duplicate qTMDWF C2 contraction with the shared pion C2 kernel,
  which allocates the local time axis from `latt_info.size[3]` and records the
  fixed source Gamma plus canonical 16-Gamma sink basis.
- Removed the `dagger_of_sink` source mode. qDA now computes local C2 for each
  explicit `da_src_gammalist` entry and writes one 16-sink-Gamma file per
  source; the charm-mass workflow uses fixed source `5`.
- Made soft-factor pion and insertion channels explicitly paired. Its Gamma
  insertions now use the canonical raw PyQUDA basis, including raw `Y5`, and
  the HDF5 schema stores pair members and the raw-to-physical transform.
- Changed pion EMT C2 and pion qTMD contractions to process one sink Gamma at
  a time, eliminating the approximately 16-fold propagator-sized temporary
  while preserving the final Gamma axis.
- Completed the S8T8 numerical gate with one rank (`1.1.1.1`), four ranks
  using time decomposition (`1.1.1.4`), and an additional four-rank qTMDWF
  spatial-decomposition smoke (`2.2.1.1`).  Across 14,546 compared numerical
  datasets, the maximum absolute difference was `5.68e-14` and the maximum
  relative L2 difference was `2.62e-13`.  The old qTMDWF implementation failed
  in the four-rank time-decomposed run by attempting to store a local-time
  array in a global-`Nt` buffer; the shared C2 kernel completed correctly.
- An l64 cfg1050 light-quark memory smoke with a resident two-level multigrid
  hierarchy completed on 80-GiB A100s.  Device allocation rose from about
  `35.8 GiB` after inversion to `49.5 GiB` after the pion EMT C2 and remained
  unchanged through the minimal pion qTMD contraction, confirming that neither
  optimized path recreates a 16-Gamma propagator stack.  The same setup cannot
  fit on 40-GiB A100s because the baseline propagators, sink smearing and
  resident multigrid hierarchy already exceed that memory budget.

## 2026-07-16: Repair the qDA straight-link GI operator

- Restored the shared pion-C2 `dagger_of_sink` mode,
  \(\Gamma_{\rm src}^{(g)}=\gamma_5\Gamma_g^\dagger\gamma_5\), as a
  backend-aware relational source mode. qDA again uses this paired-channel C2
  convention; its explicit `5/X/T` list remains specific to the nonlocal DA
  outputs.
- Replaced qDA's invalid reference to a nonexistent TMD staple transporter
  with the existing `create_fw_prop_PDF_GI` straight-link transport. Both CG
  and GI now act on the forward propagator, while the backward line remains
  undisplaced. Positive and negative longitudinal branches restart from the
  original forward propagator.
- Kept the qDA HDF5 layout/provenance and sample-log identity unchanged. Old
  nonzero-separation CG output used the opposite transported line and must be
  regenerated; the former GI branch could not run.
- The S8T8 numerical gate used tolerance `1e-15`, source Gammas `5/X/T`, all
  16 sink Gammas, `z=0,+1,+2,-1,-2`, and `pz=0,1`. One rank (`1.1.1.1`) and
  four ranks (`2.2.1.1`) agree with maximum relative L2 difference
  `3.42e-16`; final true residuals were `5.35e-16` and `8.53e-16`.
- A deterministic local SU(3) transformation applied after propagator
  generation changed CG by at least `0.2606` in relative L2 while GI changed
  by at most `1.50e-16`. The transformed gauge was explicitly loaded once
  before its covDev calls, matching the resident-gauge precondition used by
  the production helper.

## 2026-07-16: Complete the disconnected EMT two-sided derivative

- Replaced the historical right-acting-only disconnected derivative with the
  full fixed-time, finite-momentum `overleftrightarrow_D` trace.  The stored
  primitive includes the closed-loop Wick minus exactly once.
- Used gamma5 hermiticity to reconstruct the left-acting term from the existing
  right-acting contraction at the opposite momentum.  The production path
  still performs eight `covDev` calls per source and flow time; only the small
  momentum projection is augmented when a requested `-q` is absent.
- Added the exact `Gamma -> gamma5 Gamma^dag gamma5` partner/sign map to the
  canonical raw PyQUDA Gamma metadata.  Embedded and standalone ringed kinetic
  measurements now use the same completed vector-diagonal primitive.
- Corrected loop Fourier provenance to the phase actually generated by
  PyQUDA, `exp(+2 pi i q.(x-origin)/L)`.  The source-origin rephasing factor
  remains negative.  Loop provenance is v2 and EMT operator schema is v5;
  one-sided shards, sample logs and canonical files must be regenerated.
- The S8T8 numerical gate used tolerance `1e-15`, seven momenta
  `q=0,+/-x,+/-y,+/-z`, pure Z4 and HP16, and both one rank (`1.1.1.1`) and
  four ranks (`2.2.1.1`).  The maximum one/four-rank relative L2 differences
  were `4.88e-15` for EMT, `1.06e-16` for standalone ringed, `3.00e-15` for
  pion connected and `2.00e-13` for proton connected.  Vector loops obeyed
  `L(q)=L(-q)^*`, zero-momentum vector loops were real, and embedded and
  standalone kinetic arrays agreed element by element.  An actual local
  `SU(3)` transformation of the S8T8 gauge, counter source and solved field
  changed the completed derivative by relative L2 `3.72e-16` (one rank) and
  `2.92e-16` (four ranks).
- Added an independent actual-field S8T8 comparison against the explicit
  left/right stochastic estimator using extra `covDev(xi)` calls.  With 256
  counter-Z4 sources, all 16 Gamma channels, four derivative directions,
  `q=0,+/-x,+/-y,+/-z`, and every time slice, the gamma5-reconstructed and
  direct estimators differed globally by `0.981` paired standard errors for
  Wilson--clover solved fields (`tol=1e-15`) and by `0.978` for the
  identity-propagator control.  Because the two estimators differ per source,
  this paired ensemble test, rather than elementwise source equality, is the
  relevant numerical check.  One-rank and four-rank estimator means agreed at
  relative L2 about `2.0e-16`.
- Documented why the explicit `covDev(xi_f)` left/right estimator and the
  production gamma5/opposite-momentum reconstruction are distinct at finite
  noise but have the same expectation.  The authoritative disconnected note
  now derives both estimators and states the paired-SEM validation criterion;
  the application, Gamma-basis, pion and proton EMT documents carry concise
  cross-references and the S8T8 result.

## 2026-07-16: Simplify pion source-Gamma modes

- Removed the redundant `fixed_g5` alias and replaced every active pion
  qTMD/PDF, EMFF and current-response default with the explicit canonical
  source label `5`.  New tags use `src5`, while HDF5 provenance records
  `source_gamma_mode=fixed` and `source_gamma_label=5`.
- Removed the unused `same_as_sink` relational mode.  Shared pion
  contractions now accept only a fixed canonical Gamma label or the
  `dagger_of_sink` paired-channel convention used by qDA local C2.
- Kept qDA nonlocal `5/X/T` source scans and all 16 sink channels unchanged.
  Old source-mode strings and `.srcfixed_g5` output identities are not
  compatible.
- The S8T8 gate used tolerance `1e-15`, one rank (`1.1.1.1`) and four
  ranks (`2.2.1.1`).  Old `fixed_g5` and explicit `5` qTMD/PDF and
  EMFF outputs were bitwise identical at fixed MPI layout; qDA
  `dagger_of_sink` and `5/X/T x 16` outputs were also unchanged.  Maximum
  one/four-rank relative L2 differences were `1.03e-13` for pion qTMD/PDF,
  `8.30e-14` for EMFF and `3.41e-16` for qDA.

## 2026-07-16: Correct pion qTMD/PDF unequal-boost lines

- Unified connected pion qTMD/PDF with the pion EMT line convention: the
  positive-boost propagator is the spectator and the negative-boost propagator
  is the active operator line. Equal boosts reuse one inversion; unequal
  boosts now perform the physically required second source inversion.
- The positive spectator is positive-boost smeared at the fixed sink, the
  sequential source receives negative outer smearing, and CG/GI qTMD plus
  CG/GI PDF all displace or transport the negative active propagator.
- Added explicit `--pos-boost` and `--neg-boost` application arguments, encoded
  both in the default setup tag, and recorded the two boosts and line convention
  in C2/three-point HDF5 output. Historical unequal-boost output must be
  regenerated; zero-boost output remains the numerical reference.
- Removed the duplicate `application/pion_TMD_CG` workflow. The canonical
  `application/pion_TMD` entry point now covers CG/GI qTMD and CG/GI PDF.
- The S8T8 gate used solver tolerance `1e-15`, one rank (`1.1.1.1`) and four
  ranks (`2.2.1.1`). Zero-boost reference/candidate outputs and the opposite-
  boost shared-helper/explicit-line outputs were bitwise identical. The old
  opposite-boost single-source path differed in C2 and all four operator paths.
  The largest one/four-rank relative L2 difference was `3.22e-15`, and the
  largest true solver residual was `9.52e-16`.

## 2026-07-16: Share the pion C2 contraction kernel

- Removed the duplicate pion EMT C2 Gamma insertion, gamma5-hermitian backward
  line, spin-color trace, momentum projection, and MPI gather implementation.
  The EMT wrapper now handles only endpoint smearing, phase selection,
  source-time rolling, and I/O around the shared pion C2 kernel.
- Confirmed that pion qTMDWF, qTMD, and EMFF already use the same shared
  contraction interfaces.  Added guards against reintroducing workflow-local
  C2 contraction algebra or a 16-Gamma propagator-sized intermediate.
- Soft-factor C2 was deliberately left unchanged for a later, separate audit.
- The S8T8 gate used solver tolerance `1e-15`, fixed source labels `5/X/T`,
  `dagger_of_sink`, equal and opposite boosts, one rank (`1.1.1.1`), four-rank
  time decomposition (`1.1.1.4`), and four-rank spatial decomposition
  (`2.2.1.1`).  Across 70 dataset comparisons, the largest
  reference/candidate relative L2 difference was `2.84e-16`; the largest
  one/four-rank relative L2 difference was `2.50e-16`.

## 2026-07-17: Share the memory-light qTMDWF Gamma contraction

- Replaced the Frontier qTMDWF `G16_fw_Gsrc` allocation with a shared
  two-stage spin-color contraction.  The largest Gamma-dependent temporary is
  now `[16, local sites]`, rather than sixteen complete propagators.
- Switched both Aurora k0/k4 applications from duplicated inline contractions
  to the same shared kernel.  Platform scripts now retain only Wilson-line
  transport, scheduling and output responsibilities.
- Kept the source Gamma, all 16 sink Gamma channels, momentum phases, CG
  shifts, Wilson-index order, time rolling, and HDF5 layout unchanged.
- Added an algebraic old/new estimator comparison and static memory guard.
  Actual-field S8T8 one-rank and four-rank results are recorded under
  `runs/TEST/frontier_qtmdwf_memory_s8t8_validation/`.
- The actual-field gate used solver tolerance `1e-15`, both transverse
  directions, `bT=0,1`, `bz=0,+/-1`, zero/nonzero momentum, all 16 Gamma
  channels and every time slice.  Old/new relative L2 differences were
  `9.99e-17` (one rank) and `2.27e-16` (four ranks, `2.2.1.1`); the new
  one/four-rank difference was `8.11e-16`.

## 2026-07-17: Fix pion resume, backend allocation, and response time

- Made connected pion qTMD and EMFF sample logs actual source-level resume
  state.  Rank 0 reads exact lines once, completed work is skipped before
  inversion, and a line is durably appended only after all requested outputs
  close.  No HDF5 probe, fingerprint, marker, or concurrent-log protocol was
  added.
- Centralized pion host conversion and queue-aware allocation.  EMFF no longer
  uses backend-specific `asnumpy` or unbound `zeros`; soft-factor Gamma stacks
  are constructed on the propagator backend and exact SYCL queue.
- Defined every first-order and nested current-response tau window relative to
  the source.  Projectors alone use
  `tau_abs=(source_time+tau_rel) mod Nt`; saved C2/C3/response arrays are rolled
  to source-relative time.  First-order/current-current schemas are now
  versions 3/2 and store source position plus relative/absolute tau lists.
- Fixed both response applications to broadcast rank-0 gathered C2/C3/response
  arrays before all ranks select sink times.  The previous code failed on
  non-root ranks in an actual four-rank run.
- Added exact-log, queue-allocation, nonzero-source-time, periodic-wrap, and
  schema tests.  The CPU test suite passes with 255 tests and 12 skips.
- The S8T8 gate used tolerance `1e-15` with one rank and four ranks
  (`2.2.1.1`).  Reference/candidate datasets were identical.  Candidate
  one/four-rank relative L2 differences were at most `7.25e-14`; the maximum
  absolute difference was `1.14e-13`.  Repeated qTMD/EMFF runs skipped before
  inversion and did not change any HDF5 size or modification time.

## 2026-07-17: Unify pion bilinear and nonlocal-line infrastructure

- Added one shared pion 16-Gamma contraction core for propagator pairs and
  prebuilt backward lines. Pion C2, qTMD, EMFF, qTMDWF, and qDA now use this
  implementation; qTMD no longer broadcasts complete gathered results back
  to non-root ranks solely for serial output.
- Unified qDA with qTMDWF: the positive-boost forward line is the fixed
  spectator and CG/GI displacement acts on the negative-boost backward active
  line. Positive and negative longitudinal branches restart from the original
  backward propagator.
- Replaced soft-factor rank-local transverse `roll` operations by global
  `LatticePropagator.shift`, which is correct when the transverse direction is
  MPI decomposed.
- Introduced `FlowedFermionBilinearKernel`; connected pion and proton EMT no
  longer inherit disconnected noise, shard, resume, or finalizer machinery.
- Consolidated Aurora/Frontier qTMDWF production around one application runner,
  one root-written 16-Gamma HDF5 file, and exact-line source resume. qDA and
  charm-mass workflows use the same lightweight log helpers.
- Split pion response calculations from analysis: production inversions and
  contractions remain in `pyquda_measurement_utils`, while tau selection,
  rolling, ratios, channel extraction, explicit sums, and HDF5 writers live in
  `application/analysis_helper/pion_current_response_analysis.py`.
- S8T8 reference/candidate checks at solver tolerance `1e-15` passed for
  connected pion EMT, disconnected quark EMT, and standalone ringed output in
  one-rank and spatially decomposed four-rank layouts. Reference and candidate
  files were bitwise identical within each layout; the largest cross-layout
  relative L2 difference was `4.09e-15`.
- A full nonzero-`bT` soft-factor contraction agreed between one and four
  ranks with relative L2 difference `8.89e-16`. qTMDWF and qDA CG contractions
  were exactly equal for the same sources in both layouts; their cross-layout
  relative L2 difference was `1.22e-16`. The maximum true solver residual was
  `9.90e-16`.
- A 16-rank `2.2.2.2` MPS smoke also produced one root-written qTMDWF HDF5
  containing all 16 Gamma groups. Its soft-factor/qTMDWF/qDA relative L2
  differences from one rank were at most `1.69e-15`.
- The final CPU regression suite passes with 260 tests and 12 skips.

## 2026-07-17: Lighten pion qTMD configuration and backend helpers

- Removed the unused `pf`, qTMD/PDF momentum-list, and insertion-time copies
  from `pion_TMD`; the application runner remains the owner of sequential
  sink kinematics, Fourier phases, output labels, and time trimming.
- Removed the unused pion EMFF/qDA `save_propagators` parameter and the unused
  lattice-info argument from the soft-factor propagator loader.
- Centralized small-matrix backend and SYCL-queue placement in
  `matrix_on_backend` and `matrix_stack_on_backend`.  Pion contractions,
  response sources, and soft-factor contractions now use the same helpers.
- Kept the soft-factor `momentum_tag`, qTMDWF `eta`, qTMD final-momentum
  convention, single-\(t_{\rm sep}\) scheduling, and Wilson-index order
  unchanged.
- The full CPU suite passes with 263 tests and 12 skips.  A one-rank S8T8
  reference/candidate run at tolerance `1e-15`, nonzero momentum, and
  \(t_{\rm sep}=3\) compared 65 qTMD/PDF files (1872 datasets) and two EMFF
  files (864 datasets); all numerical datasets were bitwise identical.
- A fresh four-rank S8T8 run with geometry `2.2.1.1` used the same tolerance,
  momentum, and separation.  Reference/candidate comparisons were bitwise
  identical for qTMD/PDF (65 files, 1872 datasets), EMFF (2 files, 864
  datasets), first-order response (1 file, 31 datasets), and soft factor
  including nonzero transverse displacement (6 files, 25 datasets).

## 2026-07-17: Correct proton qTMD endpoint boost usage

- Kept `boost_in` as the application-owned source-smearing parameter and made
  the shared proton qTMD measurement use `boost_out` for the C2 sink, matching
  the existing fixed-sink sequential-source convention.
- The canonical production applications still use zero source and sink
  boosts, so their standard output is unchanged.  Older manually produced
  unequal-boost C2 data used the source boost at both endpoints and must be
  recomputed.

## 2026-07-17: Unify four-dimensional HYP smearing

- Audited every active `gauge.hypSmear` call and fixed the final
  `dir_ignore` argument to literal `-1`.
- Removed the ringed and Aurora proton EMT environment overrides for that
  argument, and made HYP provenance explicitly say `dir_ignore=-1`.
- QUDA converts both the historical value `4` and `-1` to its internal
  four-dimensional sentinel `4`, so this cleanup changes interface identity
  and documentation but not the smeared gauge field.
- A one-rank S8T8 field-level comparison found bitwise-identical smeared links
  and plaquettes for `dir_ignore=4` and `dir_ignore=-1`.  The full CPU suite
  passes with 266 tests and 12 skips.
## 2026-07-18: Document raw and physical Gamma conventions

- Expanded `docs/EMT_gamma_and_raw_bilinears.md` into the central registry for
  `pyquda_bitmask16_with_physics_transform_v1`.
- Clarified that qTMD/PDF `corr[...,gamma,...]` stores the raw PyQUDA matrices
  used in production.  `gamma_basis_schema` is a version key; the exact
  convention is defined by `gamma_matrices`, `gamma_pyquda_ids`, and
  `physical_from_pyquda` in each HDF5 file.
- Corrected the pion/proton qTMD tables to show the raw signs
  `Y5_raw=-gamma_y gamma5` and `T5_raw=-gamma_t gamma5`, and distinguished raw
  tensor products from the optional Hermitian factor `1j`.

## 2026-07-23: Add mean-first disconnected EMT shards

- Added `save_raw_per_vector=False` to disconnected quark EMT production. Every
  shard stores complex128 means for local bilinears, derivative bilinears, and
  flowed-noise norm; enabling the flag additionally preserves the existing
  per-vector raw payload and source bookkeeping.
- Mean-only HP parts accumulate flow batches without allocating a complete
  part-sized raw buffer. Storage mode is part of the sample-log fingerprint,
  and shard metadata records the mean definition, vector count, axes, HP range,
  base, and part.
- Kept Tmunu derived from the unsymmetrized derivative primitive instead of
  duplicating it in shards. Raw-plus-mean files remain compatible with the
  current canonical finalizer, while mean-only input fails with an explicit
  unsupported-payload error.
- Added the `--save-raw-per-vector` application and wrapper flag, documented
  extensible base/part naming without planned-base-count tags, and added shard,
  batching, fingerprint, EMT-linearity, and finalizer regression coverage.
- Python compileall, shell syntax, diff hygiene, and a standalone HDF5 schema
  smoke passed. Full PyQUDA pytest/GPU regression still requires an Aurora
  compute allocation because MPICH cannot initialize in the login shell.
