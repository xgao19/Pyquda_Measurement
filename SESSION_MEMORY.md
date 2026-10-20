# PyQUDA Measurement Session Memory

Last updated: 2026-10-20

This file contains only reusable environment facts, stable conventions, and
pitfalls that are easy to repeat.  Commit history and completed-work summaries
belong in `log.md` and Git history.

## Memory Policy

- Read this file before resuming repository work.
- Do not turn it into a changelog, validation diary, or list of old outputs.
- Add an item only when it is likely to change a future implementation or
  prevent a repeated mistake.
- Before a commit, update `log.md`; update this file only if reusable knowledge
  actually changed.

## Runtime Environments

### Perlmutter

- Use `/global/cfs/cdirs/m4559/xgao/software_gradientflow`, its
  `venv-quda-develop`, `quda-develop/install`, and editable `PyQUDA-develop`;
  do not fall back to the old `m3760` stack.
- Activate with `systems/perlmutter/activate-venv-quda.sh`, then verify `python`
  and `QUDA_PATH`. Keep `MPICH_GPU_SUPPORT_ENABLED=1` and verify MPI-enabled
  h5py when parallel HDF5 is required.
- `login32` is suitable only for small GPU smoke tests. Multi-rank work requires
  a valid Slurm allocation.

### Aurora

- The checkout is
  `/lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement`;
  activate it with
  `/lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh`.
- Use `backend="dpnp"`, `backend_target="sycl"`, PALS inside a compute
  allocation, and MPI-enabled h5py. Do not introduce Torch XPU paths unless
  explicitly requested.
- FLAT production uses `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
  `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, integer tile affinity,
  `QUDA_ENABLE_P2P=0`, and `QUDA_ENABLE_TUNING=1`.

### Smoke Inputs

- Small flow gauge: `test_gauge/S8T8_wilson_b6.0`.
- Main workflow gauge: `test_gauge/S8T32_wilson_b6.cg.1e-08.0`.
- Baseline flow check: `systems/perlmutter/check-gradient-flow.sh`.

## Repository Hygiene And Verification

- Preserve unrelated user changes; never clean, stash, reset, or rewrite them
  implicitly. All code comments must be in English.
- Never commit generated HDF5 data, runtime logs, profiles, tuning/cache
  directories, `__pycache__`, or LaTeX intermediates.
- Convert backend arrays to NumPy explicitly. If bytecode cache writes are not
  possible, use `ast.parse` for a write-free syntax check.
- QUDA shutdown leak warnings are not sufficient evidence of failure; check the
  exit code and physics output.
- Run lightweight regressions with `python tests/run_smoke_tests.py` after
  activating the appropriate environment.
- Treat this repository as read-only while testing external l80 workflows unless
  the user explicitly authorizes repository changes.

## Local LaTeX Documents

- On Aurora, source `software_gradientflow/activate-latex.sh` for Tectonic;
  its executable and macro/font cache live under `software_gradientflow/latex`,
  outside this repository and independently of the PyQUDA environment.
- Compile with `tectonic path/to/document.tex`; use `--only-cached` after a
  successful first build to avoid network access. Track source and final PDF,
  not LaTeX intermediates (`docs/.gitignore`).

## Topological Charge Loops

- `TopologicalCharge1pt` stores finite-flow clover density with Hermitian
  `H=-i*(C-Cdag)_traceless/8`, directions xyzt and epsilon0123=+1. Its scalar
  schema is not an EMT tensor and is not already an extrapolated RGI result.
- Unlike historical EMT, every topology output interval uses the same
  `flow_substeps_per_interval`, including the first interval. Retain exactly
  one zero Fourier mode for total-charge diagnostics.
- The inspected installed QUDA `exponentiate_iQ` is singular at exactly zero
  force. Identity-link validation is t=0 only; use nontrivial fields for flow.
  This does not exclude zero Fourier momentum or zero total charge. Reject
  non-finite comparison data explicitly; matching NaNs must not count as pass.
- Avoid the inspected PyQUDA `qchargeDensity()` path: its density flag remains
  set after return. Use the manual clover loop; scalar `qcharge()` is the
  independent total-charge reference.

## EMT Conventions And Pitfalls

- The fifth `gauge.hypSmear` argument is QUDA `dir_ignore`; active 4D HYP uses
  `-1` and must not expose it as a casual runtime override.
- Active connected implementations are `pion_EMT_vibe_develop.py` and
  `proton_EMT_vibe_develop.py`. EMT outputs are HDF5-only.
- Measurement identity must include every independent axis. In multi-stream
  ensembles, stream and configuration must both appear in filenames, logs,
  metadata, and resume fingerprints.
- Proton setup tags encode preprocessing, smearing width, and explicit input and
  output boosts. C2/C3 tags additionally identify source, sink interpolator,
  polarization, and actual sink separation.
- A connected proton source job supplies a nonempty `tags` mapping keyed by the
  sink separations to compute. Mark a separation complete only after its HDF5
  file closes. Three-point files never embed C2 data.
- `PpUnpol` is `(C2[I] + C2[T])/4`, i.e. `(1+gamma_4)/4`; source interpolator
  label `5` is not a sink projector.
- Flow always measures before advancing: step 0 is unflowed and the first
  interval uses ten `epsilon/10` substeps. Later intervals default to one step;
  the fermion/gluon helpers can divide them with `substeps_per_interval`, while
  the gluon constructor reads `flow_substeps_per_interval`.
- Connected EMT uses fixed-sink convention B with
  `dst2 = gamma5 * seq_raw^dagger * gamma5` and `meson_sign = 1`.
- Proton left derivatives act on the raw sequential propagator before the final
  gamma5-hermiticity/index transform; differentiating finalized `seq_data`
  breaks gauge covariance.
- Pion sequential sources require active-line sink smearing. Pion EMFF C2 has no
  momentum transfer and uses source-side boosts at both ends.
- `_F_clover_traceless` projects each gluon field-strength matrix to `su(3)`; it
  does not make the final EMT tensor traceless.

## Disconnected EMT Loops

- Use decomposition-independent full-volume counter-based `Z4` noise. Its
  counter includes global coordinates, spin, color, configuration, base index,
  and stream salt; never reset the same local RNG seed on every rank.
- Supply configuration identity explicitly. Do not infer it from an environment
  variable or default it to zero.
- `EMTquarkLoop` and `EMTgluonLoop` are the canonical source-independent names;
  do not restore the retired `EMTc` or `EMTg` names.
- Loop files contain all absolute insertion times. Align to a hadron source with
  `tau_abs = (t0 + tau_rel) % Nt` and rephase momentum by
  `exp[-2 pi i q.(source-origin)/L]`; reject files without lattice and Fourier
  origin provenance.
- Fermion flow is four dimensional even when the observable is spatially summed
  at fixed time. The stochastic source must cover every time slice.
- Flow contractions replace QUDA's resident gauge. Restore the original gauge
  before the next inversion; the initial inversion can use the original
  resident gauge directly.
- Develop EMT, standalone ringed and qTMD share the Python HP default
  `nested_xyzt_checkerboard_v1`, with complete counts 2, 32 or 512. Shell
  wrappers do not choose an ordering. Old HP generators are not supported.
- EMT multiplier probing requires a validated shape-specific plan and a complete
  Fourier basis using the same base noise. Average over all plan colors;
  unscaled color masks would instead require summation. Standalone ringed and
  qTMD support Z_N and complete HP only.
- EMT point spin-color dilution applies all 12 projectors to each probe, sums
  projector contributions, and averages only over probes. The complete base,
  not its projector channels or probes, is one independent stochastic sample.
- Develop shards use `disconnected_1pt_base_parts_v4`, sample logs use
  `disconnected_sample_log_v2`. All new runs, including Z_N, require fresh
  output/setup identities; old files are read with the original checkout.
- Store one `vectors_per_base` count plus scheme, explicit dilution and relevant
  ordering/plan. Raw source indices are reconstructed as
  `(base_noise_index * vectors_per_base + probe_index) * factor + channel`,
  where factor is 12 for point dilution and 1 otherwise.
- EMT always stores complex128 part means; raw per-vector data are optional
  diagnostics. Base aggregation requires means and complete validated parts,
  and weights by actual probe count. A base is complete for resume only after
  the base-level sample log is written. Overlapping job ranges are unsupported.
- The EMT finalizer defaults to `.bases.h5`; `--raw-output` also writes raw
  canonical diagnostics and requires raw shards. Standalone ringed/qTMD retain
  raw data for their own finalizers. `block_interval_solves` controls part size
  independently of `flow_batch_size`.
- New EMT ringed/convergence/T44 analyses read `.bases.h5` directly. SEM uses
  independent complete bases; one base has undefined (NaN) SEM. Derive cost
  from `vectors_per_base` times the dilution factor, never a hardcoded HP table.
- Only MPI rank 0 opens serial quark/gluon HDF5 output files.
- Physical disconnected correlators require ensemble subtraction. If step1
  stores `L_avg=L_sum/Vs`, construct the hadron correlator as
  `Vs * (<C2 L_avg> - <C2><L_avg>)`; do not multiply local VEV or ringed kinetic
  normalization by `Vs` again.
- The quark loop stores the 16 local and `16x4` unsymmetrized derivative basis.
  Raw `Y5` and `T5` require minus signs for uniform `gamma_mu gamma5`; raw tensor
  masks are `[gamma_mu,gamma_nu]/2` without `i`.
- Time- or momentum-resolved derivative loops require both sides of
  `overleftrightarrow_D`. Reconstruct the left term from the right term at `-q`
  with `Gamma# = gamma5 Gamma^dag gamma5`; the stored loop contains exactly one
  closed-fermion-loop Wick minus.
- Loop projection uses `exp(+2 pi i q.(x-origin)/L)`, so source rephasing uses
  the opposite sign.
- Setup tags describe the measurement setup, not the planned total base count;
  bases may be extended without changing measurement identity.

## qTMD Conventions And Pitfalls

- Keep common CG/GI geometry and transport in `qtmd_operator_utils.py`;
  connected code must not import disconnected production modules.
- qTMDWF and qDA local pion C2 use fixed source Gamma labels and a full 16-Gamma
  sink scan. Put source/sink interpolators in tags, HDF5 provenance, and resume
  identity rather than a user-overridable smearing tag.
- Fixed-length GI staple indices are `[b_T, b_z, eta, transverse_direction]`,
  with even `b_z`, `eta >= abs(b_z)/2`, `b_T >= 0`, and length `2*eta + b_T`.
- `covDev` composition acts on endpoint fields in reverse geometric segment
  order. Keep geometric segments unchanged and reverse only endpoint transport.
- Prefer cached gauge-only staple transporters; retain the direct covariant-shift
  path for validation.
- The disconnected estimator is `xi^dagger P Gamma O_b eta`, with
  `eta=D^{-1}xi`; apply `O_b` to `eta`, never to `xi`.
- Reset the shifted solution before changing transverse direction when building
  disconnected `CG_qTMD`.
- Preserve the local-limit invariant
  `GI_PDF(bz=0) = CG_PDF(bz=0) = CG_qTMD(bT=0,bz=0)`.
- Disconnected qTMD uses source-independent tags, full-volume counter `Z4`,
  base/HP-interval shards, and explicit destination-side finalization.

## Ringed Normalization, Smearing, And Proton Memory

- Standalone ringed, EMT, and qTMD resume from fingerprinted base-level sample
  logs. Finalized bases must not depend on local shards remaining available.
- Multigrid blocks, solver tolerance, and maxiter are runtime controls, not
  disconnected measurement identity.
- Average the ringed kinetic expectation `K` over configurations before nonlinear
  normalization; never average per-configuration `1/K`.
- Keep `boosted_smearing(src, *, w, boost)` unchanged for fermions and
  propagators. Smear all propagator spin-color columns in one batched distributed
  FFT rather than repeating FFT setup per column.
- Wait on the owning SYCL queue at dpnp/NumPy distributed-FFT boundaries: before
  forward FFT inputs, after momentum multiplication, and before returning to
  QUDA. MPI barriers do not replace queue ownership fences.
- Do not add a persistent smearing device-kernel cache without validating SYCL
  queue and MPI-decomposition ownership.
- Preserve proton U insertion order `-(((R1 + R2) + R3) + R4)` and D insertion
  order `term2 - term1`. Construct and release Wick intermediates sequentially,
  with queue waits before release.
- The raw sequential builder treats the forward propagator as read-only; do not
  copy the full propagator. Wait on its queue and collect garbage after C2 before
  sequential construction.

## Lattice Data Preprocessing

- Active preprocessing lives in `Lat_Data_Preprocessing`; do not modify its
  archived `legacy` tree during active workflow maintenance.
- Connected proton EMT preprocessing is sample-log driven, reconstructs exact
  paths without globs, and handles C2 and C3 separately with antiperiodic
  temporal-boundary signs.
- Disconnected preprocessing keeps local/ringed normalization on spatially
  averaged loops and restores exactly one factor of `Vs` only when constructing
  hadron disconnected three-point functions.
