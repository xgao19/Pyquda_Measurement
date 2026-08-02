# PyQUDA Measurement Session Memory

Last updated: 2026-08-02

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
- Spatial and 4D hierarchical probing both multiply a full-volume base source.
  Spatial HP is not time dilution. The standard flowed-EMT ordering is isotropic
  4D `interleaved_xyzt_binary_projected_to_evenodd`.
- Shards store base and HP indices; reconstruct the effective source index as
  `base * N_HP + hp`. A base is complete only after every planned HP part
  validates and the base-level sample log is written.
- Mean-only mode always stores one complex128 part mean. Raw mode stores both
  per-vector data and the same part mean. `block_interval_solves` controls part
  size independently of `flow_batch_size`.
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
