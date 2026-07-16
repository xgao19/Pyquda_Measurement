# PyQUDA Measurement Session Memory

Last updated: 2026-07-13

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

- The default software root is
  `/global/cfs/cdirs/m4559/xgao/software_gradientflow`; do not fall back to the
  old `/global/cfs/cdirs/m3760/xgao/software` stack.
- The default Python environment is
  `/global/cfs/cdirs/m4559/xgao/software_gradientflow/venv-quda-develop`.
- The QUDA install prefix is
  `/global/cfs/cdirs/m4559/xgao/software_gradientflow/quda-develop/install`.
- PyQUDA is installed editable from
  `/global/cfs/cdirs/m4559/xgao/software_gradientflow/PyQUDA-develop` (core at
  `PyQUDA-develop/pyquda_core`).
- Use `systems/perlmutter/activate-venv-quda.sh` for the module stack, but verify
  that `python` and `QUDA_PATH` resolve to the locations above; do not accept
  stale helper defaults silently.
- The activation helper supplies the required Cray MPI/HDF5 settings.  Keep
  `MPICH_GPU_SUPPORT_ENABLED=1` and verify `h5py.get_config().mpi` when parallel
  HDF5 is required.
- `login32` is useful for small GPU smoke tests; multi-rank production tests
  still require a valid Slurm allocation.

### Aurora

- Main checkout:
  `/lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement`.
- Activation helper:
  `/lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh`.
- Use `backend="dpnp", backend_target="sycl"`; do not introduce Torch XPU
  paths unless explicitly requested.
- Launch multi-rank jobs through PALS inside a compute allocation and keep
  parallel h5py enabled.

### Smoke Inputs

- Small flow gauge: `test_gauge/S8T8_wilson_b6.0`.
- Main workflow gauge: `test_gauge/S8T32_wilson_b6.cg.1e-08.0`.
- Baseline flow check: `systems/perlmutter/check-gradient-flow.sh`.

## Repository Hygiene And Verification

- Preserve unrelated user changes in a dirty worktree; never clean or rewrite
  them implicitly.
- Use `apply_patch` for manual edits and keep all code comments in English.
- Never commit generated HDF5 data, logs, profiles, tuning/cache directories,
  `__pycache__`, or LaTeX intermediates.
- Build LaTeX in `/tmp` with `texlive/2024`; copy back only tracked PDFs.
- On CuPy, avoid implicit CuPy-to-NumPy conversion.  Explicitly use `.get()`
  before constructing NumPy/backend gamma arrays.
- If `py_compile` cannot write `__pycache__` on a restrictive filesystem, use
  `ast.parse` for a write-free syntax check.
- Some successful QUDA jobs print shutdown leak warnings.  Check the exit code
  and physics output before treating those warnings as a failed measurement.
- Lightweight regression entry point:

```bash
source systems/perlmutter/activate-venv-quda.sh
python tests/run_smoke_tests.py
```

## EMT Conventions And Pitfalls

- Active implementations are
  `pyquda_measurement_utils/pion_EMT_vibe_develop.py` and
  `pyquda_measurement_utils/proton_EMT_vibe_develop.py`.  The removed legacy
  `EMT_meson.py` should not be restored into active workflows.
- EMT vibe outputs are HDF5-only.  Hadron correlator tags include source
  position; source-independent EMT quark-loop tags do not.
- Quark EMT 3pt files contain only 3pt observables.  Read denominators and full
  momentum coverage from the separate EMT 2pt files.
- Proton `PpUnpol` two-point projection is `(C2[I] + C2[T])/4`, corresponding
  to `(1+gamma_4)/4`; the source interpolator label `5` is not a sink projector.
- Flow schedule is always measure first, then flow: step 0 is unflowed, the
  first interval uses ten `epsilon/10` substeps, and later intervals use one
  `epsilon` step.
- Meson/proton connected EMT uses fixed-sink convention B with
  `dst2 = gamma5 * seq_raw^dagger * gamma5` and `meson_sign = 1`.
- Proton left-acting derivatives must act on the raw sequential propagator
  before the final gamma5-hermiticity/index transform.  Applying `covDev`
  directly to finalized `seq_data` breaks gauge covariance.
- Pion sequential sources require active-line sink smearing.  For pion EMFF,
  the 2pt has no momentum transfer and must use source-side boosts on both ends;
  do not mix source and sink boost choices in the denominator.
- Gluon 1pt stores the full gluonic building block.  The traceless operation in
  `_F_clover_traceless` projects each field-strength matrix to `su(3)`; it does
  not make the final EMT tensor traceless.

## EMT Disconnected One-Point Loops

- For the current `l64c64a076` ensemble, `am_q=-0.015` is the strange test mass
  and `am_q=-0.049` is the light test mass. Keep their data, sample logs, MG
  benchmarks, and tuning caches in separate run directories; these values are
  not universal defaults for other ensembles.
- Never construct a distributed stochastic source by resetting the same
  backend RNG seed on every MPI rank.  Equal local shapes repeat the local
  field; use global-coordinate counter noise for decomposition independence.
- Quark loops use decomposition-independent full-volume counter-based `Z4`
  noise.  The counter includes global `x,y,z,t`, spin, color, configuration,
  base-noise index, and an optional stream salt.
- Production configuration identity must be supplied explicitly on the CLI;
  never infer it from an environment variable or silently default it to zero.
- One source-independent EMTc file is written per configuration and contains
  every absolute insertion time.  Reuse it for all hadron source times and
  align time with `tau_abs = (t0 + tau_rel) % Nt` and rephase spatial momentum
  by `exp[-2 pi i q.(x0-origin)/L]`.  Reject loop files without explicit global
  lattice/Fourier-origin provenance.
- Disconnected shards and canonical files persist only base-noise and HP
  indices; reconstruct the effective source index as `base * N_HP + hp`.
- Gluon EMT loops are also source independent and use
  `EMTg/<lat>.EMTg.<cfg>.<ama>.<sm>.h5`.  Quark and gluon wrappers must share
  the same flow grid; the production epsilon default is `0.207936`.
- Fermion flow is four dimensional.  With `xi_f=K(t_f)xi` and
  `eta_f=K(t_f)D^{-1}xi`, a fixed-time loop estimates
  `Tr[P_tau Gamma K D^{-1} K^dag]`: the observable is spatially summed at fixed
  `tau`, but the initial stochastic source must cover every time slice.
- Flow contractions load the flowed gauge into QUDA's global resident state.
  Restore the original gauge before the next stochastic inversion; loading it
  only once before the source loop causes later inversions to use the wrong
  resident gauge.
- A full initial `loadGauge(U)` already leaves the original gauge/MG state
  resident.  The first inversion must use it directly; only inversions after a
  flowed-gauge context need `thin_update_only=True` restoration.
- A single initial-time projector is incomplete at nonzero flow time.  A
  complete time-dilution basis is unbiased only after all projectors are
  summed, with the corresponding inversion cost.
- Spatial HP and 4D HP both multiply a full-volume base source.  Spatial HP has
  time-independent probing signs; it is not a 3D or time-diluted source.
- Flowed EMT defaults to isotropic 4D
  `interleaved_xyzt_binary_projected_to_evenodd` HP.  The S8T8 counter-Z4 test
  found 4D HP16 effective on every time slice, while spatial HP left temporal
  neighbors uncancelled and increased variance.
- Only MPI rank 0 should open the serial quark/gluon HDF5 output file.
- The physical disconnected correlator requires ensemble subtraction:
  `<C2 L> - <C2><L>`.  A single-configuration product is only an unsubtracted
  diagnostic proxy.
- Quark EMT now stores the complete 16 local plus `16x4` unsymmetrized
  derivative basis; derive the old EMT from vector channels.  Raw PyQUDA `Y5`
  and `T5` need minus signs for uniform `gamma_mu gamma5`, and raw tensor masks
  are `[gamma_mu,gamma_nu]/2` without `i`; use the stored basis transform.
- EMT operator schema v3 stores the identity only in the 16-Gamma local basis,
  names the separate source norm `flowed_noise_norm`, and embeds the kinetic
  data under `derived/ringed` in EMTc.  Compute physical ringed factors only
  after ensemble-averaging the kinetic expectation value.
- Production quark EMT 1pt is base-oriented: base-internal part files are only
  checkpoints, a base is complete only after all HP vectors validate, and an
  explicit streaming finalizer publishes one canonical EMTc file.

## qTMD Conventions And Pitfalls

- Fixed-length GI staple index is
  `[b_T, b_z, eta, transverse_direction]`, with even `b_z`,
  `eta >= abs(b_z)/2`, `b_T >= 0`, and total length `2*eta + b_T`.
- Prefer cached gauge-only staple transporters and apply them to the shifted
  endpoint field.  Keep the direct covariant-shift path for validation.
- The disconnected estimator is `xi^dagger P Gamma O_b eta`, with
  `eta=D^{-1}xi`; apply `O_b` to `eta`, never to `xi`.  The reversed
  `eta^dagger P Gamma O_b xi` estimator targets the wrong trace, so its data
  must not be reused.
- When constructing disconnected `CG_qTMD`, reset the shifted solution before
  changing transverse direction; otherwise the `b_Y` path incorrectly starts
  from the final `b_X` displacement.
- Local-limit invariant:
  `GI_PDF(bz=0) = CG_PDF(bz=0) = CG_qTMD(bT=0,bz=0)`.
- Disconnected qTMD production uses source-independent canonical tags,
  counter-based full-volume `Z4`, base/HP-interval shards, and explicit
  destination-side finalize.

## Flowed-Quark Ringed Normalization

- Standalone ringed, EMT, and qTMD resume only from a fingerprinted base-level
  text sample log; production must not require local shards to remain after a
  base is logged. Finalizers validate parts once while merging at the destination.
- Multigrid blocks, solver tolerance and maxiter are runtime controls, not
  disconnected measurement identity; resumed base ranges may mix them.
- Standalone ringed uses `RingedQuark1pt`, a kinetic-only subclass of the EMT
  shared runner. It supports full-volume plain/HP counter noise but no
  spin-color dilution or stored ringed factors. Average `K` over configurations
  before any nonlinear normalization; never average per-configuration `1/K`.
