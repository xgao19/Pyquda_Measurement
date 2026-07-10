# PyQUDA Measurement Session Memory

Last updated: 2026-07-10

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

- Use `systems/perlmutter/activate-venv-quda.sh` rather than reconstructing the
  module stack manually.
- The production QUDA install is normally
  `/global/cfs/cdirs/m3760/xgao/software/quda/install`.
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

- Quark loops use decomposition-independent full-volume counter-based `Z4`
  noise.  The counter includes global `x,y,z,t`, spin, color, configuration,
  base-noise index, and an optional stream salt.
- One source-independent EMTc file is written per configuration and contains
  every absolute insertion time.  Reuse it for all hadron source times and
  align with `tau_abs = (t0 + tau_rel) % Nt` in analysis.
- Fermion flow is four dimensional.  With `xi_f=K(t_f)xi` and
  `eta_f=K(t_f)D^{-1}xi`, a fixed-time loop estimates
  `Tr[P_tau Gamma K D^{-1} K^dag]`: the observable is spatially summed at fixed
  `tau`, but the initial stochastic source must cover every time slice.
- Flow contractions load the flowed gauge into QUDA's global resident state.
  Restore the original gauge before the next stochastic inversion; loading it
  only once before the source loop causes later inversions to use the wrong
  resident gauge.
- A single initial-time projector is incomplete at nonzero flow time.  A
  complete time-dilution basis is unbiased only after all projectors are
  summed, with the corresponding inversion cost.
- Spatial HP and 4D HP both multiply a full-volume base source.  Spatial HP has
  time-independent probing signs; it is not a 3D or time-diluted source.
- Only MPI rank 0 should open the serial quark/gluon HDF5 output file.
- The physical disconnected correlator requires ensemble subtraction:
  `<C2 L> - <C2><L>`.  A single-configuration product is only an unsubtracted
  diagnostic proxy.
- Quark EMT 1pt writes a kinetic-only `FlowedQuarkRinged` companion from the
  same raw zero-momentum diagonal tensor; this adds no solves or derivatives.
  `CHI` remains only a scalar/noise diagnostic.  Compute the physical ringed
  factor after ensemble-averaging the kinetic expectation value.
- Production quark EMT 1pt is base-oriented: base-internal part files are only
  checkpoints, a base is complete only after all HP vectors validate, and an
  explicit streaming finalizer publishes canonical EMTc/ringed files.

## qTMD Conventions And Pitfalls

- Fixed-length GI staple index is
  `[b_T, b_z, eta, transverse_direction]`, with even `b_z`,
  `eta >= abs(b_z)/2`, `b_T >= 0`, and total length `2*eta + b_T`.
- Prefer cached gauge-only staple transporters and apply them to the shifted
  endpoint field.  Keep the direct covariant-shift path for validation.
- When constructing disconnected `CG_qTMD`, reset the shifted source before
  changing transverse direction; otherwise the `b_Y` path incorrectly starts
  from the final `b_X` displacement.
- Local-limit invariant:
  `GI_PDF(bz=0) = CG_PDF(bz=0) = CG_qTMD(bT=0,bz=0)`.
- Disconnected qTMD production uses source-independent canonical tags,
  counter-based full-volume `Z4`, base/HP-interval shards, and explicit
  finalize.  A base is complete only after all HP parts validate.

## Flowed-Quark Ringed Normalization

- Persistent production output uses fixed-interval `.block*.h5` files;
  `block_interval_solves` defaults to 64.
- HP256 sample-log resume is intentionally narrow: hierarchical probing,
  `hp_num_vectors=256`, and no spin-color dilution.  A completed sample-log
  entry represents a full base-noise sample, not one interval block.
- Per-base seeds must use `flowed_quark_ringed_norm_sample_seed(...)` so skipped
  or resumed bases do not change later stochastic sources.
- Point spin-color dilution has 12 channels.  The final kinetic observable must
  sum the spin-color trace via `spin_color_trace_factor=12`; averaging those
  channels would give the wrong normalization.
