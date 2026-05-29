---
name: aurora-pyquda-qcd
description: Use when working on Aurora-specific PyQUDA and lattice QCD measurement tasks in this repository, including installing or validating PyQUDA develop, configuring Intel GPU/SYCL dpnp backends, checking mpi4py and parallel h5py, launching PALS/PBS multi-GPU runs, debugging QUDA tuning/runtime issues, or adapting smoke/production-style measurement workflows such as EMT_proton.
---

# Aurora PyQUDA QCD

Use this skill as the operational runbook for Aurora PyQUDA work in
`software_gradientflow/Pyquda_Measurement`.

## Workflow

1. Inspect the current repo state before changing anything:
   - Check `systems/Aurora/README.md` for current validated environment notes.
   - Check the relevant `application/*/Aurora` run wrapper before inventing a new launch command.
   - Check `git status --short` and preserve unrelated user changes.
2. Prefer the validated Aurora route:
   - PyQUDA from the `develop` branch.
   - Existing QUDA install reused from `software_260507`.
   - `backend="dpnp", backend_target="sycl"`.
   - Parallel `h5py` and Aurora MPICH-linked `mpi4py`.
3. For runtime tests, start small:
   - First run import and linkage checks.
   - Then run the S8T32 test-gauge smoke.
   - Only then run larger l64 fixed_GLU or production-style measurements.
4. Run multi-rank PyQUDA tests through an active PBS allocation or SSH to an allocated compute node and launch with PALS. Avoid treating a login shell as a safe MPI/GPU runtime.

## Hard Rules

- Do not install Torch XPU unless the user explicitly asks for it; it can pollute MPI and oneAPI runtime resolution.
- Do not accept a non-parallel `h5py`; `h5py.get_config().mpi` must be `True`.
- Do not use tile-affinity wrappers or `ZE_AFFINITY_MASK` for the current dpnp path unless revalidated; they can make dpnp see zero GPUs.
- Do not reuse stale QUDA tune caches after changing rank geometry, lattice size, QUDA, PyQUDA, or backend settings.
- Do not force `ppn` for the validated 32-rank-on-6-node l64 smoke; uneven rank distribution is intentional there.

## Reference

Read `references/aurora-pyquda-best-practices.md` when you need exact module
commands, installation steps, verification commands, PALS launch templates,
S8T32 smoke settings, l64 fixed_GLU smoke settings, or known Aurora failure
modes.
