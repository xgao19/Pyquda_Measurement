# PyQUDA / QUDA Session Memory

Last updated: 2026-05-18

This file captures the important state from the current collaboration so the
next session can resume quickly.

## Current goal

The user will mainly use PyQUDA to write programs and run tasks on Perlmutter.
The validated environment should be treated as the default baseline unless the
user asks to change it.

## Validated QUDA build

- QUDA source: `/global/cfs/cdirs/m3760/xgao/software/quda`
- QUDA install prefix: `/global/cfs/cdirs/m3760/xgao/software/quda/install`
- Build type: `RELEASE`
- GPU arch: `sm_80`
- MPI: `ON`
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

## Validated Python environment

- Shared venv: `/global/cfs/cdirs/m3760/xgao/software/venv`
- Python version in venv: `3.13.11`
- Installed and verified packages:
  - `Cython==3.2.4`
  - `cupy-cuda12x==14.0.1`
  - `h5py==3.16.0`
  - `mpi4py==4.1.1`
  - `numpy==2.4.4`
  - `opt-einsum==3.4.0`
  - `packaging==26.2`

## PyQUDA checkout

- PyQUDA source: `/global/cfs/cdirs/m3760/xgao/software/PyQUDA`
- Branch: `develop`
- Editable install was completed successfully for:
  - `PyQUDA`
  - `PyQUDA-Utils`
- Submodules were updated recursively.

## Runtime / MPI / HDF5 notes

- The working Perlmutter runtime uses:
  - `cray-mpich-abi/9.0.1`
  - `cray-hdf5-parallel/1.14.3.7`
- The helper script:
  - activates the venv
  - exports `QUDA_PATH`
  - exports `PYQUDA_ROOT`
  - sets `HDF5_DIR`
  - preloads the Cray HDF5 and MPI GPU transport libraries
  - keeps `MPICH_GPU_SUPPORT_ENABLED=1`
- This setup was used to avoid the HDF5 runtime mismatch warning and to keep
  parallel `h5py` working.

## Perlmutter helper files

All of these live under:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter`

Files:

- `configure-quda`
- `activate-venv-quda.sh`
- `submit_batch_template.sh`
- `requirements.txt`
- `check-gradient-flow.sh`
- `check-gradient-flow.py`
- `README.md`

## Smoke test status

The minimal smoke test has been validated on `login32` and passes.

Test gauge used:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T8_wilson_b6.0`

The smoke test checks:

- `nvidia-smi`
- `readNERSCGauge(...)`
- `gradientGaugeFlow("wilson", ...)`
- `gradientGaugeFlow("symanzik", ...)`
- fermion `gradientFlow(...)`

The Python entry point is:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.py`

The shell wrapper is:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh`

## How to activate the environment

Recommended:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
```

Then:

```bash
export QUDA_PATH=/global/cfs/cdirs/m3760/xgao/software/quda/install
```

After that, `python -c "import pyquda, cupy, h5py, mpi4py"` should work.

## How to run the smoke test

Wrapper:

```bash
bash /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.sh
```

Direct Python entry if the environment is already active:

```bash
python /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/check-gradient-flow.py
```

## Important example outputs

On `login32`, the smoke test produced:

- Wilson plaquette before flow:
  - `[0.5919862407536087, 0.5911411986845838, 0.5928312828226336]`
- Wilson plaquette after one step:
  - `[0.6228197128397271, 0.6220381245881715, 0.6236013010912828]`
- Symanzik plaquette after one step:
  - `[0.6374540708371997, 0.6367304972845912, 0.6381776443898083]`
- Fermion flow:
  - `fermion_norm2=1.7070053377544578`
  - `fermion_sample=(0.9234826602037387+0.9234826601680953j)`

These are good reference values for future sanity checks.

## EMT Meson work

Primary memory file from now on:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/SESSION_MEMORY.md`

There is a separate runtime working directory:

- `/global/cfs/cdirs/m3760/xgao/software/EMT_meson`

The Perlmutter application copy now lives in:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/EMT_meson/perlmutter`

The original legacy measurement source was removed after migration.  If needed,
recover it from Git history:

- `pyquda_measurement_utils/EMT_meson.py`

The active pion/meson EMT development copy is:

- `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/pyquda_measurement_utils/pion_EMT_vibe_develop.py`

Rename note:

- `EMT_meson_vibe_develop.py` was renamed to `pion_EMT_vibe_develop.py`.
- Perlmutter EMT meson entry scripts now import `pyquda_measurement_utils.pion_EMT_vibe_develop`.
- `proton_EMT_vibe_develop.py` inherits shared quark/gluon 1pt utilities from `pion_EMT_vibe_develop.py`.

Current EMT code organization:

- `pion_EMT_vibe_develop.py`
  - holds the active `QuarkEMT` and `GluonEMT` implementations
  - has a file-level English formula/convention docstring starting from correlation-function definitions
  - documents meson 2pt, pre-sequential 3pt, fixed-sink sequential source, sink-summed 3pt, quark EMT insertion, scalar diagnostic insertion, quark 1pt, gluon 1pt, ringed fermion normalization, and gradient flow
  - uses `flow_epsilon` consistently
  - uses direct `__init__(self, parameters)` parameter handling
  - no longer uses `EMTParameters`, `emt_default_config`, `emt_momentum_grid`, `EMTIOConfig`, or `emt_io_config`
  - uses `self.qlist = parameters["qext"]` and `self.pilist = parameters["p_2pt"]`
  - uses proton/pion-style gamma names: `my_gammas`, `my_pyquda_gammas`, `pyquda_gammas_order`
  - no longer creates top-level CuPy gamma matrices; gamma matrices are moved to the active backend/queue through helper functions
- `bw_seq_pyquda.py`
  - contains `create_meson_bw_seq_pyquda(...)`
  - proton sequential-source helpers were not renamed
- `io_corr.py`
  - holds EMT file-name helpers and HDF5 writers only
  - includes EMT-specific helpers such as `get_emt_meson_2pt_file_tag(...)` and `save_emt_meson_2pt_hdf5(...)`
- `Pyquda_EMT_gluon_1pt.py`
- `Pyquda_EMT_quark_1pt.py`
- `Pyquda_EMT_quark_3pt.py`
  - are thin entry scripts that define run parameters directly, initialize QUDA, load the gauge, and call the measurement
  - old commented-out prototype calls and unused imports were removed
  - current copies are also under `application/EMT_meson/perlmutter`
  - now import `QuarkEMT` / `GluonEMT` from `pyquda_measurement_utils.pion_EMT_vibe_develop`

Current EMT physics / IO conventions:

- EMT output is HDF5-only; `.npy` output was removed from the vibe path.
- HDF5 tag naming includes `lat / cfg / ama / src / sm`.
- The active test gauge is:
  - `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0`
- The EMT scripts currently use `lat_tag = "l64c64a076"` and `sm_tag = "1HYP_GSRC_W10_k0"` as file tags, even though the smoke-test gauge geometry is S8T32.
- `GEN_SIMD_WIDTH` was removed from the EMT entry scripts.
- `flow_epsion` was standardized to `flow_epsilon`.
- `pos_boost` and `neg_boost` are used:
  - same boost uses one source inversion plus a copy
  - different boosts use separate forward/backward source smearings and inversions
- Quark and gluon gradient-flow schedules are now aligned:
  - measure first, then flow
  - `step = 0` is unflowed
  - the first interval is subdivided into 10 small steps immediately after measuring `step == 0`
  - output index `step` is intended to correspond to `step * flow_epsilon`
- The file-level comments explicitly state that gradient flow smooths UV fluctuations at flow radius approximately `sqrt(8 t)` and that the renormalized EMT combination is assembled in later analysis.

Current meson 2pt and quark 3pt conventions:

- Meson 2pt has a dedicated EMT HDF5 output path:
  - directory: `EMT2pt`
  - dataset: `C2`
  - shape: `(sink_gamma, momentum, time)`
  - current baseline shape: `(16, 125, 32)`
- Meson 2pt uses forward and backward/antiquark propagators:
  - `Tr[Gamma_sink S_bw Gamma_src S_fw]`
  - source gamma is fixed by `src_interpolator`
  - sink gamma scans all 16 gamma structures:
    - `5, T, T5, X, X5, Y, Y5, Z, Z5, I, SXT, SXY, SXZ, SYT, SYZ, SZT`
- Quark 3pt uses only the standard meson sequential-source convention, called convention B during development:
  - `prop_fw`: source-to-insertion forward line
  - `seq_bw_prop`: sink-to-insertion sequential backward line
  - `dst2 = gamma5 * seq_bw_prop^dagger * gamma5`
  - sequential source uses `gamma5 * Gamma_sink^dagger * gamma5`
  - `meson_sign = 1`
- The old pre-B 3pt contraction branch was removed after testing showed the B result is the complex conjugate of the old branch for the `gamma5`, `pf=0` sanity check.
- Quark 3pt includes momentum-transfer projection:
  - `C3_chi` shape: `(N_tsep, N_flow, N_qext, Nt)`
  - `C3_Tmunu` shape: `(N_tsep, N_flow, N_qext, 4, 4, Nt)`
  - q=0 is currently at `q0_index = 62` for the `[-2,2]^3` q-grid.
- The 3pt HDF5 attrs intentionally record:
  - `contraction_convention = "B"`
  - `meson_sign = 1`
  - `src_interpolator`
  - `sink_interpolator`
  - `n_qext`

Current quark/gluon 1pt and ringed fermion conventions:

- Quark 1pt measures stochastic trace estimators with Z_n noise:
  - `eta = D^{-1} xi`
  - `CHI[0] ~ xi^dagger eta`
  - `CHI[1] ~ xi^dagger xi`
  - `T_{nu,mu}^q ~ -1/2 xi^dagger gamma_nu [D_{+mu} - D_{-mu}] eta`
- The saved quark 1pt `Tmunu` diagonal trace contains the flowed-fermion kinetic bilinear used for ringed fermion normalization:
  - `sum_mu T_{mu mu}^q(0,t) = -1/2 <bar_chi overleftrightarrow{not D} chi>`
  - reconstruct it at q=0 from `avg/Tmunu/T11`, `T22`, `T33`, and `T44`
  - `CHI` is saved as scalar trace/noise diagnostic and is not the standard ringed-fermion normalization by itself
- Gluon 1pt measures the flowed gluonic EMT building block:
  - `T_{mu nu}^g(q,t) = 2/V3 sum_x Phi_q(x) sum_{rho != mu,nu} Tr[F_{mu rho} F_{nu rho}]`
  - the final EMT tensor is not traceless-projected in the contraction code
  - `_F_clover_traceless` only projects each clover field-strength matrix onto the su(3) gauge algebra
- Analysis should combine connected 3pt, quark 1pt, and gluon 1pt data with the appropriate gradient-flow coefficients, mixing/trace terms, and vacuum subtractions outside the contraction kernel.

Current EMT baseline data generated on 2026-05-04:

- The current code was run on `login32` and saved into:
  - `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data`
- The latest quark/2pt/3pt baseline was regenerated after fixing the quark flow schedule to match gluon.
- Command used for latest quark baseline:

```bash
source /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/systems/perlmutter/activate-venv-quda.sh
cd /global/cfs/cdirs/m3760/xgao/software/EMT_meson
python Pyquda_EMT_quark_1pt.py --config_num 0
python Pyquda_EMT_quark_3pt.py --config_num 0 --src_interpolator 5 --sink_interpolator 5
```

- Baseline files and timestamps:
  - `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data/EMTg/l64c64a076.EMTg.0.0.x0y0z0t0.1HYP_GSRC_W10_k0.h5`
    - timestamp: `2026-05-04 08:55:03`
    - not rerun for the quark flow schedule fix, because gluon already used the correct schedule
  - `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data/EMTc/l64c64a076.EMTc.0.0.x0y0z0t0.1HYP_GSRC_W10_k0.h5`
    - timestamp: `2026-05-04 17:06:41`
  - `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data/EMT2pt/l64c64a076.EMT2pt.0.0.x0y0z0t0.1HYP_GSRC_W10_k0.h5`
    - timestamp: `2026-05-04 17:10:37`
  - `/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data/EMT3pt/l64c64a076.EMT3pt.0.0.x0y0z0t0.1HYP_GSRC_W10_k0.spin5.h5`
    - timestamp: `2026-05-04 17:11:15`
- Verified current baseline shapes:
  - `EMTg/Tmunu/T..`: `(125, 11, 32)`
  - `EMTc/raw/Tmunu_pervec`: `(1, 4, 4, 125, 11, 32)`
  - `EMTc/raw/CHI_pervec`: `(1, 2, 125, 11, 32)`
  - `EMTc/avg/CHI`: `(2, 125, 11, 32)`
  - `EMT2pt/C2`: `(16, 125, 32)`
  - `EMT3pt/C2`: `(32,)`
  - `EMT3pt/C3_chi`: `(2, 11, 125, 32)`
  - `EMT3pt/C3_Tmunu`: `(2, 11, 125, 4, 4, 32)`
- The `EMT3pt` baseline attrs include:
  - `contraction_convention = "B"`
  - `meson_sign = 1`
  - `src_interpolator = "5"`
  - `sink_interpolator = "5"`
  - `n_qext = 125`

Validation history for EMT:

- The three EMT measurements were repeatedly run on `login32` using the bundled S8T32 test gauge.
- Earlier refactors were compared against the original EMT outputs and were numerically consistent.
- After the B-only cleanup, the B-only output was compared against the previous B sanity output:
  - `C2 max_rel = 1.5e-17`
  - `C3_chi max_rel = 8.4e-16`
  - `C3_Tmunu max_rel = 4.8e-16`
- For the `gamma5`, `pf=0` sanity check, the old pre-B branch and B branch were found to be complex conjugates:
  - `C3_chi: B - conj(old) max_rel ~ 3.6e-15`
  - `C3_Tmunu: B - conj(old) max_rel ~ 3.6e-15`
- After adding the explicit qext dimension to 3pt output, q=0 was compared to the old no-q baseline at the true q=0 index and matched to roundoff.
- After gamma/backend cleanup, smoke tests passed and q=0 comparisons remained roundoff-level.

## Repo notes

- The Perlmutter README was expanded with setup, requirements, smoke test, and batch usage instructions.
- `requirements.txt` in the Perlmutter directory matches the validated venv.
- EMT Perlmutter application scripts are copied under:
  - `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/EMT_meson/perlmutter`
- The repository may have local uncommitted changes in the working tree; do not revert anything unless the user explicitly asks.

## If you resume later

Read `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/SESSION_MEMORY.md` first, then continue from the validated baseline above. The next likely tasks are:

1. Add or adapt PyQUDA-based task scripts.
2. Add batch job wrappers that reuse the same environment.
3. Extend smoke tests or add new physics checks using the bundled test gauge.

## 2026-05-18 update: pion sequential-source smearing and docs

Context:

- A pion EMFF production ratio check on the `l48c64a060` ensemble suggested
  the raw `C3/C2` ratio was too small by a large factor.
- Tests with artificial volume factors such as `L^3/24` or `L^3/72` were
  informative but not considered the correct physics explanation because the
  normalization should not depend on this kind of ad hoc lattice-size factor.
- Comparing against the known-good proton sequential-source construction showed
  the key missing ingredient: the meson sequential source also needs sink-side
  smearing of the active line before the sequential inversion.

Implemented code changes:

- `pyquda_measurement_utils/bw_seq_pyquda.py`
  - `create_meson_bw_seq_pyquda(...)` now accepts optional `sm_width=None` and
    `sm_boost=None`.
  - If both are provided, the code applies:
    `src_seq = boosted_smearing(src_seq, w=sm_width, boost=sm_boost)`
    before `core.invertPropagator(...)`.
  - Default `None` values preserve the old behavior for any caller that does
    not request active-line sequential-source smearing.
- `application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py`
  - EMFF now passes `parameters["width"]` and
    `parameters["pos_boost_sink"]` to `create_meson_bw_seq_pyquda(...)`.
  - Rationale: the input line `prop_neg_sink` already contains the spectator
    sink smearing with `neg_boost_sink`; the outer sequential-source smearing
    supplies the active forward-line sink smearing with `pos_boost_sink`.
- `application/pion_TMD_CG/perlmutter/Pyquda_pion_TMD_CG.py`
  - pion qTMD now passes `parameters["width"]` and
    `parameters["pos_boost"]` to `create_meson_bw_seq_pyquda(...)`.
  - Rationale: the input line has already been sink smeared with `neg_boost`;
    the sequential-source smearing supplies the active line with `pos_boost`.
- `pyquda_measurement_utils/pion_EMT_vibe_develop.py`
  - pion EMT connected 3pt now passes
    `self.width if self.CG_GaussSmear else None` and
    `self.neg_boost if self.CG_GaussSmear else None` to
    `create_meson_bw_seq_pyquda(...)`.
  - Rationale: in the current EMT implementation `prop_fw_SS` has already been
    sink smeared with `pos_boost`; the sequential propagator represents the
    opposite active line and receives `neg_boost`.

Validation for the sequential-source smearing change:

- `python3 -m py_compile` passed for:
  - `pyquda_measurement_utils/bw_seq_pyquda.py`
  - `pyquda_measurement_utils/pion_EMT_vibe_develop.py`
  - `application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py`
  - `application/pion_TMD_CG/perlmutter/Pyquda_pion_TMD_CG.py`
- EMFF S8T32 smoke test on `login32` with test gauge:
  - `/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0`
  - `qmax=0`, one source, `src_interpolator=5`, `sink_interpolator=5`,
    `pf=0.0.0`, `t_insert=2.4`
  - output was written to `/tmp/pion_emff_main_seqsmear` on `login32`
  - ratio check after the fix:
    - `dt=2`: `C3/C2` at `tau=dt` approximately
      `0.6536689621 + 0.0001869748i`
    - `dt=2` middle-window average approximately
      `0.6918008316 + 0.0003120646i`
    - `dt=4`: `C3/C2` at `tau=dt` approximately
      `0.6910497146 - 0.0002191994i`
    - `dt=4` middle-window average approximately
      `0.7847968233 + 0.0001809609i`
  - Before this active-line sequential-source smearing diagnosis, the same
    style of S8T32 test had ratios of order `0.05`; the improvement strongly
    supports the smearing fix as the real source of the production ratio issue.
- pion qTMD S8T32 smoke test on `login32` passed:
  - command used small settings: `qmax=0`, `b_z=0`, `b_T=0`, one source,
    `t_insert=2`, `src_interpolator=fixed_g5`, `sink_interpolator=5`
  - output was written to `/tmp/pion_tmd_main_seqsmear`
  - completed forward inversion, pion 2pt, meson sequential inversion,
    qTMD CG, GI_PDF, CG_PDF contractions, and HDF5 writing.
- pion EMT minimal runtime smoke test on `login32` passed:
  - temporary script wrote to `/tmp/pion_emt_main_seqsmear`
  - used `qext=[[0,0,0,0]]`, `p_2pt=[[0,0,0,0]]`, `pf=[0,0,0,0]`,
    `t_separations=[2]`, `flow_steps=0`, `src_interpolator=5`,
    `sink_interpolator=5`
  - completed source smearing, 2pt, sequential-source inversion, and step-0
    connected EMT contraction.

Local-limit consistency check:

- On `login32`, compared the S8T32 smoke outputs:
  - EMFF dataset: `SS/T/PX0PY0PZ0`
  - qTMD local CG datasets:
    - `SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0`
    - `SS/T/PX0PY0PZ0/b_Y/eta0/bT0/bz0`
  - GI_PDF dataset: `SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0`
  - CG_PDF dataset: `SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0`
- Conditions:
  - `qext=0`
  - `pf=0`
  - `dt=2`
  - source and sink are `gamma5`
  - local current is `T` / `gamma_4`
  - `bT=0`, `bz=0`, `eta=0`
- Results:
  - max absolute difference between EMFF and each qTMD/PDF local output:
    `8.401198274707147e-14`
  - max relative difference:
    `7.016753011768688e-16`
  - qTMD `b_X - b_Y` max absolute difference: `0.0`
- Conclusion:
  - At zero displacement and zero momentum transfer,
    `EMFF = local qTMD CG = GI_PDF = CG_PDF` at machine precision for the
    tested current/gamma setup.

Production directory update:

- The production run directory:
  - `/global/cfs/cdirs/m5208/xgao/runs/l48c64a060/pion_EMFF`
- The active production scripts there are:
  - `Pyquda_pion_EMFF_pf0.py`
  - `Pyquda_pion_EMFF_pf3.py`
- Both scripts were patched to pass:
  - `parameters["width"]`
  - `parameters["pos_boost_sink"]`
  to `create_meson_bw_seq_pyquda(...)`.
- Only these production Python scripts were edited; data, logs, and submit
  scripts were not modified.
- The production `run.sh` and submit scripts set:
  - `PYQUDA_MEASUREMENT_ROOT=/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement`
  - `PYTHONPATH` includes this root
  so they will use the updated repo-side `bw_seq_pyquda.py`.
- `ast.parse` passed for both production scripts.
- `py_compile` was not used in that production directory because Python tried
  to write `__pycache__` and the filesystem rejected the write.

Documentation update:

- Updated the pion-related LaTeX docs:
  - `docs/pion_EMFF/pion_EMFF.tex`
  - `docs/pion_qTMD/pion_qTMD.tex`
  - `docs/pion_EMT/pion_EMT.tex`
  - `docs/pion_qTMDWF/pion_qTMDWF.tex`
- Added the smearing kernel matching `boosted_smearing_pyquda.py`:

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

- Documented that:
  - `boost=[kx,ky,kz]` is an integer momentum-smearing vector in units of
    `2*pi/L_i`
  - smearing is implemented as a three-dimensional spatial FFT convolution
  - the same kernel is broadcast over all Euclidean time slices
  - current implementation assumes identity gauge / no explicit gauge rotation
- Updated EMFF/qTMD/EMT sequential-source formulas to include active-line
  sequential-source smearing before inversion.
- Added qTMD local-limit consistency statement:
  - at `bT=bz=eta=0` and `q=0`, local qTMD CG, CG_PDF, GI_PDF, and EMFF
    should agree for the same local current.
- LaTeX equation begin/end counts were checked and balanced for all four edited
  tex files.
- No PDF files were regenerated because `pdflatex`, `latexmk`, and `tectonic`
  were not available in the current environment.

Current uncommitted working tree notes:

- As of this update, the repo has uncommitted changes in:
  - `application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py`
  - `application/pion_TMD_CG/perlmutter/Pyquda_pion_TMD_CG.py`
  - `docs/pion_EMFF/pion_EMFF.tex`
  - `docs/pion_EMT/pion_EMT.tex`
  - `docs/pion_qTMD/pion_qTMD.tex`
  - `docs/pion_qTMDWF/pion_qTMDWF.tex`
  - `pyquda_measurement_utils/bw_seq_pyquda.py`
  - `pyquda_measurement_utils/pion_EMT_vibe_develop.py`
- Do not revert these unless the user explicitly asks.

Disconnected qTMD 1pt local/PDF sanity check:

- Added a small regression helper:
  - `tests/test_qtmd_disconnected_local_pdf_limit.py`
- The test compares the S8T32 login-node smoke outputs:
  - `GI_PDF` at `bz=0`, `q=0`
  - `CG_PDF` at `bz=0`, `q=0`
  - `CG_qTMD` at `bT=0`, `bz=0`, `q=0`
- Expected local-limit identity:
  - `O_GI_PDF(bz=0) = O_CG_PDF(bz=0) = O_CG_qTMD(bT=0,bz=0) = 1`
  - therefore the corresponding disconnected loops should agree for the same
    gauge field, stochastic source, gamma matrix, and momentum.
- Verified exact equality on the current S8T32 smoke files:
  - `GI_PDF vs CG_PDF raw maxdiff = 0.0`
  - `GI_PDF vs CG_qTMD b_X raw maxdiff = 0.0`
  - `GI_PDF vs CG_qTMD b_Y raw maxdiff = 0.0`
  - the averaged `gamma5` HDF5 paths also agree exactly.
- Documented this in:
  - `docs/qTMD_disconnected_1pt/qTMD_disconnected_1pt.tex`
- The test helper skips automatically when the ignored smoke-test HDF5 files
  are absent, so it is safe for normal source-only environments.

Disconnected qTMD 1pt nonzero-bz sanity check:

- Added a second small regression helper:
  - `tests/test_qtmd_disconnected_nonzero_bz.py`
- The test compares:
  - `CG_PDF` at `bz=+/-1`, `q=0`
  - `CG_qTMD` at `bT=0`, `bz=+/-1`, `q=0`, for both `b_X` and `b_Y`
  - `GI_PDF` at `bz=+/-1`, `q=0`
- Expected coordinate-gauge identity:
  - `O_CG_PDF(bz=+/-1) = O_CG_qTMD(bT=0,bz=+/-1,b_X)`
  - `O_CG_PDF(bz=+/-1) = O_CG_qTMD(bT=0,bz=+/-1,b_Y)`
- The first HDF5 comparison exposed a real `CG_qTMD` bug:
  - the `b_Y` branch was continuing from the final shifted source of the `b_X`
    branch instead of restarting from the local stochastic source.
- Fixed this in:
  - `pyquda_measurement_utils/Disconnected_1pt_qTMD_vibe_develop.py`
  by resetting `shifted_xi` when the `CG_qTMD` transverse direction changes.
- After rerunning the S8T32 `CG_qTMD_bz1bt1` smoke output:
  - `bz=+1 CG_PDF vs CG_qTMD b_X raw maxdiff = 0.0`
  - `bz=+1 CG_PDF vs CG_qTMD b_Y raw maxdiff = 0.0`
  - `bz=-1 CG_PDF vs CG_qTMD b_X raw maxdiff = 0.0`
  - `bz=-1 CG_PDF vs CG_qTMD b_Y raw maxdiff = 0.0`
- `GI_PDF` and `CG_PDF` use matching Wilson-index labels but differ at
  nonzero `bz` on the nontrivial test gauge, as expected because the GI Wilson
  line is active.
