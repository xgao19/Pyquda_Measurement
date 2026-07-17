# `pyquda_DA_k6.py` README

## 1. Purpose

`pyquda_DA_k6.py` is a PyQUDA-based measurement script for the charmonium DA correlator in the `b_T = 0` limit.

In the language of the existing codebase:

- it is the `DA` special case of the more general `TMDWF` workflow,
- it keeps the full sink gamma basis,
- it supports one or more source gamma choices,
- and it measures both
  - `2pt` correlators,
  - `CG DA` correlators,
  - `GI DA` correlators.

The script is designed for production-style measurements on the `l64c64a040` ensemble using smeared sources and PyQUDA inversions.

---

## 2. Physics Overview

### 2.1 What is being measured

The target observable is a nonlocal bilinear correlator with a spatial separation along the `z` direction.

The script focuses on the `b_T = 0` case, so there is no transverse displacement. In that sense, this is the straight-line DA limit of the broader TMDWF construction.

The relevant operator structure is schematically

\[
\bar q(x)\,\Gamma_g\,W(x,x+z\hat z)\,q(x+z\hat z),
\]

where:

- `\Gamma_g` is the sink Dirac structure,
- `W(x,x+z\hat z)` is a straight Wilson line in the `z` direction,
- `z` is the longitudinal separation,
- and the source Dirac structure is chosen independently.

---

### 2.2 Common correlator form used in the code

The code writes both `CG` and `GI` DA contractions in the common form

\[
C^{(g,\mathrm{src})}(q,t;z)
=
\sum_x e^{i q\cdot x}\,
\mathrm{Tr}_{c,s}
\Big[
\gamma_5\,S_b^\dagger(x)\,\gamma_5\,
\Gamma_g\,
F(x;z)\,
\Gamma_{\mathrm{src}}
\Big].
\]

Here:

- `S_b` is the backward propagator,
- `\Gamma_g` is the sink gamma,
- `\Gamma_src` is the source gamma,
- `F(x;z)` is the forward part with longitudinal displacement,
- `Tr_{c,s}` means trace over color and spin,
- and the phase factor projects to definite momentum.

---

### 2.3 Difference between `CG` and `GI`

This distinction is the most important physics point in the script.

#### `CG` block

In the `CG` block,

\[
F(x;z) = S_f(x+z\hat z),
\]

meaning the forward propagator is shifted in space, but no explicit gauge link is inserted.

In the source code, this is implemented by plain lattice shifts:

- `create_fw_prop_TMD_CG(...)`
- internally using `prop.shift(...)`

So the `CG` correlator is a fixed-gauge object, not a manifestly gauge-invariant nonlocal bilinear.

#### `GI` block

In the `GI` block,

\[
F(x;z) = W(x,x+z\hat z)\,S_f(x+z\hat z),
\]

where `W(x,x+z\hat z)` is the straight Wilson line.

In the source code, this is implemented by gauge-covariant shifts:

- `create_fw_prop_PDF_GI(...)`
- internally using `gauge.pure_gauge.covDev(...)`

So the `GI` correlator is the gauge-invariant nonlocal DA correlator.

---

### 2.4 Two-point correlator

The script also measures a local 2pt correlator:

\[
C^{(g)}_{2pt}(q,t)
=
\sum_x e^{iq\cdot x}
\mathrm{Tr}_{c,s}
\Big[
\gamma_5 S_b^\dagger(x)\gamma_5\,
\Gamma_g\,
S_f(x)\,
\Gamma_{\mathrm{src}}
\Big].
\]

This 2pt part is used mainly as the standard meson correlator for spectroscopy / overlap / mass-related checks and for consistency with the DA workflow.

For qDA it uses the shared `dagger_of_sink` relation

\[
\Gamma_{\rm src}^{(g)}=\gamma_5\Gamma_g^\dagger\gamma_5.
\]

Consequently, the stored Gamma index labels a paired sink/source channel. It
does not mean that all sink channels share one fixed source Gamma.

---

## 3. Script Structure

The script is organized into the following sections.

### 3.1 CLI arguments

The script accepts:

- `--config_num`
- `--mpi_geometry`

These control which gauge configuration is loaded and how MPI is initialized.

---

### 3.2 User-facing run configuration

This section contains the parameters most likely to be changed by a user:

- `data_dir`
- `lat_tag`
- `sm_tag`
- `da_src_gammalist`

These are the first things to inspect before running a new study.

---

### 3.3 Physics / measurement setup

This section defines the measurement parameters passed into

`pion_TMDWF_measurement(parameters)`.

Important entries are:

- `b_T = 0`: selects the DA limit,
- `b_z`: maximum longitudinal separation,
- `pzmin`, `pzmax`: momentum range,
- `width`: smearing width,
- `pos_boost`, `neg_boost`: boosts used in source/sink smearing.

The DA workflow is a two-propagator correlator rather than a fixed-sink
sequential three-point function. It independently constructs and inverts the
positive- and negative-boost source lines, so it does not share the active-line
ambiguity corrected in connected pion qTMD/PDF. CG and straight-link GI DA
transport the designated forward line while retaining the independently
inverted backward line.

---

### 3.4 Helper functions

The local helpers are:

#### `sync_cuda()`

Used to force CUDA synchronization only at timing boundaries.

#### `save_da_correlators(...)`

Saves one DA output file for each chosen source gamma.

Each saved file contains:

- all Wilson-line separations,
- all selected momenta,
- all sink gamma channels,
- all Euclidean times.

#### `Measurement.contract_DA(...)`

This is the core DA contraction routine shared by both:

- `CG`
- `GI`

The operator is always applied to the forward propagator. The only difference
between the two branches is the propagation-update function:

- `Measurement.create_fw_prop_TMD_CG`
- `Measurement.create_fw_prop_PDF_GI`

The latter is the existing straight-link, gauge-covariant PDF transport. A
general staple transporter is neither needed nor used in the DA limit
\(b_T=0\).

---

### 3.5 Lattice / inverter / source preparation

This section:

- sets lattice geometry,
- sets the clover mass and solver parameters,
- loads the gauge field,
- performs HYP smearing,
- builds the clover inverter,
- prepares gamma matrices,
- builds the source positions.

---

### 3.6 Main source loop

For each source position, the script runs:

1. source creation,
2. boosted smearing,
3. forward/backward inversions,
4. 2pt contraction,
5. CG DA contraction,
6. GI DA contraction,
7. save output,
8. write source-completion log.

This is the main measurement workflow.

---

## 4. Important User Parameters

### `da_src_gammalist`

This controls the source Gamma channels in the `CG`/`GI` DA blocks only.

It is a list of source gamma channels to be measured in the nonlocal DA correlators.

Examples:

- `["5"]`
- `["5", "T5"]`
- `["5", "X", "T"]`
- `gammalist`

For each chosen source gamma:

- the code measures a separate DA dataset,
- but still keeps the full sink gamma basis.

So one source choice corresponds to one DA output file, and inside that file the sink gamma index still runs over the full `gammalist`.

---

## 5. Data Flow in the DA Blocks

Inside `Measurement.contract_DA(...)`, the logic is:

1. build the common sink gamma structure,
2. loop over source gamma choices,
3. initialize a forward propagator copy,
4. loop over all Wilson-line separations,
5. shift or gauge-covariantly transport the forward propagator,
6. do the spin-color contraction,
7. project to all sink gamma channels,
8. Fourier transform to momentum space,
9. gather the result to the root rank,
10. save the collected correlators.

The backward propagator is not transported. The negative branch restarts from
the original forward propagator, so the GI updates are the one-link sequence
\(0,-1,-2,\ldots\) rather than a jump from the end of the positive branch.

The local C2 uses the generic relational source mode
`dagger_of_sink`. For every sink channel \(g\),

```text
Gamma_src(g) = gamma5 @ Gamma_sink(g).conj().T @ gamma5
```

Thus the C2 Gamma axis represents paired sink/source channels; it is not a
scan of sink Gammas at one fixed source Gamma. This convention is independent
of `da_src_gammalist`.

---

## 6. Output Layout

### 6.1 2pt output

The 2pt correlator is saved through:

- `save_proton_c2pt_hdf5(...)`

inside `Measurement.contract_2pt_pion(...)`.

It stores the sink gamma channels and momenta for the local 2pt measurement.

### 6.2 DA output

The DA correlators are saved through:

- `save_qTMDWF_hdf5_noRoll(...)`

using tags of the form

```text
{sm_tag}.{block_tag}.src{src_name}
```

where:

- `block_tag` is `CG` or `GI`,
- `src_name` is the selected source gamma label.

Each DA file contains:

- all Wilson-line separations,
- all selected momenta,
- all sink gamma channels,
- all Euclidean times.

---

## 7. Files This Script Depends On

The most important external logic lives in:

- [pyquda_DA_k6.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/pyquda_DA_k6.py)
- [Pyquda_Measurement/pyquda_measurement_utils/pion_qTMDWF_pyquda.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/Pyquda_Measurement/pyquda_measurement_utils/pion_qTMDWF_pyquda.py)
- [Pyquda_Measurement/pyquda_measurement_utils/boosted_smearing_pyquda.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/Pyquda_Measurement/pyquda_measurement_utils/boosted_smearing_pyquda.py)

In particular:

- `contract_2pt_pion(...)` defines the 2pt contraction,
- `create_fw_prop_TMD_CG(...)` defines the fixed-gauge shift,
- `create_fw_prop_PDF_GI(...)` defines the gauge-covariant shift.

---

## 8. Typical Modifications

### To change the quark mass

Edit the `mass` parameter in the script, or use the separate
[pyquda_charm_mass.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/pyquda_charm_mass.py)
debugging script.

### To change DA source channels

Edit:

```python
da_src_gammalist = [...]
```

### To change momentum range

Edit:

- `pzmin`
- `pzmax`

### To change Wilson-line extent

Edit:

- `b_z`

---

## 9. Minimal Conceptual Summary

If someone only needs the shortest handover version:

- The script measures charmonium DA correlators in the `b_T = 0` limit.
- `CG` means shifted propagator without explicit gauge links.
- `GI` means gauge-covariantly shifted propagator, i.e. Wilson line included.
- The sink always runs over the full 16-gamma basis.
- The source gamma choices are set by `da_src_gammalist`.
- The local C2 uses the paired `dagger_of_sink` source convention.
- The workflow is: smeared source -> inversion -> 2pt -> CG DA -> GI DA -> save.
