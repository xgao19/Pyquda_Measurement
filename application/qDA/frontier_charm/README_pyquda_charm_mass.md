# `pyquda_charm_mass.py` README

This DA calculation uses independent positive- and negative-boost source
propagators. It is a two-propagator correlator and does not use the fixed-sink
sequential active-line convention of connected pion qTMD/PDF.

## 1. Purpose

`pyquda_charm_mass.py` is a stripped-down PyQUDA script for tuning the charmonium two-point correlator before running the full DA production workflow.

It keeps only the cheapest and most useful part of the pipeline:

```text
point source -> boosted Gaussian smearing -> propagator inversion -> 2pt contraction
```

The main reason for this script is practical:

- tuning the quark mass is much cheaper with a 2pt correlator than with the full DA measurement,
- tuning the Gaussian smearing width is also easiest at the 2pt level,
- and choosing the best boosted-smearing momentum should also be done with the 2pt signal first.

So this script is the recommended place to debug or tune:

- `mass`
- `width`
- `pos_boost`
- `neg_boost`

before using `pyquda_DA_k6.py`.

---

## 2. Physics Content

This script measures the charmonium meson two-point correlator in momentum space:

\[
C^{(g)}_{2pt}(q,t)
=
\sum_x e^{iq\cdot x}
\mathrm{Tr}_{c,s}
\Big[
\gamma_5\,S_b^\dagger(x)\,\gamma_5\,
\Gamma_g\,
S_f(x)\,
\Gamma_{\mathrm{src}}
\Big].
\]

Here:

- `S_f` is the forward propagator,
- `S_b` is the backward propagator,
- `\Gamma_g` is the sink gamma channel,
- `\Gamma_src` is the source gamma convention used in the 2pt function,
- the phase factor projects onto the requested spatial momentum.

The actual contraction is performed by:

- `Measurement.contract_2pt_pion(...)`

from

- [pion_qTMDWF_pyquda.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/Pyquda_Measurement/pyquda_measurement_utils/pion_qTMDWF_pyquda.py)

Even though the class name says `pion`, in the present workflow it is being used as the meson 2pt contractor for the charmonium study.

---

## 3. What Should Be Tuned Here

There are three main tuning directions.

### 3.1 Quark mass

The quark mass is controlled by:

```python
mass = ...
```

This is the most important spectroscopy parameter.

The usual strategy is:

- use zero or very low momentum,
- keep boosted smearing turned off,
- choose a reasonable fixed Gaussian width,
- then vary `mass` until the extracted meson mass matches the target.

Recommended setup for this stage:

- `parameters["pzmin"] = 0`
- `parameters["pzmax"] = 2`
- `parameters["pos_boost"] = [0, 0, 0]`
- `parameters["neg_boost"] = [0, 0, 0]`

Why this is good:

- the contraction cost is cheap, so including `p_z = 0, 1` is fine,
- zero boost usually gives the cleanest low-momentum signal,
- and the resulting effective mass is easiest to interpret.

Once the meson mass is acceptable, you can keep that `mass` fixed and move on to smearing optimization.

---

### 3.2 Gaussian smearing width

The Gaussian smearing width is controlled by:

```python
parameters["width"] = ...
```

This parameter changes the overlap of the interpolating operator with:

- the ground state,
- the excited states,
- and the noise level.

Recommended setup for this stage:

- keep the tuned `mass` fixed,
- still use zero or very low momentum,
- keep boosted smearing turned off,
- scan several `width` values.

Recommended choices:

- `parameters["pzmin"] = 0`
- `parameters["pzmax"] = 2`
- `parameters["pos_boost"] = [0, 0, 0]`
- `parameters["neg_boost"] = [0, 0, 0]`

The goal is to find a reasonable balance:

- larger width often suppresses excited-state contamination better,
- but overly large width can also worsen noise or distort overlap.

So the practical target is:

- early plateau formation,
- stable effective mass,
- acceptable statistical noise.

---

### 3.3 Boosted smearing momentum

The quark and antiquark boosted-smearing momenta are controlled by:

```python
parameters["pos_boost"] = [0, 0, +k]
parameters["neg_boost"] = [0, 0, -k]
```

In the current script, these are applied through:

- `boosted_smearing(srcD, w=..., boost=parameters["pos_boost"])`
- `boosted_smearing(srcD, w=..., boost=parameters["neg_boost"])`

The usual idea is:

- fix the tuned `mass`,
- fix the chosen Gaussian width,
- then scan `pos_boost = -neg_boost`
- to see which boost gives the best signal for the largest target hadron momentum.

This is especially useful when you care about larger momenta, for example:

- `p_z = 7`
- `p_z = 8`
- `p_z = 9`
- `p_z = 10`

The practical goal is not to make the low-momentum signal best, but to maximize signal quality in the momentum range you actually want to use later in the DA analysis.

---

## 4. Recommended Tuning Workflow

This is the suggested order of operations.

### Step 1: Tune quark mass

Use:

- low momentum only,
- no boosted smearing,
- a conventional smearing width.

A good starting point is:

```python
mass = ...
parameters["pzmin"] = 0
parameters["pzmax"] = 2
parameters["width"] = 1.0
parameters["pos_boost"] = [0, 0, 0]
parameters["neg_boost"] = [0, 0, 0]
```

Then vary `mass` until the extracted charmonium mass is close to the desired target.

---

### Step 2: Tune Gaussian width

Fix the chosen mass, still keep momentum low and boost zero:

```python
parameters["pzmin"] = 0
parameters["pzmax"] = 2
parameters["pos_boost"] = [0, 0, 0]
parameters["neg_boost"] = [0, 0, 0]
```

Now scan:

```python
parameters["width"] = ...
```

and compare:

- how quickly the effective mass plateaus,
- how much excited-state contamination remains,
- how noisy the signal becomes.

Pick a width that gives a good compromise between plateau quality and noise.

---

### Step 3: Tune boosted smearing

Now fix:

- `mass`
- `width`

and scan:

```python
parameters["pos_boost"] = [0, 0, +k]
parameters["neg_boost"] = [0, 0, -k]
```

for several values of `k`.

The best choice is the one that improves the signal most strongly for the largest momentum channels you care about in production.

If the intended production range is

- `p_z = 7` to `10`,

then these are the momenta to inspect most carefully in the 2pt signal.

---

## 5. Momentum Interpretation

The script projects momentum using

```python
p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
```

so the relevant momentum is longitudinal only.

The lattice momentum is

\[
p_z = \frac{2\pi n_z}{L a},
\]

where:

- `n_z` is the integer momentum index,
- `L = 64` is the spatial lattice extent,
- `a` is the lattice spacing.

The corresponding physical momentum in GeV is

\[
p_z[\mathrm{GeV}] =
\frac{2\pi n_z}{L a[\mathrm{fm}]}\times 0.197326.
\]

If the label `l64c64a040` means the lattice spacing is approximately

\[
a \approx 0.040\ \mathrm{fm},
\]

then one unit of lattice momentum is approximately

\[
p_{\mathrm{unit}} \approx 0.485\ \mathrm{GeV}.
\]

In that case:

- `p_z = 7`  corresponds to about `3.40 GeV`
- `p_z = 8`  corresponds to about `3.88 GeV`
- `p_z = 9`  corresponds to about `4.36 GeV`
- `p_z = 10` corresponds to about `4.85 GeV`

This estimate is useful for judging which boosted-smearing momentum range is relevant for your production goals.

If the actual lattice spacing used in analysis differs from `0.040 fm`, these numbers should be updated accordingly.

---

## 6. Main User Parameters

### `mass`

This is the clover quark mass used in:

```python
dirac = core.getClover(...)
```

It controls the physical meson mass and is the main spectroscopy tuning knob.

---

### `parameters["width"]`

This sets the width of the boosted Gaussian smearing kernel.

It controls source/sink overlap with the ground state and excited states.

---

### `parameters["pos_boost"]`

This is the momentum used in boosted smearing for the forward / quark-side source.

Typical choice during tuning:

- `[0, 0, 0]` for mass and width scans,
- `[0, 0, +k]` for boost scans.

---

### `parameters["neg_boost"]`

This is the momentum used in boosted smearing for the backward / antiquark-side source.

Typical choice during tuning:

- `[0, 0, 0]` for mass and width scans,
- `[0, 0, -k]` for boost scans.

The common symmetric choice is:

\[
\mathrm{pos\_boost} = -\,\mathrm{neg\_boost}.
\]

---

### `parameters["pzmin"]`, `parameters["pzmax"]`

These define the momentum channels included in the 2pt projection.

Because the 2pt contraction is cheap, it is fine to include a small low-momentum window such as:

```python
pzmin = 0
pzmax = 2
```

during early tuning.

---

### Source Gamma

The mass-tuning C2 uses the fixed canonical raw source label `5` and scans all
16 sink Gamma channels.  The source label is part of the filename, sample-log
identity and HDF5 provenance.

---

## 7. Script Flow

For each source position, the script does:

1. build a point source,
2. apply boosted Gaussian smearing to source and sink channels,
3. invert forward and backward propagators,
4. construct momentum phases,
5. contract the 2pt correlator,
6. save bookkeeping information for completed source positions.

So the code path is intentionally short and cheap compared with the full DA script.

---

## 8. Output and Logging

The script writes the 2pt correlator through:

- `Measurement.contract_2pt_pion(...)`

and tracks finished source positions through:

```text
sample_log/charm_mass_{sm_tag}_{conf}
```

This keeps mass-tuning bookkeeping separate from the full DA production logs.

---

## 9. Relationship to the Production Script

Use this script first, then move to:

- [pyquda_DA_k6.py](/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/pyquda_DA_k6.py)

once the following are settled:

- quark mass,
- Gaussian smearing width,
- boosted smearing momentum.

So the workflow is:

```text
pyquda_charm_mass.py  ->  tune mass / width / boost
pyquda_DA_k6.py       ->  production DA measurement
```

---

## 10. Minimal Handover Summary

If someone only needs the short version:

- `pyquda_charm_mass.py` is the cheap tuning script.
- Tune `mass` first with zero boost and low momentum.
- Then tune `width`, still with zero boost and low momentum.
- Then fix both and scan `pos_boost = -neg_boost`.
- The momentum range `p_z = 7 - 10` is the most relevant if those are the target production momenta.
- If `a \approx 0.040 fm`, then `p_z = 7 - 10` is about `3.4 - 4.9 GeV`.
