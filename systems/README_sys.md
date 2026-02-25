# README: PyQUDA Multigrid Inversion & Contraction Reproducibility (Aurora vs Frontier vs Perlmutter)

## 1. Goal
Compare **Aurora**, **Frontier**, and **NERSC Perlmutter** results for PyQUDA **multigrid** propagator inversions, focusing on:
- **Reproducibility** (repeat runs with identical settings)
- Sensitivity to **GPU geometry / parallel decomposition**
- **Cross-machine** differences
- **Contraction** consistency under different backends

---

## 2. Controlled Setup
- Gauge file: `l6464f21b7130m00119m0322a.1050.coulomb.1e-14.HYP`
- Dirac construction:
  ```python
  dirac = core.getDirac(latt_info, mass, tol, 5000, xi_0, csw_r, csw_t, multigrid)
  ```
- Primary variables:
  - `tol` (tested: `1e-10`, `1e-12`)
  - GPU count (and implied GPU geometry / MPI decomposition)
  - Machine (Aurora vs Frontier vs Perlmutter)

---

## 3. Difference Metric
Relative 2-norm difference between two propagators `a` and `b`:
\[
\text{rel\_diff} = \frac{\|a-b\|_2}{\|b\|_2}
\]

Implementation (equivalent form):
```python
rel = ((a-b).norm2()**0.5) / (b.norm2()**0.5)
```

---

## 4. Results Summary

### 4.1 `tol = 1e-10` (multigrid inversion)

#### (A) Same machine, different GPU geometry / decomposition
- **Aurora** (32 / 64 / 128 GPUs):
  - numerator/denominator: `4.448624904681622e-09 / 1.3505199017357683`
  - rel diff: ~ **3.3e-09**
- **Frontier** (32 / 64 GPUs):
  - numerator/denominator: `3.773176817373545e-09 / 1.3505199024428265`
  - rel diff: ~ **2.8e-09**
- **Perlmutter** (16 / 32 GPUs):
  - numerator/denominator: `2.718988359493192e-09 / 1.350519901872753`
  - rel diff: ~ **2.0e-09**

**Conclusion:** Changing GPU geometry introduces **~(2–3)e-9** relative differences across all three systems.

#### (B) Repeatability (same settings, repeated runs)
- **Frontier**: two consecutive inversions with identical settings → **difference = 0**
- **Aurora**: most runs → **difference = 0**; occasional:
  - `1.257066678898858e-10 / 1.3505199017352623` → rel diff ~ **9.3e-11**
- **Perlmutter**: same settings repeated runs → **difference = 0**

**Conclusion:** Results are generally **reproducible**; rare non-zero cases (seen on Aurora) are consistent with the `1e-10` tolerance scale.

#### (C) Cross-machine (fixed 32 GPUs; compare to Aurora/Frontier)
- **Aurora vs Frontier** (32 GPUs):
  - `1.317790151400172e-08 / 1.3505199024170775` → rel diff ~ **9.8e-09**
- **Perlmutter vs Aurora** (32 GPUs):
  - `6.372621114401625e-09 / 1.3505199018319425` → rel diff ~ **4.7e-09**
- **Perlmutter vs Frontier** (32 GPUs):
  - `1.4079119321861158e-08 / 1.3505199024170793` → rel diff ~ **1.0e-08**

**Conclusion:** Cross-machine differences at `tol=1e-10` are **~(5e-9–1e-8)**, typically larger than same-machine geometry changes.

---

### 4.2 `tol = 1e-12` (multigrid inversion, focus on 32 GPUs)

#### (A) Repeatability (same settings, repeated runs)
- **Frontier**: two consecutive inversions → **difference = 0**
- **Aurora**: most runs → **difference = 0**
- **Perlmutter**: (not explicitly re-tested here) — cross-machine numbers below are from 32-GPU comparisons.

#### (B) Cross-machine (fixed 32 GPUs)
- **Aurora vs Frontier** (32 GPUs):
  - `1.49523450779167e-10 / 1.3505199032156316` → rel diff ~ **1.1e-10**
- **Perlmutter vs Aurora** (32 GPUs):
  - `6.176887763827968e-11 / 1.3505199032244184` → rel diff ~ **4.6e-11**
- **Perlmutter vs Frontier** (32 GPUs):
  - `1.6094743909577198e-10 / 1.3505199032156157` → rel diff ~ **1.2e-10**

**Conclusion:** Tightening the tolerance from `1e-10` to `1e-12` reduces cross-machine differences to **~(5e-11–1e-10)**.

---

## 5. Contraction Test (backend-only; propagator fixed)
Goal: isolate contraction differences by **fixing the propagator input**.

- **Aurora backend:** `dpnp.einsum`
- **Frontier backend:** `opt_einsum.contract`
- **Perlmutter backend:** `opt_einsum.contract`

### Test A (Aurora vs Frontier)
- Aurora loaded the propagator saved on Frontier (from the `tol=1e-12` inversion), then only performed contraction.
- Result: Aurora vs Frontier contraction relative difference: ~ **2e-15**

### Test B (Perlmutter vs Frontier)
- Perlmutter loaded the propagator saved on Frontier (from the `tol=1e-12` inversion), then only performed contraction.
- Result: difference reported as `3.84978606e-55` (effectively **0** within double precision)

**Conclusion:** Contraction backend differences are negligible; the dominant discrepancies originate from **inversion**, not contraction.

---

## 6. Practical Error-Scale Takeaways (one-line memory)
- Same machine, different GPU geometry (`tol=1e-10`): **~(2–3)e-9** (Aurora/Frontier/Perlmutter)
- Same machine, same settings, repeated runs: mostly **0**; occasional **~1e-10** on Aurora (consistent with `tol=1e-10`)
- Cross-machine at 32 GPUs:
  - `tol=1e-10`: **~(5e-9–1e-8)**
  - `tol=1e-12`: **~(5e-11–1e-10)**
- Contraction-only (propagator fixed): **~1e-15 or smaller (effectively 0)**

---

## 7. One-Sentence Summary
PyQUDA multigrid inversions are broadly **reproducible** on Aurora, Frontier, and NERSC Perlmutter. Changing GPU geometry introduces **~(2–3)e-9** relative differences, while cross-machine differences are **~(5e-9–1e-8)** at `tol=1e-10` and shrink to **~(5e-11–1e-10)** at `tol=1e-12`. Contraction backend differences are **negligible** when the propagator is held fixed.
