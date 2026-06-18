# Proton EMT Connected Validation Summary

This note summarizes the Aurora S8T32 validation evidence for the connected proton EMT implementation. It is a regression and implementation sanity record, not a proof of final physical normalization, renormalization, operator mixing, disconnected pieces, or GFF extraction.

## Scope

Validated code paths:

- connected proton sequential-source contraction
- raw-only sequential builder used by proton EMT
- left-acting covariant derivative on the raw sequential propagator
- `C3_chi` and `C3_Tmunu` HDF5 output structure
- `FLOW_STEPS=0` and `FLOW_STEPS=1`
- `QMAX=0` and `QMAX=1` momentum phases
- U insertion four-term algebra for the two identical up-quark contributions

Main validation directory:

```text
/lus/flare/projects/StructNGB/xgao/run/test_gauge/EMT_proton/validation_suite_20260603_141831
```

Runtime setup:

```text
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh
backend="dpnp", backend_target="sycl"
h5py.get_config().mpi == True
EMT_PROTON_DISABLE_SMEARING=1
EMT_PROTON_GAUSS_SMEAR=0
EMT_PROTON_MG_BLOCK=none
```

All tests used the S8T32 test gauge and 2 MPI ranks with `mpi_geometry=1.1.1.2`. HYP gauge preprocessing was kept enabled. The point-source/sink setting was used to isolate EMT and sequential-source gauge covariance from boosted-smearing behavior.

## Raw-Only Sequential Builder

The memory-saving EMT path builds raw sequential propagators with `create_bw_seq_raw_pyquda(...)` and then finalizes them with the same gamma5-hermiticity transform used by the original `create_bw_seq_pyquda(...)` return path.

Regression result:

```text
finalized public sequential vs raw-only + final transform
maxabs_diff = 0.0
rel_diff    = 0.0
```

This confirms the raw-only builder is algebraically equivalent to the original finalized sequential object while avoiding simultaneous retention of `dst_seq` and `raw_seq` in the proton EMT path.

## Gauge Covariance

Gauge covariance was tested against both a constant global SU(3) transform and a deterministic local SU(3) transform. Gauge-invariant observables compared were `C2`, `C3_chi`, and `C3_Tmunu`.

For `FLOW_STEPS=0`, `QMAX=0`, `T_SEPS=2`:

```text
global vs baseline:
  C2 rel diff       = 2.0400105459331367e-10
  C3_chi rel diff   = 1.3562834773044713e-11
  C3_Tmunu rel diff = 7.709301865102795e-12

local vs baseline:
  C2 rel diff       = 1.1442846823864556e-10
  C3_chi rel diff   = 5.0028681744551824e-11
  C3_Tmunu rel diff = 9.244013511964703e-12
```

Baseline output checks:

```text
C2 shape              = [32]
C3_chi shape          = [2, 1, 1, 1, 1, 32]
C3_Tmunu shape        = [2, 1, 1, 1, 1, 4, 4, 32]
C2 file match maxabs  = 0.0
Tmunu symmetry maxabs = 0.0
finite C2/C3          = True
```

Interpretation: the connected EMT insertion is gauge-covariant at solver/reduction roundoff. This is the central regression for the left-acting derivative fix: the derivative must act on the raw sequential propagator before final gamma5/conjugation.

## Flow Step 1

The gauge-transform tests were repeated with `FLOW_STEPS=1`.

Output structure and flow separation:

```text
C3_chi shape                 = [2, 1, 1, 2, 1, 32]
C3_Tmunu shape               = [2, 1, 1, 2, 1, 4, 4, 32]
flow chi step diff maxabs    = 3.7546601424624366e-05
flow Tmunu step diff maxabs  = 7.517508948577508e-05
Tmunu symmetry maxabs        = 0.0
finite C2/C3                 = True
```

Gauge covariance across the full flow-step axis:

```text
global C3_Tmunu rel diff = 7.709301865102795e-12
local  C3_Tmunu rel diff = 9.244013511964703e-12
```

Interpretation: step 1 changes the correlators nontrivially while preserving gauge covariance at the same roundoff scale as `FLOW_STEPS=0`.

## QMAX=1 Phase And Momentum Checks

The `QMAX=1` run produced the expected 27 momentum-transfer vectors:

```text
momentum_transfer_list shape = [27, 4]
unique momenta               = 27
contains [-1,0,1]^3 x {0}    = True
q=0 index                    = 13
finite C3_chi/C3_Tmunu       = True
```

The q=0 slice exactly matched the independent `QMAX=0` baseline:

```text
C3_chi q=0 maxabs diff   = 0.0
C3_chi q=0 rel diff      = 0.0
C3_Tmunu q=0 maxabs diff = 0.0
C3_Tmunu q=0 rel diff    = 0.0
```

Pure phase-library checks passed:

```text
phase(-q) - conj(phase(q)) maxabs  = 0.0
real-field F(-q)-conj(F(q)) maxabs = 0.0
shifted-delta self-consistency     = 0.0
num momenta                        = 27
```

## U Insertion Four-Term Sanity

The current U insertion algebra was decomposed into four terms `R1`, `R2`, `R3`, and `R4`.

Source-level identity:

```text
up_quark_insertion_pyquda == -(R1+R2+R3+R4)
maxabs diff = 0.0
rel diff    = 0.0
```

Linearity through time slicing, momentum projection, identity smearing, and sequential inversion:

```text
raw full U sequential vs sum(term-by-term raw sequential)
maxabs diff = 1.1375467583158215e-15
rel diff    = 6.410260242046326e-12
```

Finalization and scalar contraction consistency:

```text
public final sequential vs raw-only final transform maxabs diff = 0.0
scalar public vs raw-final maxabs diff = 0.0
```

U and D raw sequential objects were not accidentally identical:

```text
U raw vs D raw maxabs diff = 8.67703074251285e-05
U raw vs D raw rel diff    = 0.4889647373273185
```

Term norms:

```text
R1 source norm = 0.5107448377342304
R2 source norm = 0.3600856319024264
R3 source norm = 0.12823014069611352
R4 source norm = 0.12822725950899447
full source norm = 1.052219495381614

R1 raw norm = 0.000607659762253114
R2 raw norm = 0.0005007871421310211
R3 raw norm = 0.0002662499020316525
R4 raw norm = 0.0002650693775078178
raw full norm = 0.0016184913932864025
down raw norm = 0.0007393849924803527
```

Interpretation: all four U terms are present, nonzero, and linearly preserved through sequential inversion. The two smaller identical-up exchange-like terms, `R3` and `R4`, are similar in norm but not artificially forced equal.

## Remaining Physics Checks

These validation runs do not settle:

- overall Euclidean normalization and factors of `1/2` relative to a chosen publication convention
- ringed-fermion normalization; the connected `C3_Tmunu` output is an unringed flowed bilinear
- renormalization and quark/gluon EMT mixing
- disconnected diagrams and vacuum subtraction
- finite-volume or continuum extrapolation behavior
- final gravitational form factor extraction

Those should be checked at the analysis level against the chosen EMT convention and reference calculation.
