# Canonical Gamma-basis schema and raw-bilinear analysis

This is the analysis-facing registry for the Gamma convention shared by the
connected pion/proton qTMD, PDF and EMT workflows and by disconnected
quark-loop workflows.  The code source of truth is
`pyquda_measurement_utils/fermion_bilinear_basis.py`; HDF5 files also store
`gamma_list`, `gamma_pyquda_ids`, `gamma_matrices`, and
`physical_from_pyquda`, so an analysis should verify the file metadata instead
of assuming an ordering.

## Schema registry

The current HDF5 schema identifier is

```text
pyquda_bitmask16_with_physics_transform_v1
```

This string is a version key, not a self-contained mathematical definition.
It means that the saved Gamma axis contains the **raw matrices returned by
PyQUDA bitmask IDs**, while the same file carries an explicit transform to the
uniform physics-labelled basis.  The convention is fully specified by the
following HDF5 fields:

```text
attribute gamma_basis_schema
attribute gamma_basis_order
attribute physical_transform_definition
attribute gamma5_definition
attribute axial_definition
attribute raw_tensor_definition
attribute hermitian_tensor_from_raw_factor

dataset   gamma_list                  [16]
dataset   gamma_pyquda_ids            [16]
dataset   gamma_matrices              [16,4,4]
dataset   physical_gamma_list         [16]
dataset   physical_from_pyquda        [16,16]
dataset   gamma5_hermiticity_partner  [16]
dataset   gamma5_hermiticity_sign     [16]
```

For qTMD/PDF files,
`corr[wilson,momentum,gamma,time]` is contracted with
`gamma_matrices[gamma]`.  It has **not** been transformed to the physical
basis.  Likewise, EMT primitive local and derivative Gamma axes are raw until
`physical_from_pyquda` is applied.  The matrix dataset, rather than the
human-readable label alone, is the definitive record of what production used.

An analysis can make this convention check explicit:

```python
import h5py
import numpy as np

with h5py.File(path, "r") as h5:
    if h5.attrs["gamma_basis_schema"] != (
        "pyquda_bitmask16_with_physics_transform_v1"
    ):
        raise ValueError("unsupported Gamma convention")
    labels = h5["gamma_list"].asstr()[...]
    gamma_raw = h5["gamma_matrices"][...]
    raw_to_physical = h5["physical_from_pyquda"][...]

    # Example for corr[wilson,momentum,gamma,time].
    corr_raw = h5["corr"][...]
    corr_physical = np.einsum(
        "ab,wmbt->wmat", raw_to_physical, corr_raw
    )
```

Here `corr_physical[..., A, :]` corresponds to
\(\Gamma_A^{\rm phys}\), whereas `corr_raw[..., B, :]` is the actual
PyQUDA matrix used in the contraction.

## Euclidean representation used by PyQUDA

Directions are ordered `X,Y,Z,T`, corresponding to
\(\gamma_1,\gamma_2,\gamma_3,\gamma_4\).  In the DeGrand--Rossi representation
returned by the current PyQUDA `gamma()` implementation,

\[
\gamma_1=\begin{pmatrix}
0&0&0&i\\0&0&i&0\\0&-i&0&0\\-i&0&0&0
\end{pmatrix},\quad
\gamma_2=\begin{pmatrix}
0&0&0&-1\\0&0&1&0\\0&1&0&0\\-1&0&0&0
\end{pmatrix},
\]
\[
\gamma_3=\begin{pmatrix}
0&0&i&0\\0&0&0&-i\\-i&0&0&0\\0&i&0&0
\end{pmatrix},\quad
\gamma_4=\begin{pmatrix}
0&0&1&0\\0&0&0&1\\1&0&0&0\\0&1&0&0
\end{pmatrix},
\]
\[
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4
=\operatorname{diag}(1,1,-1,-1).
\]

They obey \(\gamma_\mu^\dagger=\gamma_\mu\) and
\(\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}\).  The complete raw HDF5
Gamma axis is:

| HDF5 index | label | PyQUDA ID | raw matrix |
|---:|---|---:|---|
| 0 | `5` | 15 | \(\gamma_5\) |
| 1 | `T` | 8 | \(\gamma_4\) |
| 2 | `T5` | 7 | \(-\gamma_4\gamma_5\) |
| 3 | `X` | 1 | \(\gamma_1\) |
| 4 | `X5` | 14 | \(\gamma_1\gamma_5\) |
| 5 | `Y` | 2 | \(\gamma_2\) |
| 6 | `Y5` | 13 | \(-\gamma_2\gamma_5\) |
| 7 | `Z` | 4 | \(\gamma_3\) |
| 8 | `Z5` | 11 | \(\gamma_3\gamma_5\) |
| 9 | `I` | 0 | \(\mathbf 1\) |
| 10 | `SXT` | 9 | \(\gamma_1\gamma_4\) |
| 11 | `SXY` | 3 | \(\gamma_1\gamma_2\) |
| 12 | `SXZ` | 5 | \(\gamma_1\gamma_3\) |
| 13 | `SYT` | 10 | \(\gamma_2\gamma_4\) |
| 14 | `SYZ` | 6 | \(\gamma_2\gamma_3\) |
| 15 | `SZT` | 12 | \(\gamma_3\gamma_4\) |

For \(\mu<\nu\), the raw tensor channel is
\(\gamma_\mu\gamma_\nu=[\gamma_\mu,\gamma_\nu]/2\), without an extra
factor of \(i\).  Multiply by `1j` to obtain the Hermitian convention
\(i[\gamma_\mu,\gamma_\nu]/2\).

The stored physical transform is defined by

\[
\Gamma_A^{\rm phys}=\sum_B
  (\texttt{physical\_from\_pyquda})_{AB}\Gamma_B^{\rm raw}.
\]

It is the identity except for `Y5` and `T5`, whose diagonal entries are `-1`.
Thus every physical axial label consistently means \(\gamma_\mu\gamma_5\).
Concretely,

| label | matrix used in production | physics-labelled matrix |
|---|---|---|
| `X5` | \(\gamma_1\gamma_5\) | \(\gamma_1\gamma_5\) |
| `Y5` | \(-\gamma_2\gamma_5\) | \(\gamma_2\gamma_5\) |
| `Z5` | \(\gamma_3\gamma_5\) | \(\gamma_3\gamma_5\) |
| `T5` | \(-\gamma_4\gamma_5\) | \(\gamma_4\gamma_5\) |

The tensor labels require a separate convention choice.  Production stores
\(\gamma_\mu\gamma_\nu=[\gamma_\mu,\gamma_\nu]/2\).  The common Hermitian
physics convention \(i[\gamma_\mu,\gamma_\nu]/2\) is therefore `1j` times the
stored raw channel; this factor is not part of `physical_from_pyquda`.

## Reconstructing the disconnected EMT

Operator schema v5 stores the completed two-sided derivative, not the
historical right-acting-only contraction. With
`Gamma_sharp = gamma5 @ Gamma.conj().T @ gamma5`, production forms

```text
L_D[A,mu,q] = -0.5 * (
    A_right[A,mu,q] - A_right[A_sharp,mu,-q].conj())
```

where `A_right` uses the symmetric lattice derivative
`(Dplus-Dminus)/2`. The saved `L_D` already contains one closed-fermion-loop
Wick minus. The datasets `gamma5_hermiticity_partner` and
`gamma5_hermiticity_sign` specify the exact raw-basis map. Internal `-q`
projections are not added to the canonical `qext` axis.

This completion matters for nonzero momentum and also for the fixed-time
temporal derivative at zero spatial momentum. Old one-sided EMT and ringed
files cannot be interpreted as schema-v5 data.

The production completion is not expected to match an explicit
`covDev(xi_f)` left-derivative estimator source by source. The direct form

```text
-0.5 * [xi_f^dag P Gamma D eta_f - (D xi_f)^dag P Gamma eta_f]
```

and the production `Gamma_sharp/-q` form are distinct stochastic quadratic
forms. Hermiticity and cyclicity prove equality only after the noise average
has reconstructed the trace. Both are unbiased; their finite-source
difference has zero expectation but nonzero variance. An actual S8T8 test with
256 counter-Z4 sources and `q=0,+/-x,+/-y,+/-z` found a global paired
difference of `0.981` SEM for solved fields at tolerance `1e-15`; one-rank and
four-rank means agreed at relative L2 approximately `2.0e-16`.

The canonical EMTquarkLoop primitive axes are

```text
raw/local_bilinear_pervec      [source,gamma,q,flow,t_abs]
raw/derivative_bilinear_pervec [source,gamma,derivative,q,flow,t_abs]
avg/local_bilinear             [gamma,q,flow,t_abs]
avg/derivative_bilinear        [gamma,derivative,q,flow,t_abs]
```

Raw per-vector data are spatial sums and have not been divided by
`volume_norm = Lx*Ly*Lz`.  The `avg` data have already been averaged over the
source axis and divided by `volume_norm`.

The vector positions must be looked up from `gamma_list`; with the current
schema they are `[X,Y,Z,T] = [3,5,7,1]`.  Define

\[
B_{\nu\mu}=L^D_{\gamma_\nu,\mu},\qquad
T_{\mu\nu}=\frac12(B_{\mu\nu}+B_{\nu\mu}).
\]

The following portable NumPy example reconstructs every averaged component
and checks the stored \(T_{44}\):

```python
import h5py
import numpy as np

with h5py.File("example.EMTquarkLoop.h5", "r") as h5:
    labels = [x.decode() for x in h5["gamma_list"][...]]
    vector = [labels.index(x) for x in ("X", "Y", "Z", "T")]
    d_avg = h5["avg/derivative_bilinear"][...]
    # B axes: [nu,mu,q,flow,t_abs]
    B = np.take(d_avg, vector, axis=0)
    T = 0.5 * (B + np.swapaxes(B, 0, 1))
    np.testing.assert_allclose(T[3, 3], h5["avg/Tmunu/T44"][...])
```

For stochastic errors, do not load a large production source axis at once.
Stream blocks and apply the spatial-volume normalization only after summing:

```python
with h5py.File("example.EMTquarkLoop.h5", "r") as h5:
    raw = h5["raw/derivative_bilinear_pervec"]
    labels = [x.decode() for x in h5["gamma_list"][...]]
    vector = [labels.index(x) for x in ("X", "Y", "Z", "T")]
    total = np.zeros((4, 4) + raw.shape[3:], np.complex128)
    for start in range(0, raw.shape[0], 8):
        block = raw[start:start + 8]
        B = np.take(block, vector, axis=1)  # [source,nu,mu,q,flow,t]
        total += np.sum(0.5 * (B + np.swapaxes(B, 1, 2)), axis=0)
    T = total / raw.shape[0] / float(h5.attrs["volume_norm"])
```

HP data must first be grouped into complete bases using
`raw/base_noise_index` and `raw/hp_index`; individual partial HP prefixes are
not independent estimators.

## Other primitive operators

To convert a raw local or derivative Gamma axis to the physical basis:

```python
def physical_gamma_axis(values, transform, gamma_axis):
    moved = np.moveaxis(values, gamma_axis, 0)
    result = np.tensordot(transform, moved, axes=(1, 0))
    return np.moveaxis(result, 0, gamma_axis)

with h5py.File("example.EMTquarkLoop.h5", "r") as h5:
    local_phys = physical_gamma_axis(
        h5["avg/local_bilinear"][...],
        h5["physical_from_pyquda"][...],
        gamma_axis=0,
    )
```

After this transform, selecting physical `[X5,Y5,Z5,T5]` from the derivative
array and symmetrizing in the same way gives

\[
A_{\mu\nu}=\frac12\left(
L^D_{\gamma_\mu\gamma_5,\nu}
+L^D_{\gamma_\nu\gamma_5,\mu}\right).
\]

Selecting `[SXY,SXZ,SXT,SYZ,SYT,SZT]` from the local array gives the six raw
tensor-current channels.  Multiply those values by `1j` only when the intended
analysis convention is the Hermitian tensor current.

## Connected primitive axes

For pion connected three-point files:

```python
D = h5["C3_derivative_bilinear"][...]
# [tsep,gamma,derivative,q,flow,t]
B = np.take(D, vector, axis=1)
T = 0.5 * (B + np.swapaxes(B, 1, 2))
T_in_file_order = T.transpose(0, 4, 3, 1, 2, 5)
# [tsep,flow,q,mu,nu,t] == C3_Tmunu
```

For proton connected three-point files:

```python
D = h5["C3_derivative_bilinear"][...]
# [flavor,polarization,gamma,derivative,q,flow,t]
B = np.take(D, vector, axis=2)
T = 0.5 * (B + np.swapaxes(B, 2, 3))
T_in_file_order = T.transpose(0, 1, 5, 4, 2, 3, 6)
# [flavor,polarization,flow,q,mu,nu,t] == C3_Tmunu
```

Connected primitives and their derived datasets are already contracted
correlators; no `volume_norm` division should be added during this
reconstruction.

## Storage weight of one current file

No EMT datasets are compressed.  Therefore element counts give the large-file
storage ratios directly, while HDF5 metadata matters only for small test files.

For a disconnected EMTquarkLoop with large `N_eff`, the source-scaling payload is

```text
derivative primitive : local primitive : flowed-noise norm
                     = 64 : 16 : 1
```

The embedded ringed kinetic has only one momentum channel and adds
`1/Nq` unit on this scale.  In the actual S8T8, `N_eff=8192`, `Nq=9`,
`Nflow=2`, `Nt=8` 4D-HP16 file (1,531,264,312 bytes):

| part | bytes | share of complete file |
|---|---:|---:|
| `raw/derivative_bilinear_pervec` | 1,207,959,552 | 78.886% |
| `raw/local_bilinear_pervec` | 301,989,888 | 19.722% |
| `raw/flowed_noise_norm_pervec` | 18,874,368 | 1.233% |
| `derived/ringed/kinetic_pervec` | 2,097,152 | 0.137% |
| all averaged primitives and ten `avg/Tmunu` datasets | 209,664 | 0.014% |
| source bookkeeping, basis metadata, and HDF5 overhead | remainder | 0.008% |

Thus preserving all per-vector 64 derivative channels, not the derived EMT,
sets the disconnected file size.

For either pion or proton connected three-point scientific arrays, all leading
axes cancel and the logical payload ratio is

```text
C3_derivative_bilinear : C3_local_bilinear : C3_Tmunu : C3_chi
                       = 64 : 16 : 16 : 1
```

That is 65.98%, 16.49%, 16.49%, and 1.03% of the four main arrays.  A real
S8T8 proton `tsep=4` file with one polarization, nine momenta, and two flow
times was 358,028 bytes: derivative 61.78%, local 15.44%, derived `C3_Tmunu`
15.44%, `C3_chi` 0.97%, basis datasets 2.29%, and HDF5 metadata/overhead 4.08%.
For larger production files, the ratios approach `64:16:16:1`.

An EMTgluonLoop file has ten equal-shape upper-triangle `Tmunu/Tij` datasets, so
each component is 10% of its scientific array payload.
