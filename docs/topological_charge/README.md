# Flowed topological charge: conventions, derivation and validation

## Scope

`TopologicalCharge1pt` measures a deterministic, hadron-independent gluon loop
on each gauge configuration. It contains no stochastic estimator, inversions,
spin-color dilution, source count, or nucleon contractions. The scalar output
has its own schema, `topological_charge_loop_v1`, not an EMT component label.
The companion [LaTeX source](topological_charge.tex) and
[compiled PDF](topological_charge.pdf) record the derivation in typeset form.

### Rebuild the PDF on Aurora

```bash
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-latex.sh
cd /lus/flare/projects/StructNGB/xgao/software_gradientflow/Pyquda_Measurement_develop
tectonic docs/topological_charge/topological_charge.tex
```

Tectonic 0.17.0 and its macro/font cache are installed under
`software_gradientflow/latex`, outside this repository. It automatically
reruns to resolve references and normally emits only the final PDF. After
the initial build, add `--only-cached` to rebuild without network access.
`docs/.gitignore` excludes LaTeX intermediates; the final PDF is tracked.

## 1. Clover normalization and orientation

Directions are `(0,1,2,3)=(x,y,z,t)`, with Euclidean `epsilon_0123=+1`.
Let `C_mu,nu` be the sum of the four consistently oriented plaquettes based at
the same site. Define

\[
 A_{\mu\nu}=\frac{C_{\mu\nu}-C_{\mu\nu}^\dagger}{8}
 -\frac{\mathbf1}{3}\operatorname{Tr}
   \frac{C_{\mu\nu}-C_{\mu\nu}^\dagger}{8},\qquad
 H_{\mu\nu}=-i A_{\mu\nu}.
\]

Thus `A` is anti-Hermitian traceless and `H` is Hermitian traceless, with
`H_nu,mu=-H_mu,nu`. For links written as
`U_mu=exp(i a g A_mu^a T^a)` and Hermitian generators
`Tr(T^a T^b)=delta_ab/2`, an oriented plaquette is
`1+i a^2 g G_mu,nu+...`. Four leaves give `H=a^2 g G+O(a^4)`.
The precise leading discretization correction assumes smooth fields.

The dimensionless density is

\[
 q_{\rm lat}(x)=\frac1{32\pi^2}
 \epsilon_{\mu\nu\rho\sigma}\operatorname{ReTr}
 [H_{\mu\nu}(x)H_{\rho\sigma}(x)]
 =\frac{\operatorname{ReTr}
 (H_{01}H_{23}-H_{02}H_{13}+H_{03}H_{12})}{4\pi^2}.
\]

There are 24 nonzero epsilon terms. Each of the three disjoint plane pairings
occurs eight times: two orders within each pair, and two orders of the pairs.
Antisymmetry cancels the within-pair epsilon signs; trace cyclicity equates
the reversed products. The pairing signs are `+,-,+`. This derivation does
**not** assume that the color matrices commute. Hermiticity implies
`Tr(H K)*=Tr(K H)=Tr(H K)`; taking the real part removes roundoff only.

`q_lat = a^4 q_phys + discretization errors`. With a physical field strength
that does not include the coupling, `q_phys=g^2 epsilon Tr(G G)/(32 pi^2)`.
If a paper absorbs `g` in its field strength, it is absent from that formula.
Do not multiply this link-based implementation by another `g^2`, an EMT factor
of two, a flavor count, or a closed-quark-loop minus sign.

QUDA's total-charge implementation stores anti-Hermitian lower-triangle
planes `(10,20,21,30,31,32)` and uses a negative `1/(4 pi^2)` prefactor.
Substituting `A=iH` and reversing each lower-triangle pair yields exactly the
formula above. The test compares the independently computed total `qcharge()`;
near-integer total charge alone cannot fix a normalization or sign.

| Convention | Relation / caution |
|---|---|
| Hermitian vs anti-Hermitian | `H=-iA`; two factors reverse the trace sign |
| Coordinate order | `epsilon_xyzt=+1`; writing time first changes orientation |
| Fourier transform | `exp(+i q.(x-origin))`, not its complex conjugate |
| Euclidean vs Minkowski | Wick rotation, epsilon and gamma5 must be transformed together |
| Form-factor extraction | The paper's `ubar i gamma5 u` is not an extra `i` in this real Euclidean density |

This version fixes and verifies the Euclidean loop convention. It does not
claim an independently validated Euclidean-to-Minkowski polarized-ratio sign.
That sign must be derived with the actual interpolator, gamma basis, spinor
normalization and projectors when implementing the hadron analysis.

## 2. Fourier projection, source alignment and volume

The stored loop is

\[
 L(\mathbf n,t_f,\tau)=\frac1{V_s}\sum_{\mathbf x}
 e^{2\pi i\sum_j n_j(x_j-o_j)/L_j}q_{\rm lat}(t_f,\mathbf x,\tau),
 \quad o=(0,0,0,0),\quad V_s=L_xL_yL_z.
\]

Consequently `L(-n)=L(n)*`, and
`Q_total(tf)=Vs sum_tau L(0,tf,tau)` is dimensionless. MPI ranks project locally,
insert their contribution at the correct absolute global time, and use one
root-only sum reduction per output flow time. No full lattice density is
gathered or saved. The result is returned only on root.

For a source `s=(s_vec,t0)`, the source-relative loop is

\[
 L_s(\mathbf n,t_f,\tau_{\rm rel})=
 e^{-2\pi i\sum_j n_j(s_j-o_j)/L_j}
 L(\mathbf n,t_f,(t_0+\tau_{\rm rel})\bmod N_t).
\]

The gluonic time alignment is periodic, without a fermionic boundary sign.
With a compatible per-configuration, per-source polarized nucleon correlator,
the disconnected covariance is

\[
 C_{3,\Gamma}^{Q}=V_s\left[
 \langle C_{2,\Gamma}L_s\rangle
 -\langle C_{2,\Gamma}\rangle\langle L_s\rangle\right].
\]

Keep configuration and stream identity and perform alignment before averaging.
Even if the ensemble expectation of a pseudoscalar vanishes by symmetry, do
not silently replace the finite-sample covariance by a product average.
This formula is a correlator building block, not a complete form-factor ratio.
The momentum routing and needed C2 momenta must be derived for the chosen
source/sink projection. Existing `PpUnpol` data must not be assumed to retain
the required spin information. Missing C2 components may need re-contraction
or new inversions if propagators were not retained.

## 3. Flow, RGI limit and physics interpretation

Measurements include zero flow time and each requested endpoint. New topology
uses the same integration substeps for every interval. Output interval and
integration step are distinct quantities. The shared primitive leaves the
historical EMT first-interval ten-substep schedule unchanged.

At fixed positive physical flow time, the continuum loop defines a finite
flowed observable. The RGI operator is obtained by the ordered limits

\[
 F_Q^{\rm RGI}(Q^2)=\lim_{t_f\to0}\lim_{a\to0}
 F_Q^{\rm GF}(Q^2,a,t_f).
\]

Luescher's small-flow-time expansion and Eq. (21) identify the RGI limit;
mixing with the singlet axial divergence is part of the operator analysis.
It does not require a separate axial-divergence measurement for this chosen
limit construction, but finite-time corrections need not be purely linear
in flow time: logarithmic running/mixing can matter. [Luescher (2021)](https://arxiv.org/pdf/2109.07965).

The RGI operator combines the renormalized topological density with an axial
divergence term in a fixed renormalization framework. This explains why a
finite-flow nonforward matrix element must not simply be called the final
RGI density or an MSbar operator. No standalone MSbar conversion is applied
here. [Luescher and Weisz (2021), section 6](https://arxiv.org/pdf/2103.15440).

In a continuum smooth periodic field, the global charge is flow-time
independent; a local nonforward matrix element is not. A clover charge at
finite spacing also has cutoff effects and is not forced to remain constant
or integer. A physical analysis requires a window with
`a << sqrt(8 tf)` and flow radius small compared with insertion-to-source/sink
distances and other resolved scales. The numerical validation times are not
a demonstration that this window exists.

For the proposed nucleon convention,

\[
 \frac{\langle N(p')|q_{\rm top}|N(p)\rangle}{M_N}
 =\bar u(p')i\gamma_5u(p)F_Q(Q^2),\qquad
 d_Q=-F_Q(0).
\]

The forward pseudoscalar bilinear vanishes. Determine `F_Q` at nonzero transfer
and extrapolate to zero; do not divide a zero-transfer correlator by a vanishing
kinematic factor. Generally `Q^2=|p'-p|^2-(E'-E)^2`, not simply spatial transfer
squared. A Breit-frame density is a frame-dependent interpretation, not an
ordinary relativistic probability density. Chiral-limit relations must not be
imposed at physical masses without their corrections.
[Fukushima and Uji (2026)](https://arxiv.org/pdf/2607.03123).

Earlier nucleon calculations using the anomalous Ward identity demonstrate
nonzero-transfer gluonic insertions and polarized signal extraction, but
their overlap-defined density and normalization are not a direct matching
prescription for this flowed-clover implementation.
[Gong et al.](https://arxiv.org/pdf/1511.03671).

## 4. Cost and validation map

Each measured time needs six clover planes, three color traces, one momentum
projection, one host transfer of the small projected array, and one MPI
reduction. There are no inversions. Six clover planes are reused within a
measurement, not cached at every flow time. Flow and clover construction are
expected to dominate at modest momentum count; use measured timings rather
than claiming a speedup from trace counts alone.

For 125 momenta, 21 times and Nt=80, the complex loop occupies 3.20 MiB.
An 80^4 real density would occupy 312.5 MiB **per flow time**, and is not saved.

| Identity / risk | Check |
|---|---|
| Epsilon factor eight, plane signs, noncommuting matrices | `test_epsilon_reduction_and_normalization` |
| Color conjugation and odd spatial reflection | `test_color_gauge_invariance_and_reflection` |
| Clover coefficient, Hermiticity, tracelessness, identity links | `test_clover_projection` and GPU identity case |
| Conjugate Fourier modes, total charge, source phase/time | `test_fourier_source_alignment_and_total` |
| Immutable input, root ownership, endpoints, constant substeps | `test_schedule_input_ownership_and_hdf5` |
| Existing EMT schedule | `test_emt_preserves_first_interval_schedule` plus GPU baseline 5036ce0 |
| Global time placement and root-only MPI | Existing `test_projected_reduce.py` plus GPU decompositions |
| Actual link-level gauge invariance and decomposition independence | Small deterministic GPU fields, 1 and 100 ranks |
| Independent normalization | QUDA `qcharge()` and 24-term epsilon reference |
| Integration error | `.01`, `.005`, `.0025` at the same output times, both flows |
| I/O and provenance | CPU HDF5 test and GPU comparison report |

See [validation status](validation_summary.md) for results, including pending
checks. Never infer successful GPU validation from CPU mocks.

## Native density API precaution

The installed QUDA matrix exponential also has an independent zero-force
singularity: at exactly zero input it evaluates `acos(0/0)`. The GPU validation
therefore checks identity links only at t=0 and uses nontrivial fields for
flow. This is a validation-scope limitation, not a QUDA fix. Zero Fourier
momentum and zero total topological charge remain included. Non-finite data
are rejected by the validation rather than accepted as matching NaNs.

The inspected installed `GaugeDirac.qchargeDensity()` leaves
`compute_qcharge_density=True` after returning a Python-owned density array.
Later observables can then write through a retained pointer. This is a static
API-lifetime concern, not a demonstrated cause of earlier production failures.
This feature never calls that API or changes its flags. Validation uses fresh
gauge copies and the scalar `qcharge()` API, which clears its own flag.
