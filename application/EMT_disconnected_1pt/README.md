# EMT Disconnected One-Point Workflows

This application measures hadron-independent flowed one-point building blocks
used for EMT disconnected diagrams and ringed-fermion normalization studies.
The Perlmutter entry points are:

```bash
bash perlmutter/run_quark_1pt.sh
bash perlmutter/run_gluon_1pt.sh
```

The default smoke-test gauge is:

```text
Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

## Observables

The quark workflow estimates the stochastic loop

```text
L_q,munu(q,tau;t_f) = sum_x exp(i q.x) Tr_sc[eta^\dagger(x,t_f) Gamma_munu xi(x,t_f)]
```

where `t_f` is the gradient-flow time.  The implementation stores the
symmetrized upper-triangle components `mu <= nu` of `Tmunu`; the missing lower
triangle is obtained by symmetry in analysis.

The quark workflow also stores `CHI`, which is useful for flowed-fermion and
ringed-fermion normalization diagnostics.  The common ringed-fermion kinetic
combination should be reconstructed downstream from the flowed kinetic
expectation values, typically using the zero-momentum diagonal EMT components.

The gluon workflow stores the flowed gluonic EMT building block

```text
L_g,munu(q,t_f) = sum_x exp(i q.x) O_g,munu(x,t_f)
```

again with only the upper triangle written.

## Disconnected Diagram Combination

For a hadron two-point function `C2_H(p,t)` and a one-point loop
`L_munu(q,tau;t_f)`, the disconnected three-point building block is formed in
analysis as

```text
C3_disc,munu(pf,pi;t,tau;t_f)
  = < C2_H(pf,t) L_munu(q,tau;t_f) >cfg
    - < C2_H(pf,t) >cfg < L_munu(q,tau;t_f) >cfg
```

with

```text
q = pf - pi
```

The corresponding ratio is usually built as

```text
R_disc,munu(t,tau;t_f) = C3_disc,munu(t,tau;t_f) / C2_H(t)
```

up to the same kinematic, renormalization, and gradient-flow matching factors
used for the connected EMT analysis.  Vacuum subtraction must be performed at
the ensemble-analysis level because the one-point function is hadron
independent.

## Hierarchical Probing

The quark workflow supports two stochastic-source schemes:

```text
EMT_1PT_NOISE_SCHEME=zn
EMT_1PT_NOISE_SCHEME=hierarchical_probing
```

For hierarchical probing,

```text
effective_n_inversions = n_base_noise * hp_num_vectors
```

where `n_base_noise` is `EMT_1PT_N_VEC` and `hp_num_vectors` is
`EMT_1PT_HP_NUM_VECTORS`.  `hp_num_vectors` must be a positive power of two.

Two site-orderings are currently available:

```text
EMT_1PT_HP_ORDERING=global_xyzt_gray_projected_to_evenodd
EMT_1PT_HP_ORDERING=spatial_xyz_then_t_gray_projected_to_evenodd
```

The default is `global_xyzt_gray_projected_to_evenodd` to preserve the validated
baseline behavior.  For production studies, `spatial_xyz_then_t_gray_projected_to_evenodd`
is a useful candidate because early HP vectors emphasize spatial separation
before time separation.

No spin-color dilution or time dilution is currently implemented in this
workflow.

## HDF5 Layout

Quark output:

```text
attrs/
  measurement
  flow_type, flow_epsilon, flow_steps, flow_times
  qext, pf, p_2pt
  volume_norm
  upper_triangle_only
  mass, csw, tol, maxiter
  n_vec, n_base_noise, effective_n_inversions
  n_zn, rand_seed
  noise_scheme, hp_num_vectors, hp_ordering

raw/Tmunu_pervec
raw/CHI_pervec
raw/source_index
raw/base_noise_index
raw/hp_index

avg/CHI
avg/Tmunu/T11, T12, ..., T44
```

The bookkeeping datasets mean:

```text
source_index      effective source index after HP expansion
base_noise_index  original stochastic base-noise index
hp_index          hierarchical-probing vector index for that base noise
```

Gluon output:

```text
attrs/
  measurement
  flow_type, flow_epsilon, flow_steps, flow_times
  qext, pf, p_2pt
  volume_norm
  upper_triangle_only

Tmunu/T11, T12, ..., T44
```

## Minimal HP Smoke Test

Example:

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/EMT_disconnected_1pt/perlmutter

EMT_1PT_FLOW_STEPS=1 \
EMT_1PT_QMAX=0 \
EMT_1PT_N_VEC=1 \
EMT_1PT_NOISE_SCHEME=hierarchical_probing \
EMT_1PT_HP_NUM_VECTORS=2 \
EMT_1PT_HP_ORDERING=spatial_xyz_then_t_gray_projected_to_evenodd \
bash run_quark_1pt.sh
```

Expected checks:

```text
attrs/noise_scheme = hierarchical_probing
attrs/hp_num_vectors = 2
attrs/effective_n_inversions = 2
raw/source_index = [0, 1]
raw/base_noise_index = [0, 0]
raw/hp_index = [0, 1]
```
