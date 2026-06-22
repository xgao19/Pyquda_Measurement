# Flowed-Quark Ringed Normalization

This workflow produces the ringed-field normalization for flowed quark fields.
It is not EMT-specific: EMT connected/disconnected quark bilinears and future
flowed scalar, vector, axial, tensor, derivative, or multi-quark operators should
use the same factor when their quark mass, Dirac operator, gauge preprocessing,
flow schedule, and flavor convention match.

## Observable

The measured kinetic expectation value is

```text
K(tf) = (1 / V4) sum_x < bar chi_f(tf,x)
                          overleftrightarrow{Dslash}
                          chi_f(tf,x) >
```

The implementation uses 4D stochastic sources.  The official normalization
input is the spacetime-averaged kinetic trace:

```text
avg/kinetic_spacetime[flow]
  = (1 / (N_eff Nt)) sum_{r,t} raw/kinetic_pervec[r, flow, t]
```

## Ringed Factors

For the default single-flavor fundamental SU(3) convention,

```text
Z_ring_bilinear(tf) = -2*Nc / ((4*pi)^2 * tf^2 * K(tf))
Z_ring_field_sqrt(tf) = sqrt(Z_ring_bilinear(tf))
```

The unflowed step has `tf=0`, so `flow=0` factors are stored as `NaN`.

Consumers should apply:

```text
flowed quark bilinear        -> multiply by Z_ring_bilinear[flow]
two flowed quark bilinears   -> multiply by Z_ring_bilinear[flow]**2
single flowed quark field    -> multiply by Z_ring_field_sqrt[flow]
gluon-only operator          -> no ringed factor
unflowed quark operator      -> no ringed factor
```

Mixed flowed/unflowed operators count only the flowed quark fields.

## HDF5 Layout

The output lives under `data/FlowedQuarkRinged/` and contains:

```text
raw/kinetic_pervec              [N_eff, Nflow, Nt]
raw/source_index                [N_eff]
raw/base_noise_index            [N_eff]
raw/hp_index                    [N_eff]
avg/kinetic_spacetime           [Nflow]
avg/Z_ring_field_sqrt           [Nflow]
avg/Z_ring_bilinear             [Nflow]
flow_times                      [Nflow]
```

Important attributes:

```text
measurement = flowed_quark_ringed_norm
normalization_scope = all_flowed_quark_fields
operator = bar_chi_overleftrightarrow_Dslash_chi
Nc = 3
flavor_convention = single_flavor_trace_for_this_dirac_operator
flow_type, flow_epsilon, flow_steps, flow_times
mass, csw, tol, maxiter
gauge_preprocessing
t_boundary
noise_scheme, n_vec, n_zn, hp_num_vectors, hp_ordering
volume_average = spacetime_average_from_raw_kinetic_pervec
flow0_factor = NaN
```

## Aurora Smoke

Use the Aurora PyQUDA develop environment and run multi-rank tests through PALS
inside a compute allocation:

```bash
cd application/flowed_quark_ringed_norm/Aurora
FLOWED_RINGED_FLOW_STEPS=1 \
FLOWED_RINGED_N_VEC=1 \
FLOWED_RINGED_NOISE_SCHEME=zn \
bash submit_or_run_interactive.sh
```

For a hierarchical-probing smoke:

```bash
FLOWED_RINGED_FLOW_STEPS=1 \
FLOWED_RINGED_N_VEC=1 \
FLOWED_RINGED_NOISE_SCHEME=hierarchical_probing \
FLOWED_RINGED_HP_NUM_VECTORS=2 \
bash submit_or_run_interactive.sh
```

For l64 config 1050 smoke, set the production gauge path, mass, flow type,
epsilon, and steps to match the connected production run, but keep the stochastic
source count tiny during validation.
