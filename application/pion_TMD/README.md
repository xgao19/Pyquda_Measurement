# Connected Pion TMD Application

This directory contains the connected pion TMD workflow with optional CG qTMD,
GI qTMD, and PDF/local-limit contractions.

This is the canonical connected pion entry point for CG qTMD, GI qTMD, CG PDF,
and GI PDF.  The former duplicate CG-only application has been removed.  The GI
qTMD path uses the fixed-length staple convention shared with the disconnected
qTMD code. Both connected and disconnected production obtain the shared
Wilson-index geometry, CG/PDF shifts, and cached GI staple transport from
`pyquda_measurement_utils/qtmd_operator_utils.py`.

## Perlmutter Entry Point

Run the smoke/production wrapper from the repository root or from this
directory:

```bash
bash application/pion_TMD/perlmutter/run_pion_TMD.sh
```

For momentum smearing, pass both quark-line boosts explicitly:

```bash
bash application/pion_TMD/perlmutter/run_pion_TMD.sh \
  --pos-boost 0.0.1 --neg-boost 0.0.-1
```

The connected-line convention is:

```text
positive-boost line = spectator
negative-boost line = active operator line
```

Equal boosts require one source inversion and reuse a copy.  Unequal boosts
require two inversions.  The positive spectator is positive-boost smeared at
the sink before constructing the sequential source; the outer sequential
smearing uses the negative boost.  CG/GI qTMD and CG/GI PDF operators all act
on the unsmeared-at-insertion negative active propagator.  Only this
negative-active orientation is produced.  The exchanged orientation requires
swapping the two complete lines, not only swapping one smearing argument.

Useful runtime switches:

- `PION_TMD_RUN_CG_QTMD=0/1`
- `PION_TMD_RUN_GI_QTMD=0/1`
- `PION_TMD_RUN_PDF=0/1`

The qTMD runner currently accepts exactly one sink separation per invocation.
It defaults to `--t_separations 2`; pass a different value directly on the
Python or run-script CLI.  Run separate invocations for different separations.
Production interfaces use the plural `t_separations` name consistently; the
former insertion-time option is not accepted.

GI qTMD production always uses the cached transporter path.  The direct
covariant-shift implementation is test/reference code only.

The output HDF5 attributes record `pos_boost`, `neg_boost`,
`operator_insertion_line=neg_boost`, and
`boost_line_convention=pos_spectator_neg_active`.  Default setup tags encode
both boosts, so unequal-boost runs cannot share a sample log accidentally.
Historical unequal-boost output used one positive source line for both roles
and must be regenerated.  Zero-boost output is numerically unchanged.

## Pion Channel Identity

The setup/smearing tag does not contain an interpolator label.  Output and
resume identities encode the channel explicitly:

```text
C2:        <sm_tag>.src<SRC>
qTMD/PDF:  <sm_tag>.src<SRC>.sink<SINK>.<operator-gamma>
```

The same source, sink, and operator labels are stored as HDF5 attributes.  C2
stores `sink_interpolator=all_16_gamma_scan` because all 16 sink gamma channels
are written into the file.

The production default is the explicit canonical source label `5`, so the
standard pseudoscalar source is written as `src5`.  The shared pion C2
contraction accepts any canonical Gamma label as a fixed source, or
`dagger_of_sink` for a one-to-one paired source/sink scan.  The former
`fixed_g5` and `same_as_sink` modes are not accepted.

For the default source, new HDF5 files record `source_gamma_mode=fixed` and
`source_gamma_label=5`, in addition to `src_interpolator=5`.

## Lightweight Resume

At startup rank 0 reads the existing text sample log once and broadcasts the
exact set of completed source tags.  A completed source is skipped before any
source inversion.  A tag is appended, with `flush` and `fsync`, only after C2
and every enabled CG/GI qTMD/PDF output has closed successfully.  The log is the
only resume state: moved HDF5 files are not checked.

The log name intentionally remains lightweight and does not encode every run
flag or operator-grid bound.  Reuse one log only when the enabled products,
`qmax`, `b_T`, `b_z`, `eta`, momenta, and all other physical grids are unchanged.
Use a new data/setup identity whenever any of those choices changes.  Multiple
jobs must not update the same log concurrently.

The default smoke gauge is:

```text
test_gauge/S8T32_wilson_b6.cg.1e-08.0
```

## GI qTMD Staple Convention

The GI Wilson-line index is:

```text
[b_T, b_z, eta, transverse_direction]
```

The fixed-length staple path is:

```text
x
-> x + (eta + b_z / 2) zhat
-> x + (eta + b_z / 2) zhat + b_T e_perp
-> x + b_z zhat + b_T e_perp
```

The total staple length is `2 * eta + b_T`, independent of `b_z`.  Current
constraints are `b_z` even and `eta >= abs(b_z) / 2`.

## Validation

The GI staple composition order was validated with an independent,
position-dependent, noncommuting SU(3) ordered-product CPU test.  It covers
positive and negative `b_z` and both transverse directions at
`rtol=atol=1e-13`; the former segment order differs from the correct result by
more than `1e-8`.  The geometric path and the reverse `covDev` composition
order are shown in the "Connected GI qTMD Update" figure in
[`docs/pion_qTMD/pion_qTMD.pdf`](../../docs/pion_qTMD/pion_qTMD.pdf).

The repository `test_gauge/S8T8_wilson_b6.0` gauge was also tested on one
Perlmutter node after one production-style 4D HYP step.  Runs used one rank
(`1.1.1.1`) and four ranks (`2.2.1.1`).  For `b_T=1`, `b_z=0,+2,-2`, both
transverse directions, and the straight-PDF limits, cached staples agreed with
an independent direct-`covDev` reference to relative L2 error at most
`3.62e-16`.  Field covariance, cached-link endpoint covariance, and
`xi_dagger Gamma O eta` invariance were all at relative L2 error at most
`4.08e-16`; the link-free CG positive control changed by `0.29--0.96`.
Complete gathered staple fields were bitwise identical between the one- and
four-rank layouts.  On the non-straight paths, the former segment order differed
by relative L2 error `0.438--0.675`.

A minimal connected GI-qTMD run used one source, `tol=1e-15`, `qmax=0`,
`eta=2`, `b_z=2`, `b_T=1`, and `t_sep=3`, with CG/PDF outputs disabled.  The
one- and four-rank runs produced the same 16-file, 192-dataset HDF5 structure;
the maximum dataset relative L2 difference was `1.31e-13`, the maximum absolute
difference was `1.42e-16`, and the largest true residual was `8.54e-16`.

The unequal-boost line correction was validated on S8T8 at solver tolerance
`1e-15` with one rank (`1.1.1.1`) and four ranks (`2.2.1.1`).  Zero-boost
reference/candidate outputs were bitwise identical.  For
`pos=[0,0,1], neg=[0,0,-1]`, the shared helper was bitwise identical to an
independent two-source explicit-line calculation for C2, CG/GI qTMD and CG/GI
PDF.  The largest one/four-rank relative L2 difference was `3.22e-15`.

The optional test script is:

```bash
python tests/test_connected_gi_qtmd_link_cache_consistency.py
```

It expects paired HDF5 outputs under `/tmp/pyquda_connected_gi_qtmd_consistency`.
