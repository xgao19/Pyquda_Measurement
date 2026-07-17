# Connected Pion TMD Application

This directory contains the connected pion TMD workflow with optional CG qTMD,
GI qTMD, and PDF/local-limit contractions.

This is the canonical connected pion entry point for CG qTMD, GI qTMD, CG PDF,
and GI PDF.  The former duplicate CG-only application has been removed.  The GI
qTMD path uses the fixed-length staple convention shared with the disconnected
qTMD code.

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

The connected pion GI qTMD workflow has passed S8T32 smoke tests on Perlmutter
`login32`.  A nonzero-staple consistency test with `b_z=2`, `b_T=1`, `eta=1`,
and `qmax=0` verified that the link cache agrees with the direct test reference
to roundoff.

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
