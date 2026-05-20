# Connected Pion TMD Application

This directory contains the connected pion TMD workflow with optional CG qTMD,
GI qTMD, and PDF/local-limit contractions.

## Relation To The Older CG Workflow

- `application/pion_TMD_CG/perlmutter` is the older CG-only workflow.
- `application/pion_TMD/perlmutter` keeps that connected CG path available and
  adds connected GI qTMD support.
- The GI qTMD path uses the fixed-length staple convention shared with the
  disconnected qTMD code.

## Perlmutter Entry Point

Run the smoke/production wrapper from the repository root or from this
directory:

```bash
bash application/pion_TMD/perlmutter/run_pion_TMD.sh
```

Useful runtime switches:

- `PION_TMD_RUN_CG_QTMD=0/1`
- `PION_TMD_RUN_GI_QTMD=0/1`
- `PION_TMD_RUN_PDF=0/1`
- `PION_TMD_GI_STAPLE_MODE=link_cache` for the cached transporter path
- `PION_TMD_GI_STAPLE_MODE=direct_covdev` for the direct covariant-shift path

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
and `qmax=0` verifies that `link_cache` and `direct_covdev` agree to roundoff.

The optional test script is:

```bash
python tests/test_connected_gi_qtmd_link_cache_consistency.py
```

It expects paired HDF5 outputs under `/tmp/pyquda_connected_gi_qtmd_consistency`.
