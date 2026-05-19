# qTMD Disconnected One-Point Workflow

This application measures hadron-independent stochastic quark loops for qTMD
and PDF-style disconnected diagrams.  The output is intended to be combined
with pion or proton two-point functions in downstream analysis:

```text
C3_disc = < C2_H L_qTMD > - < C2_H > < L_qTMD >
```

The first implementation supports:

```text
GI_PDF
CG_PDF
CG_qTMD
GI_qTMD
```

For `GI_qTMD`, `b_z` is the physical final z separation and must be even in
the fixed-staple-length convention.  The staple legs are
`eta + b_z / 2` and `eta - b_z / 2`, so use `eta >= abs(b_z) / 2`.
The default `QTMD_1PT_GI_STAPLE_MODE=link_cache` builds gauge-only staple
transporters once per gauge configuration and reuses them for all stochastic
sources.  Use `QTMD_1PT_GI_STAPLE_MODE=direct_covdev` only as a reference
debug mode.

The link-cache path has been checked against direct `covDev` on the S8T32 test
gauge, including a nonzero transverse staple case `bT=1, bz=2, eta=1`.

## Perlmutter Smoke Test

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/qTMD_disconnected_1pt/perlmutter
bash run_qTMD_1pt.sh
```

Useful environment variables:

```text
QTMD_1PT_OPERATOR_KIND=GI_PDF
QTMD_1PT_QMAX=0
QTMD_1PT_ETA=0
QTMD_1PT_BZ=0
QTMD_1PT_BT=0
QTMD_1PT_N_VEC=1
QTMD_1PT_NOISE_SCHEME=zn
QTMD_1PT_HP_NUM_VECTORS=1
QTMD_1PT_HP_ORDERING=global_xyzt_gray_projected_to_evenodd
QTMD_1PT_GI_STAPLE_MODE=link_cache
```

For hierarchical probing:

```text
QTMD_1PT_NOISE_SCHEME=hierarchical_probing
QTMD_1PT_HP_NUM_VECTORS=2
```

## HDF5 Layout

The writer stores:

```text
attrs/
  measurement = disconnected_qTMD_1pt
  operator_kind
  qext
  W_index_list
  gamma_list
  volume_norm
  noise_scheme
  n_base_noise
  hp_num_vectors
  effective_n_inversions
  hp_ordering
  gi_qtmd_staple_mode
  loop_convention = eta_dagger_Gamma_O_b_xi

raw/loop_pervec
raw/source_index
raw/base_noise_index
raw/hp_index

avg/SS/<gamma>/PX...PY...PZ.../b_X_or_b_Y/eta0/bT.../bz...
```

`raw/loop_pervec` has shape:

```text
[effective_source, Wilson_index, gamma, momentum, time]
```

The averaged datasets are divided by the spatial volume.

## Sanity Checks

Useful lightweight checks are kept under `tests/` and read existing smoke
outputs when available:

```text
test_qtmd_disconnected_local_pdf_limit.py
test_qtmd_disconnected_nonzero_bz.py
test_qtmd_disconnected_gi_staple_pdf_limit.py
test_qtmd_disconnected_gi_staple_link_cache_hdf5.py
test_disconnected_gi_qtmd_link_cache.py
```

The GI staple checks verify the local/PDF limits, cached-link equality with the
direct `covDev` reference path, and the nonzero-transverse-staple HDF5 output.
