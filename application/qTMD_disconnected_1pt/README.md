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
```

It does not yet implement a fully gauge-invariant staple qTMD Wilson line.

## Perlmutter Smoke Test

```bash
cd /global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/application/qTMD_disconnected_1pt/perlmutter
bash run_qTMD_1pt.sh
```

Useful environment variables:

```text
QTMD_1PT_OPERATOR_KIND=GI_PDF
QTMD_1PT_QMAX=0
QTMD_1PT_BZ=0
QTMD_1PT_BT=0
QTMD_1PT_N_VEC=1
QTMD_1PT_NOISE_SCHEME=zn
QTMD_1PT_HP_NUM_VECTORS=1
QTMD_1PT_HP_ORDERING=global_xyzt_gray_projected_to_evenodd
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
