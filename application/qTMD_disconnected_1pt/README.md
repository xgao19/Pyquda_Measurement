# qTMD Disconnected One-Point Workflow

This application measures source-independent quark loops for `GI_PDF`,
`CG_PDF`, `CG_qTMD`, and `GI_qTMD`, to be combined with hadron two-point
functions through

```text
C3_disc = <C2_H L_qTMD> - <C2_H><L_qTMD>.
```

Production has one path: decomposition-independent counter-based `Z4`,
base/HP-part shards, and explicit finalization. `config_num` is mandatory;
`QTMD_1PT_RAND_SEED` is a stream salt and is recorded only as `noise_stream`.
There is no backend RNG, source-position tag, or monolithic output mode.

For `GI_qTMD`, the fixed-length staple requires even `b_z` and
`eta >= abs(b_z)/2`. Gauge-only staple links are built once per configuration
and reused for every stochastic source. The slow direct-covDev implementation
exists only under `tests/` as a numerical reference.

Run a measurement and then finalize complete bases:

```bash
cd application/qTMD_disconnected_1pt/perlmutter
bash run_qTMD_1pt.sh --config_num 1000
bash run_finalize_qTMD_1pt.sh --config_num 1000
```

Configuration identity is a required CLI argument.  The Python and shell
entries do not read `QTMD_1PT_CONFIG_NUM` and never default it to zero.

Main production controls are:

```text
QTMD_1PT_OPERATOR_KIND=GI_PDF
QTMD_1PT_N_VEC=1
QTMD_1PT_N_ZN=4
QTMD_1PT_RAND_SEED=0
QTMD_1PT_NOISE_SCHEME=zn
QTMD_1PT_HP_NUM_VECTORS=1
QTMD_1PT_HP_ORDERING=global_xyzt_gray_projected_to_evenodd
QTMD_1PT_BASE_START=0
QTMD_1PT_BASE_STOP=QTMD_1PT_N_VEC
QTMD_1PT_BLOCK_INTERVAL_SOLVES=64
QTMD_1PT_SHARD_DIR=<data>/qTMD1pt/shards
```

Parts are named
`<stem>.baseXXXXXX.partXXXX.hpSTART-END.h5`. Rank 0 writes each part through an
atomic rename. Resume validates metadata, shapes, momentum/Wilson/gamma lists,
and exact source/base/HP indices. A completion marker is published only after
all parts of a base pass validation. Different jobs may own non-overlapping
base ranges; overlapping ranges are unsupported.

The finalizer uses the shared disconnected shard validator and publishes only
after bases `0 ... N_VEC-1` are complete. The canonical file is

```text
qTMD1pt/<lat>.qTMD1pt.<cfg>.<ama>.<sm>.h5
```

with raw layout

```text
raw/loop_pervec  [source, Wilson_index, gamma, momentum, time]
raw/source_index
raw/base_noise_index
raw/hp_index
```

and the existing `avg/SS/...` hierarchy divided by the spatial volume.
