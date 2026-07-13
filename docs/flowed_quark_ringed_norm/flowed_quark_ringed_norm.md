# Standalone flowed-quark ringed normalization

The standalone workflow measures

```text
K(t_f) = <bar chi(t_f) overleftrightarrow{Dslash} chi(t_f)>
```

with full-volume counter-based `Z4` sources. It is useful for dedicated
high-statistics studies and optional point spin-color dilution. `config_num` is
required; the stream salt is stored as `noise_stream`.

## Checkpoint model

Standalone ringed now uses exactly the same base/HP-part infrastructure as EMT
and disconnected qTMD. A part contains a contiguous HP interval from one base,
raw `kinetic_pervec`, and global source/base/HP/spin/color bookkeeping. Rank 0
writes atomically. Only after all parts close does it append one exact base line
to a fingerprinted text log under `sample_log_disconnected/`. Resume reads only
this log and does not require transferred HDF5 files to remain locally. There
is no JSON completion marker or `.block*.h5` compatibility path.

Point spin-color dilution costs 12 solves per HP pattern. A part boundary never
splits those 12 projectors. The finalized trace multiplies the raw channel
average by `spin_color_trace_factor=12`.

## Finalization and ensemble analysis

Measurement jobs write shards only. Transfer the parts, then publish one
configuration at the destination after all bases are available:

```bash
python application/flowed_quark_ringed_norm/finalize_ringed_shards.py \
  --shard-dir <dir> --canonical-tag <tag-without-.h5> --n-base-noise <N>
```

The canonical file contains `flow_times`, `raw/kinetic_pervec`, source
bookkeeping, and `avg/kinetic_spacetime`. It deliberately contains no ringed
factors and records `ringed_factors_stored=False`.

The nonlinear factor must be formed after an equal-weight configuration
average:

```text
Z_bilinear(t_f) = -2 Nc / ((4 pi)^2 t_f^2 <K(t_f)>_cfg)
Z_field_sqrt(t_f) = sqrt(Z_bilinear(t_f))
```

Use explicit inputs and output:

```bash
python application/flowed_quark_ringed_norm/analyze_ringed_ensemble.py \
  --input cfg1.h5 --input cfg2.h5 --output ringed_ensemble.h5
```

Never average configuration-local `1/K`. The `t_f=0` factor is undefined and
is stored as NaN in the ensemble result.
