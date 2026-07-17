# Pion Current-Current Response Diagnostic

This is a minimal GPU diagnostic for the nested local-current response

```text
S_12 = D^{-1} O_2 D^{-1} O_1 S
```

It contracts `S_12` with the ordinary pion antiquark line using the pion
two-point contraction helper.  The default smoke run uses the S8T32 test gauge,
`gamma_1 = gamma_2 = T`, `q_1 = [0,0,1]`, `q_2 = [0,0,-1]`, and `tsep = 2`.

Run on `login32` or another GPU node:

```bash
bash application/pion_current_current_response/perlmutter/run_pion_current_current_response.sh
```

Use `--src_pos X.Y.Z.T` for a nonzero source position.  Both insertion windows
are defined relative to that same source time.  Each relative insertion time is
converted only when its projector is built,

```text
tau_abs = (source_time + tau_rel) mod Nt,
```

and the saved C2 and nested-response correlators are rolled to a
source-relative time axis.  The version-2 HDF5 schema stores the source
position/time and both relative and absolute tau lists.

The HDF5 output is written under `data/current_current_response/` and includes
a compact `summary/` group for analysis.
