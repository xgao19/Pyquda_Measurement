# Flowed topological charge loop

This gauge-only measurement produces the finite-flow building block for a
disconnected nucleon insertion. It does not compute propagators, C2, C3, a
polarized ratio, or an RGI-extrapolated form factor.

Edit the constants near the top of `Aurora/Pyquda_topological_charge_1pt.py`,
then submit `qsub Aurora/submit_topological_charge_1pt.pbs` from a login node.
The example requests 10 Aurora nodes, 100 ranks and one hour. Do not run its
Python program directly on a login node. A matching allocation is required.
No Perlmutter example or validation is claimed.

The example uses Coulomb-fixed a050 stream b configuration 600 with **no extra
HYP**. Both Wilson and Symanzik flow are supported. `flow_epsilon` is the output
interval, not the integrator step: all intervals use the same number of
substeps, unlike the historical EMT first-interval convention. Default times
are `0,.1,.2,.3,.4` in units of `a^2`, with integration step `.01`.
Optional `flow_first_interval_substeps` overrides only the first interval;
omitting it preserves the uniform schedule above. Metadata records both the
first and subsequent integration step sizes. These are numerical test choices,
not an established physical flow window.

Output is one HDF5 file per configuration and setup under
`data/topologicalChargeLoop/<stream>.<configuration>/`:

| Dataset | Shape | Meaning |
|---|---|---|
| `loop` | `[Nq,Nflow,Nt]` complex128 | Spatial average; absolute time; positive Fourier sign |
| `momentum_list` | `[Nq,4]` int32 | Integer spatial modes, fourth entry zero |
| `flow_times_lattice` | `[Nflow]` float64 | `tf/a^2` |
| `diagnostics/Q_total` | `[Nflow]` float64 | `Vs * sum_tau loop(q=0)` |

Include exactly one zero momentum and no repeated modes. Only rank zero
returns the result dictionary and writes HDF5; other ranks return `None`.
The input gauge links remain unchanged, but resident QUDA gauge state may
change. A caller continuing fermion work must restore the appropriate gauge.
The application records identity, preprocessing and integration parameters;
set `CODE_VERSION` to the actual code revision and preserve any dirty diff.
The setup string must distinguish all changed measurement settings. Reusing a
tag overwrites that file; there is no implicit resume or sample log.

No ensemble mean is subtracted during measurement. Pair loops with matching
configuration/stream, rephase and time-align each hadron source, then form the
ensemble covariance and restore exactly one factor of `Vs`. Existing Unpol
C2 alone is not generally sufficient for this pseudoscalar form factor.
Use `source_relative_loops(..., q_axis=0)` from the analysis helper for the
array operation, not its EMT-specific file readers or strict schema validator.

See [the mathematical conventions and derivations](../../docs/topological_charge/README.md)
and [validation status](../../docs/topological_charge/validation_summary.md).
