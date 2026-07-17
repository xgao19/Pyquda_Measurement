# Analysis helpers

This directory contains shared post-processing code used by application
workflows, including measurement-data readers, observable assembly, time-axis
alignment, and memory-bounded analysis transforms.

Production measurements, operator contractions, and reusable computational
infrastructure belong in `pyquda_measurement_utils`.  Helpers in this directory
must not be imported by production measurement kernels.

`emt_ringed_stochastic_comparison.py` groups EMT-derived ringed kinetic data by
complete randomized base, constructs fixed-solve cumulative statistics, and
generates reproducible headless comparison plots.  Partial hierarchical-
probing prefixes are deliberately excluded from its uncertainty estimates.

`emt_quark_1pt_convergence.py` is the focused quark-loop benchmark entry point. It
accepts any labeled set of finalized EMTc files, reads only the primitive
channels needed for a requested `Tmunu`, and compares that component with the
embedded ringed kinetic using complete-base cumulative means and SEM.  It can
be run directly; the guided commands are in
[`application/EMT_disconnected_1pt/README.md`](../EMT_disconnected_1pt/README.md).

`emt_proton_t44_analysis.py` supplies the matching proton `PpUnpol` two-point
projector, optimized nonzero-momentum ratio, source-relative loop rephasing,
complete-base T44 reader, and fixed-gauge source/base resampling used by the
S8T8 connected/disconnected diagnostic.  Its source-translation covariance is
not a substitute for a gauge-ensemble vacuum subtraction.

The common 16-Gamma convention and portable examples for rebuilding `Tmunu`,
axial one-derivative operators, and local tensor currents from primitive HDF5
datasets are in
[`docs/EMT_gamma_and_raw_bilinears.md`](../../docs/EMT_gamma_and_raw_bilinears.md).

`pion_current_response_analysis.py` contains source-relative tau-window
selection, rolling, ratio/channel extraction, explicit-EMFF summation, and
the response HDF5 writers. The corresponding module under
`pyquda_measurement_utils` is calculation-only.
