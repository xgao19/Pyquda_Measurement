# Analysis helpers

This directory contains shared post-processing code used by application
workflows, including measurement-data readers, observable assembly, time-axis
alignment, and memory-bounded analysis transforms.

Production measurements, operator contractions, and reusable computational
infrastructure belong in `pyquda_measurement_utils`.  Helpers in this directory
must not be imported by production measurement kernels.

