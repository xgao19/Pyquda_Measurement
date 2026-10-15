"""Validate the finalized complete-base format shared by EMT analyses."""

import numpy as np


def validate_base_file(h5):
    """Return independent base indices and the derived solve cost per base."""
    if h5.attrs.get("output_kind") != "emt_complete_base_statistics_v2":
        raise ValueError("expected a new EMT .bases.h5 file; use the original checkout for legacy data")
    if h5.attrs.get("shard_schema") != "disconnected_1pt_base_parts_v4":
        raise ValueError("incompatible disconnected shard schema")
    if int(h5.attrs["emt_operator_schema_version"]) != 5:
        raise ValueError("expected EMT operator schema 5")
    if not h5.attrs["base_values_volume_normalized"]:
        raise ValueError("base observables must already be volume normalized")
    bases = np.asarray(h5["base/base_noise_index"], dtype=np.int64)
    if bases.ndim != 1 or len(bases) == 0 or not np.array_equal(bases, np.arange(len(bases))):
        raise ValueError("base indices must be nonempty and contiguous from zero")
    if int(h5.attrs["n_base_noise"]) != len(bases):
        raise ValueError("base count does not match metadata")
    mode = h5.attrs["spin_color_dilution"]
    if mode not in ("none", "point"):
        raise ValueError("invalid spin-color dilution")
    count = int(h5.attrs["vectors_per_base"])
    cost = count * (12 if mode == "point" else 1)
    if count <= 0 or int(h5.attrs["effective_n_inversions"]) != len(bases) * cost:
        raise ValueError("inconsistent complete-base solve count")
    return bases, cost
