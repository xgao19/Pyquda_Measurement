import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for flowed-quark ringed schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    effective_n_inversions,
    hierarchical_probe_pattern,
    validate_hierarchical_probing_options,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import (
    complete_block_size_ge_min,
    compute_ringed_factors,
    flowed_quark_ringed_norm_block_tag,
    flow_times,
    kinetic_spacetime_from_raw,
    natural_estimator_block_size,
)
from pyquda_measurement_utils.io_corr import (
    get_flowed_quark_ringed_norm_file_tag,
    save_flowed_quark_ringed_norm_hdf5,
)


class _FakeLatticeInfo:
    def __init__(self, global_size):
        self.global_size = list(global_size)
        self._coords = np.indices(tuple(global_size), dtype=np.int64)

    def coordinate(self, mu=None):
        if mu is None:
            return [self._coords[idx] for idx in range(4)]
        return self._coords[mu]


def test_ringed_factor_formula_sets_flow0_nan_and_squares_to_bilinear():
    times = np.asarray([0.0, 0.1, 0.2])
    kinetic = np.asarray([-3.0, -2.0, -1.0], dtype=np.complex128)

    z_field, z_bilinear = compute_ringed_factors(kinetic, times, nc=3)

    assert np.isnan(z_bilinear[0].real)
    assert np.isnan(z_field[0].real)
    np.testing.assert_allclose(z_field[1:] ** 2, z_bilinear[1:])
    expected = -6.0 / (((4.0 * np.pi) ** 2) * times[1:] ** 2 * kinetic[1:])
    np.testing.assert_allclose(z_bilinear[1:], expected)


def test_interleaved_hp_first_shell_cancels_nearest_neighbors_on_8_to_4():
    latt_info = _FakeLatticeInfo([8, 8, 8, 8])
    patterns = np.asarray(
        [
            hierarchical_probe_pattern(latt_info, hp_idx, "interleaved_xyzt_binary_projected_to_evenodd")
            for hp_idx in range(16)
        ]
    )

    for axis in range(4):
        neighbor_product = patterns * np.roll(patterns, shift=-1, axis=axis + 1)
        np.testing.assert_allclose(np.mean(neighbor_product, axis=0), 0.0)


def test_hp_inversion_count_and_power_of_two_validation():
    assert effective_n_inversions(64, "hierarchical_probing", 16) == 1024
    validate_hierarchical_probing_options(16, "interleaved_xyzt_binary_projected_to_evenodd")
    try:
        validate_hierarchical_probing_options(12, "interleaved_xyzt_binary_projected_to_evenodd")
    except ValueError:
        pass
    else:
        raise AssertionError("non-power-of-two HP vector count should fail")


def test_spin_color_trace_factor_scales_raw_average():
    raw = np.ones((2, 3, 4), dtype=np.complex128)

    np.testing.assert_allclose(kinetic_spacetime_from_raw(raw, 1), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(kinetic_spacetime_from_raw(raw, 12), [12.0, 12.0, 12.0])


def test_flowed_quark_ringed_tag_helper_uses_generic_directory():
    tag = get_flowed_quark_ringed_norm_file_tag("/data", "lat", 9, "ama", [1, 2, 3, 4], "sm")
    assert tag == "/data/FlowedQuarkRinged/lat.FlowedQuarkRinged.9.ama.x1y2z3t4.sm"


def test_flowed_quark_ringed_block_size_selection():
    assert natural_estimator_block_size("zn", 1, "none") == 1
    assert complete_block_size_ge_min(natural_estimator_block_size("zn", 1, "none"), 256) == 256
    assert complete_block_size_ge_min(natural_estimator_block_size("hierarchical_probing", 16, "none"), 256) == 256
    assert complete_block_size_ge_min(natural_estimator_block_size("hierarchical_probing", 256, "none"), 256) == 256
    assert complete_block_size_ge_min(natural_estimator_block_size("hierarchical_probing", 16, "point"), 256) == 384


def test_flowed_quark_ringed_block_tag_suffix_format():
    tag = flowed_quark_ringed_norm_block_tag("base", 3, 768, 1024)
    assert tag == "base.block0003.src000768-001023"


def test_flowed_quark_ringed_hdf5_schema(tmp_path):
    tag = str(tmp_path / "FlowedQuarkRinged" / "schema")
    kinetic_pervec = np.ones((2, 3, 4), dtype=np.complex128)
    kinetic_spacetime = np.mean(kinetic_pervec, axis=(0, -1))
    times = flow_times(0.1, 2)
    z_field, z_bilinear = compute_ringed_factors(-np.ones_like(kinetic_spacetime), times)
    attrs = {
        "measurement": "flowed_quark_ringed_norm",
        "normalization_scope": "all_flowed_quark_fields",
        "operator": "bar_chi_overleftrightarrow_Dslash_chi",
        "flow0_factor": np.nan,
        "spin_color_dilution": "point",
        "spin_color_dilution_factor": 12,
        "spin_color_trace_factor": 12,
        "site_noise_scope": "site_only",
    }
    source_bookkeeping = {
        "source_index": [0, 1],
        "base_noise_index": [0, 1],
        "hp_index": [0, 0],
        "spin_index": [0, 0],
        "color_index": [0, 1],
    }

    save_flowed_quark_ringed_norm_hdf5(
        tag,
        kinetic_pervec,
        kinetic_spacetime,
        z_field,
        z_bilinear,
        times,
        attrs=attrs,
        source_bookkeeping=source_bookkeeping,
    )

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["measurement"] == "flowed_quark_ringed_norm"
        assert h5.attrs["normalization_scope"] == "all_flowed_quark_fields"
        assert h5.attrs["spin_color_dilution"] == "point"
        assert h5.attrs["spin_color_dilution_factor"] == 12
        assert h5.attrs["spin_color_trace_factor"] == 12
        assert h5.attrs["site_noise_scope"] == "site_only"
        assert np.isnan(h5.attrs["flow0_factor"])
        np.testing.assert_allclose(h5["flow_times"][...], times)
        assert h5["raw/kinetic_pervec"].shape == (2, 3, 4)
        assert h5["raw/source_index"].shape == (2,)
        np.testing.assert_array_equal(h5["raw/spin_index"][...], [0, 0])
        np.testing.assert_array_equal(h5["raw/color_index"][...], [0, 1])
        assert "kinetic_timeslice" not in h5["avg"]
        assert h5["avg/kinetic_spacetime"].shape == (3,)
        assert h5["avg/Z_ring_field_sqrt"].shape == (3,)
        assert h5["avg/Z_ring_bilinear"].shape == (3,)
        np.testing.assert_allclose(h5["avg/Z_ring_field_sqrt"][1:] ** 2, h5["avg/Z_ring_bilinear"][1:])


def test_flowed_quark_ringed_block_hdf5_schema(tmp_path):
    base_tag = str(tmp_path / "FlowedQuarkRinged" / "schema")
    tag = flowed_quark_ringed_norm_block_tag(base_tag, 0, 0, 4)
    kinetic_pervec = np.ones((4, 3, 4), dtype=np.complex128)
    kinetic_spacetime = np.mean(kinetic_pervec, axis=(0, -1))
    times = flow_times(0.1, 2)
    z_field, z_bilinear = compute_ringed_factors(-np.ones_like(kinetic_spacetime), times)
    attrs = {
        "measurement": "flowed_quark_ringed_norm",
        "block_output": True,
        "block_index": 0,
        "block_start": 0,
        "block_stop_exclusive": 4,
        "block_size": 4,
        "block_source_count": 4,
        "effective_n_inversions_total": 8,
        "block_complete_policy": "smallest_complete_estimator_block_ge_min_solves",
        "natural_block_size": 2,
        "monolithic_full_output": False,
    }
    source_bookkeeping = {
        "source_index": [0, 1, 2, 3],
        "base_noise_index": [0, 0, 1, 1],
        "hp_index": [0, 1, 0, 1],
        "spin_index": [-1, -1, -1, -1],
        "color_index": [-1, -1, -1, -1],
    }

    save_flowed_quark_ringed_norm_hdf5(
        tag,
        kinetic_pervec,
        kinetic_spacetime,
        z_field,
        z_bilinear,
        times,
        attrs=attrs,
        source_bookkeeping=source_bookkeeping,
    )

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["block_output"]
        assert h5.attrs["block_index"] == 0
        assert h5.attrs["block_start"] == 0
        assert h5.attrs["block_stop_exclusive"] == 4
        assert h5.attrs["block_size"] == 4
        assert h5.attrs["block_source_count"] == 4
        assert h5.attrs["effective_n_inversions_total"] == 8
        assert h5.attrs["natural_block_size"] == 2
        assert not h5.attrs["monolithic_full_output"]
        assert h5["raw/kinetic_pervec"].shape == (4, 3, 4)
        np.testing.assert_array_equal(h5["raw/source_index"][...], [0, 1, 2, 3])
        np.testing.assert_allclose(h5["avg/Z_ring_field_sqrt"][1:] ** 2, h5["avg/Z_ring_bilinear"][1:])
