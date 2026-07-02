import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np
import pytest

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
    FlowedQuarkRingedNorm,
    _append_sample_log_tag_once,
    _print_block_timers,
    _reset_block_timers,
    compute_ringed_factors,
    flowed_quark_ringed_norm_block_tag,
    flowed_quark_ringed_norm_hp256_sample_log_tag,
    flowed_quark_ringed_norm_sample_seed,
    flow_times,
    hp256_sample_block_ranges,
    hp256_sample_source_range,
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
        self.t_boundary = -1
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
    tag = get_flowed_quark_ringed_norm_file_tag("/data", "lat", 9, "ama", "sm")
    assert tag == "/data/FlowedQuarkRinged/lat.FlowedQuarkRinged.9.ama.sm"


def _ringed_params(hp_num_vectors, block_interval_solves, spin_color_dilution="none"):
    return {
        "flow_type": "symanzik",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": hp_num_vectors,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "spin_color_dilution": spin_color_dilution,
        "block_interval_solves": block_interval_solves,
    }


def test_flowed_quark_ringed_interval_block_size_selection():
    assert natural_estimator_block_size("zn", 1, "none") == 1
    assert natural_estimator_block_size("hierarchical_probing", 16, "none") == 16
    assert natural_estimator_block_size("hierarchical_probing", 256, "none") == 256
    assert natural_estimator_block_size("hierarchical_probing", 16, "point") == 192

    hp256_partial = FlowedQuarkRingedNorm(_ringed_params(256, 64))
    assert hp256_partial.block_size == 64
    assert hp256_partial.natural_block_size == 256

    hp256_complete = FlowedQuarkRingedNorm(_ringed_params(256, 256))
    assert hp256_complete.block_size == 256
    assert hp256_complete.natural_block_size == 256

    hp16_complete = FlowedQuarkRingedNorm(_ringed_params(16, 64))
    assert hp16_complete.block_size == 64
    assert hp16_complete.natural_block_size == 16


def test_flowed_quark_ringed_requires_non_empty_tag_before_pyquda_setup():
    norm = FlowedQuarkRingedNorm(_ringed_params(16, 64))

    with pytest.raises(ValueError, match="non-empty output tag"):
        norm.flowed_kinetic_norm(None, [0.0, 1.0, 1e-10, 10], [1, 2, 7], tag="")


def test_flowed_quark_ringed_requires_interval_divisible_effective_sources():
    norm = FlowedQuarkRingedNorm(_ringed_params(16, 64))

    with pytest.raises(ValueError, match="divisible by block_interval_solves"):
        norm.flowed_kinetic_norm(None, [0.0, 1.0, 1e-10, 10], [1, 2, 7], tag="out")


def test_flowed_quark_ringed_block_tag_suffix_format():
    tag = flowed_quark_ringed_norm_block_tag("base", 3, 768, 1024)
    assert tag == "base.block0003.src000768-001023"


def test_flowed_quark_ringed_hp256_sample_log_helpers(tmp_path):
    seed0 = flowed_quark_ringed_norm_sample_seed(105000, 0)
    seed7 = flowed_quark_ringed_norm_sample_seed(105000, 7)
    assert seed0 != 105000
    assert seed7 != 105007
    assert seed0 != seed7
    assert flowed_quark_ringed_norm_hp256_sample_log_tag(0, seed0) == f"ringed_hp256_base000_seed{seed0}"
    assert flowed_quark_ringed_norm_hp256_sample_log_tag(7, seed7) == f"ringed_hp256_base007_seed{seed7}"
    assert hp256_sample_source_range(2) == (512, 768)
    assert hp256_sample_block_ranges(1, 64) == [
        (4, 256, 320),
        (5, 320, 384),
        (6, 384, 448),
        (7, 448, 512),
    ]

    sample_log_file = tmp_path / "sample_log" / "FlowedQuarkRinged_1HYP_RINGED_HP256_N8_1050"
    tag0 = flowed_quark_ringed_norm_hp256_sample_log_tag(0, seed0)
    assert _append_sample_log_tag_once(sample_log_file, tag0)
    assert not _append_sample_log_tag_once(sample_log_file, tag0)
    assert sample_log_file.read_text().splitlines() == [tag0]


def test_flowed_quark_ringed_hp256_sample_requires_complete_block_files(tmp_path):
    norm = FlowedQuarkRingedNorm({
        **_ringed_params(256, 64),
        "sample_log_file": str(tmp_path / "sample_log" / "FlowedQuarkRinged_1HYP_RINGED_HP256_N8_1050"),
    })
    base_tag = str(tmp_path / "FlowedQuarkRinged" / "schema")

    seed0 = flowed_quark_ringed_norm_sample_seed(105000, 0)
    assert norm._sample_log_tag(0, 105000) == f"ringed_hp256_base000_seed{seed0}"
    assert not norm._sample_block_files_exist(base_tag, 0)

    for block_index, block_start, block_stop in hp256_sample_block_ranges(0, 64):
        block_file = Path(flowed_quark_ringed_norm_block_tag(base_tag, block_index, block_start, block_stop) + ".h5")
        block_file.parent.mkdir(parents=True, exist_ok=True)
        block_file.touch()

    assert norm._sample_block_files_exist(base_tag, 0)


def test_flowed_quark_ringed_sample_log_rejects_non_hp256_mode(tmp_path):
    with pytest.raises(ValueError, match="only HP256"):
        FlowedQuarkRingedNorm({
            **_ringed_params(16, 64),
            "sample_log_file": str(tmp_path / "sample_log" / "bad"),
        })


def test_flowed_quark_ringed_base_range_selection(tmp_path):
    norm = FlowedQuarkRingedNorm({
        **_ringed_params(256, 64),
        "sample_log_file": str(tmp_path / "sample_log" / "FlowedQuarkRinged_1HYP_RINGED_HP256_N8_1050"),
        "base_start": 3,
        "base_stop": 4,
    })

    assert norm._selected_base_indices(8) == {3}


def test_flowed_quark_ringed_base_range_rejects_invalid_range(tmp_path):
    norm = FlowedQuarkRingedNorm({
        **_ringed_params(256, 64),
        "sample_log_file": str(tmp_path / "sample_log" / "FlowedQuarkRinged_1HYP_RINGED_HP256_N8_1050"),
        "base_start": 4,
        "base_stop": 4,
    })

    with pytest.raises(ValueError, match="invalid base range"):
        norm._selected_base_indices(8)


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
    norm = FlowedQuarkRingedNorm(_ringed_params(256, 64))
    kinetic_pervec = np.ones((64, 3, 4), dtype=np.complex128)
    times = flow_times(0.1, 2)
    attrs = {
        "measurement": "flowed_quark_ringed_norm",
        "effective_n_inversions": 256,
        "natural_block_size": norm.natural_block_size,
        "block_interval_solves": norm.block_size,
    }
    source_bookkeeping = {
        "source_index": np.arange(64),
        "base_noise_index": np.zeros(64, dtype=np.int32),
        "hp_index": np.arange(64),
        "spin_index": -np.ones(64, dtype=np.int32),
        "color_index": -np.ones(64, dtype=np.int32),
    }

    norm._write_block_file(
        base_tag,
        kinetic_pervec,
        times,
        attrs,
        source_bookkeeping,
        0,
        0,
        64,
    )

    tag = flowed_quark_ringed_norm_block_tag(base_tag, 0, 0, 64)
    with h5py.File(tag + ".h5", "r") as h5:
        removed_attrs = {
            "block_output",
            "block_output_policy",
            "block_size",
            "block_source_count",
            "configured_block_size",
            "effective_n_inversions_total",
        }
        assert removed_attrs.isdisjoint(h5.attrs.keys())
        assert h5.attrs["block_index"] == 0
        assert h5.attrs["block_start"] == 0
        assert h5.attrs["block_stop_exclusive"] == 64
        assert h5.attrs["effective_n_inversions"] == 256
        assert h5.attrs["natural_block_size"] == 256
        assert h5.attrs["block_interval_solves"] == 64
        assert not h5.attrs["estimator_complete"]
        assert h5.attrs["complete_estimator_units"] == 0
        assert h5.attrs["estimator_remainder"] == 64
        assert h5["raw/kinetic_pervec"].shape == (64, 3, 4)
        np.testing.assert_array_equal(h5["raw/source_index"][...], np.arange(64))
        np.testing.assert_allclose(h5["avg/Z_ring_field_sqrt"][1:] ** 2, h5["avg/Z_ring_bilinear"][1:])


def test_flowed_quark_ringed_block_estimator_complete_attrs(tmp_path):
    base_tag = str(tmp_path / "FlowedQuarkRinged" / "schema")
    norm = FlowedQuarkRingedNorm(_ringed_params(16, 64))
    kinetic_pervec = np.ones((64, 2, 4), dtype=np.complex128)
    source_bookkeeping = {
        "source_index": np.arange(64),
        "base_noise_index": np.zeros(64, dtype=np.int32),
        "hp_index": np.arange(64) % 16,
        "spin_index": -np.ones(64, dtype=np.int32),
        "color_index": -np.ones(64, dtype=np.int32),
    }

    norm._write_block_file(
        base_tag,
        kinetic_pervec,
        flow_times(0.1, 1),
        {"measurement": "flowed_quark_ringed_norm"},
        source_bookkeeping,
        0,
        0,
        64,
    )

    tag = flowed_quark_ringed_norm_block_tag(base_tag, 0, 0, 64)
    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["estimator_complete"]
        assert h5.attrs["complete_estimator_units"] == 4
        assert h5.attrs["estimator_remainder"] == 0


class _FakeRankInfo:
    mpi_rank = 0


def test_flowed_quark_ringed_timer_summary_prints_expected_labels(capsys, monkeypatch):
    monkeypatch.delenv("PYQUDA_MEASUREMENT_TIMERS", raising=False)
    timers = _reset_block_timers(2, 1)
    timers["invert"] = 4.0
    timers["contract"][0] = 1.0
    timers["contract"][1] = 2.0
    timers["flow"][0] = 3.0
    timers["write"] = 0.5

    _print_block_timers(_FakeRankInfo(), 0, 0, 4, timers, 10.0)

    out = capsys.readouterr().out
    assert "TIMER ringed_norm_block" in out
    assert "TIMER ringed_norm_invert" in out
    assert "TIMER ringed_norm_contract" in out
    assert "step=0" in out
    assert "step=1" in out
    assert "TIMER ringed_norm_flow" in out
    assert "step=0_to_1" in out
    assert "TIMER ringed_norm_block_write" in out
    assert "per_source=2.500000" in out


def test_flowed_quark_ringed_timer_summary_can_be_disabled(capsys, monkeypatch):
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "0")
    timers = _reset_block_timers(2, 1)

    _print_block_timers(_FakeRankInfo(), 0, 0, 4, timers, 10.0)

    assert capsys.readouterr().out == ""
