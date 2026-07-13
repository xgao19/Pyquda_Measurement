from unittest import SkipTest

import numpy as np
import pytest

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for flowed-quark ringed tests") from err

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    part_source_bookkeeping,
)
from pyquda_measurement_utils.disconnected_shards import (
    SHARD_SCHEMA,
    base_completion_path,
    base_part_ranges,
    completion_payload,
    expected_part_attrs,
    shard_part_path,
    write_base_completion_marker,
    write_raw_part_hdf5,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import (
    FlowedQuarkRingedNorm,
    analyze_ringed_ensemble,
    compute_ringed_factors,
    finalize_flowed_quark_ringed_norm_shards,
    flow_times,
    kinetic_spacetime_from_raw,
    natural_estimator_block_size,
)


def test_ringed_factor_formula_is_applied_after_kinetic_average():
    times = np.asarray([0.0, 0.1, 0.2])
    kinetic = np.asarray([-3.0, -2.0, -1.0], dtype=np.complex128)
    z_field, z_bilinear = compute_ringed_factors(kinetic, times, nc=3)
    assert np.isnan(z_bilinear[0].real)
    np.testing.assert_allclose(z_field[1:] ** 2, z_bilinear[1:])
    np.testing.assert_allclose(
        z_bilinear[1:],
        -6.0 / (((4.0 * np.pi) ** 2) * times[1:] ** 2 * kinetic[1:]),
    )


def test_ringed_part_layout_never_splits_spin_color_projectors():
    assert base_part_ranges(16, 64, solves_per_hp=12) == [
        (0, 0, 5), (1, 5, 10), (2, 10, 15), (3, 15, 16)
    ]
    with pytest.raises(ValueError, match="cannot hold one complete HP vector"):
        base_part_ranges(16, 8, solves_per_hp=12)
    assert natural_estimator_block_size("hierarchical_probing", 16, "point") == 192


def test_ringed_requires_explicit_configuration():
    with pytest.raises(ValueError, match="config_num is required"):
        FlowedQuarkRingedNorm({
            "flow_type": "wilson", "flow_epsilon": 0.1, "flow_steps": 1,
        })


def _ringed_attrs(config_num):
    return {
        "measurement": "flowed_quark_ringed_norm",
        "output_kind": "flowed_quark_ringed_norm",
        "shard_schema": SHARD_SCHEMA,
        "block_interval_solves": 64,
        "content": "kinetic_only",
        "producer": "standalone_ringed",
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 2,
        "flow_times": flow_times(0.1, 2),
        "mass": 0.1,
        "csw": 1.0,
        "tol": 1e-10,
        "maxiter": 100,
        "gauge_preprocessing": "test",
        "t_boundary": -1,
        "flavor_convention": "single_flavor_trace_for_this_dirac_operator",
        "derivative_convention": "gamma_mu*(Dplus_mu-Dminus_mu)",
        "Nc": 3,
        "n_zn": 4,
        "config_num": config_num,
        "noise_stream": 7,
        "noise_generator": "splitmix64_global_coordinate_v1",
        "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 2,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "spin_color_dilution": "none",
        "spin_color_dilution_factor": 1,
        "spin_color_trace_factor": 1,
        "site_noise_scope": "site_spin_color",
        "volume_norm": 8,
        "ringed_factors_stored": False,
    }


def _write_ringed_base(shard_dir, tag, config_num, base_idx):
    path = shard_part_path(shard_dir, tag, base_idx, 0, 0, 2)
    attrs = expected_part_attrs(_ringed_attrs(config_num), base_idx, 0, 0, 2, 2)
    attrs["configured_n_base_noise"] = 1
    bookkeeping = part_source_bookkeeping(
        base_idx, 0, 2, 2, include_spin_color=True
    )
    kinetic = np.full((2, 3, 4), -(config_num + 1), dtype=np.complex128)
    write_raw_part_hdf5(
        path, {"kinetic_pervec": kinetic}, attrs, bookkeeping
    )
    write_base_completion_marker(
        base_completion_path(shard_dir, tag, base_idx),
        completion_payload(tag, base_idx, 2, 64, [path]),
    )


def test_ringed_finalize_uses_shared_shards_and_stores_only_kinetic(tmp_path):
    tag = str(tmp_path / "FlowedQuarkRinged" / "lat.FlowedQuarkRinged.9.0.sm")
    shard_dir = tmp_path / "FlowedQuarkRinged" / "shards"
    _write_ringed_base(shard_dir, tag, 9, 0)
    finalize_flowed_quark_ringed_norm_shards(shard_dir, tag, 1)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5["raw/kinetic_pervec"].shape == (2, 3, 4)
        np.testing.assert_array_equal(h5["raw/source_index"], [0, 1])
        np.testing.assert_array_equal(h5["raw/spin_index"], [-1, -1])
        np.testing.assert_allclose(h5["avg/kinetic_spacetime"], -10.0)
        assert "Z_ring_field_sqrt" not in h5["avg"]
        assert "Z_ring_bilinear" not in h5["avg"]
        assert "rand_seed" not in h5.attrs


def _write_kinetic_input(path, config_num, kinetic):
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = _ringed_attrs(config_num)
    with h5py.File(path, "w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        h5.create_dataset("flow_times", data=flow_times(0.1, 2))
        h5.require_group("avg").create_dataset(
            "kinetic_spacetime", data=np.asarray(kinetic, dtype=np.complex128)
        )


def test_ensemble_analyzer_averages_k_before_inverse(tmp_path):
    first = tmp_path / "cfg1.h5"
    second = tmp_path / "cfg2.h5"
    output = tmp_path / "ensemble.h5"
    _write_kinetic_input(first, 1, [-2.0, -2.0, -2.0])
    _write_kinetic_input(second, 2, [-4.0, -4.0, -4.0])
    analyze_ringed_ensemble([first, second], output)

    with h5py.File(output, "r") as h5:
        np.testing.assert_allclose(h5["avg/kinetic_ensemble"], -3.0)
        _, expected = compute_ringed_factors(-3.0 * np.ones(3), flow_times(0.1, 2))
        np.testing.assert_allclose(h5["avg/Z_ring_bilinear"][1:], expected[1:])
        assert h5.attrs["n_configurations"] == 2


def test_spin_color_trace_factor_scales_raw_average():
    raw = np.ones((2, 3, 4), dtype=np.complex128)
    np.testing.assert_allclose(kinetic_spacetime_from_raw(raw, 12), 12.0)
