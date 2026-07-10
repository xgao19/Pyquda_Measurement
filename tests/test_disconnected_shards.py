import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    finalize_emt_quark_1pt_shards,
    ringed_kinetic_pervec_from_emt,
)
from pyquda_measurement_utils.disconnected_shards import (
    SHARD_SCHEMA,
    base_completion_path,
    base_part_ranges,
    completion_payload,
    expected_part_attrs,
    hp_vectors_per_base,
    selected_base_range,
    shard_part_path,
    write_base_completion_marker,
    write_raw_part_hdf5,
)


def test_base_ranges_and_nondivisible_part_layout():
    assert list(selected_base_range(8, 3, 5)) == [3, 4]
    assert hp_vectors_per_base("zn", 256) == 1
    assert hp_vectors_per_base("hierarchical_probing", 16) == 16
    assert base_part_ranges(1, 64) == [(0, 0, 1)]
    assert base_part_ranges(16, 64) == [(0, 0, 16)]
    assert base_part_ranges(256, 64) == [
        (0, 0, 64), (1, 64, 128), (2, 128, 192), (3, 192, 256)
    ]
    assert base_part_ranges(130, 64)[-1] == (2, 128, 130)


def _common_attrs(configured_n_vec):
    return {
        "measurement": "quark_1pt",
        "output_kind": "emt_quark_1pt",
        "shard_schema": SHARD_SCHEMA,
        "block_interval_solves": 1,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
        "flow_times": np.asarray([0.0, 0.1]),
        "qext": np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32),
        "pf": np.asarray([0, 0, 0, 0], dtype=np.int32),
        "p_2pt": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "volume_norm": 8,
        "upper_triangle_only": True,
        "operator_normalization": "unrenormalized_flowed_quark_bilinear",
        "renormalization_applied": False,
        "renormalization_stage": "analysis_stage",
        "flavor_convention": "single_flavor_trace_for_this_dirac_operator",
        "derivative_convention": "test",
        "mass": 0.1,
        "csw": 1.0,
        "tol": 1e-10,
        "maxiter": 100,
        "gauge_preprocessing": "test",
        "t_boundary": -1,
        "n_zn": 4,
        "config_num": 9,
        "rand_seed": 3,
        "noise_stream": 3,
        "noise_generator": "splitmix64_global_coordinate_v1",
        "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 2,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "configured_n_base_noise": configured_n_vec,
    }


def _write_synthetic_base(shard_dir, tag, base_idx, configured_n_vec):
    part_paths = []
    for part_idx, hp_start, hp_stop in base_part_ranges(2, 1):
        path = shard_part_path(shard_dir, tag, base_idx, part_idx, hp_start, hp_stop)
        common = _common_attrs(configured_n_vec)
        configured = common.pop("configured_n_base_noise")
        attrs = expected_part_attrs(common, base_idx, part_idx, hp_start, hp_stop, 2)
        attrs["configured_n_base_noise"] = configured
        source_idx = base_idx * 2 + hp_start
        tmunu = np.zeros((1, 4, 4, 2, 2, 3), dtype=np.complex128)
        for mu in range(4):
            tmunu[0, mu, mu, 0] = source_idx + mu + 1
        chi = np.full((1, 2, 2, 2, 3), source_idx + 1, dtype=np.complex128)
        write_raw_part_hdf5(
            path,
            {"Tmunu_pervec": tmunu, "CHI_pervec": chi},
            attrs,
            {
                "source_index": [source_idx],
                "base_noise_index": [base_idx],
                "hp_index": [hp_start],
            },
        )
        part_paths.append(path)
    write_base_completion_marker(
        base_completion_path(shard_dir, tag, base_idx),
        completion_payload(tag, base_idx, 2, 1, part_paths),
    )


def test_emt_finalize_streams_shards_and_builds_kinetic_companion(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    ringed_tag = str(tmp_path / "FlowedQuarkRinged" / "lat.FlowedQuarkRinged.9.0.sm")
    shard_dir = tmp_path / "EMTc" / "shards"
    _write_synthetic_base(shard_dir, tag, 0, configured_n_vec=1)
    _write_synthetic_base(shard_dir, tag, 1, configured_n_vec=2)

    finalize_emt_quark_1pt_shards(shard_dir, tag, ringed_tag, 2)

    with h5py.File(tag + ".h5", "r") as emt, h5py.File(ringed_tag + ".h5", "r") as ringed:
        assert emt["raw/Tmunu_pervec"].shape == (4, 4, 4, 2, 2, 3)
        np.testing.assert_array_equal(emt["raw/source_index"][()], [0, 1, 2, 3])
        np.testing.assert_array_equal(emt["raw/base_noise_index"][()], [0, 0, 1, 1])
        np.testing.assert_array_equal(emt["raw/hp_index"][()], [0, 1, 0, 1])
        raw_t = emt["raw/Tmunu_pervec"][()]
        expected_k = ringed_kinetic_pervec_from_emt(raw_t, 0, 8)
        np.testing.assert_allclose(ringed["raw/kinetic_pervec"][()], expected_k)
        np.testing.assert_allclose(ringed["avg/kinetic_spacetime"][()], np.mean(expected_k, axis=(0, -1)))
        assert "Z_ring_bilinear" not in ringed["avg"]
        assert emt.attrs["n_base_noise"] == 2
        assert emt.attrs["effective_n_inversions"] == 4


def test_finalize_rejects_partial_base_and_preserves_old_canonical(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    ringed_tag = str(tmp_path / "FlowedQuarkRinged" / "lat.FlowedQuarkRinged.9.0.sm")
    Path(tag).parent.mkdir(parents=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"

    with pytest.raises(ValueError, match="missing completion marker"):
        finalize_emt_quark_1pt_shards(tmp_path / "EMTc" / "shards", tag, ringed_tag, 1)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"
