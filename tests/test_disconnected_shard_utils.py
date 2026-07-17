"""Tests for shared disconnected noise, shard, and sample-log utilities."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_utils_vibe_develop as shard_module

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    emt_tensor_from_derivative_bilinear,
    finalize_emt_quark_1pt_shards,
    ringed_kinetic_pervec_from_derivative,
)
from pyquda_measurement_utils.fermion_bilinear_basis import basis_attrs, basis_metadata
from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    QTMD_LOOP_CONVENTION,
    QTMD_SCHEMA_VERSION,
    QTMD_TRACE_TARGET,
    finalize_disconnected_qtmd_1pt_shards,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    SHARD_SCHEMA,
    append_completed_base,
    base_part_ranges,
    hp_vectors_per_base,
    prepare_sample_log,
    sample_log_fingerprint,
    selected_base_range,
    shard_part_attrs,
    shard_part_path,
    write_raw_part_hdf5,
)


def test_base_ranges_and_nondivisible_part_layout():
    assert list(selected_base_range(8, 3, 5)) == [3, 4]
    assert hp_vectors_per_base("zn", 256) == 1
    assert hp_vectors_per_base("hierarchical_probing", 16) == 16
    assert base_part_ranges(1, 64) == [(0, 0, 1)]


def test_shard_attribute_comparison_treats_matching_nan_as_equal():
    assert shard_module._attr_equal(np.nan, np.nan)
    assert shard_module._attr_equal([1.0, np.nan], [1.0, np.nan])
    assert not shard_module._attr_equal(np.nan, 0.0)
    assert base_part_ranges(16, 64) == [(0, 0, 16)]
    assert base_part_ranges(256, 64) == [
        (0, 0, 64), (1, 64, 128), (2, 128, 192), (3, 192, 256)
    ]
    assert base_part_ranges(130, 64)[-1] == (2, 128, 130)


def _common_attrs(configured_n_vec):
    attrs = {
        "measurement": "quark_1pt",
        "output_kind": "emt_quark_1pt",
        "shard_schema": SHARD_SCHEMA,
        "block_interval_solves": 1,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
        "flow_times": np.asarray([0.0, 0.1]),
        "qext": np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32),
        "volume_norm": 8,
        "emt_operator_schema_version": 5,
        "operator_normalization": "unrenormalized_flowed_quark_bilinear",
        "renormalization_applied": False,
        "renormalization_stage": "analysis_stage",
        "flavor_convention": "single_flavor_trace_for_this_dirac_operator",
        "derivative_convention": "test",
        "mass": 0.1,
        "csw": 1.0,
        "gauge_preprocessing": "test",
        "t_boundary": -1,
        "n_zn": 4,
        "config_num": 9,
        "noise_stream": 3,
        "noise_generator": "splitmix64_global_coordinate_v1",
        "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 2,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "n_vec": configured_n_vec,
    }
    attrs.update(basis_attrs())
    return attrs


def _write_synthetic_base(shard_dir, tag, base_idx, configured_n_vec):
    for part_idx, hp_start, hp_stop in base_part_ranges(2, 1):
        path = shard_part_path(shard_dir, tag, base_idx, part_idx, hp_start, hp_stop)
        common = _common_attrs(configured_n_vec)
        common.pop("n_vec")
        attrs = shard_part_attrs(common, base_idx, part_idx, hp_start, hp_stop, 2)
        source_idx = base_idx * 2 + hp_start
        local = np.full((1, 16, 2, 2, 3), source_idx + 0.5, dtype=np.complex128)
        derivative = np.zeros((1, 16, 4, 2, 2, 3), dtype=np.complex128)
        for mu, gamma_position in enumerate((3, 5, 7, 1)):
            derivative[0, gamma_position, mu, 0] = source_idx + mu + 1
        norm = np.full((1, 2, 2, 3), source_idx + 1, dtype=np.complex128)
        write_raw_part_hdf5(
            path,
            {
                "local_bilinear_pervec": local,
                "derivative_bilinear_pervec": derivative,
                "flowed_noise_norm_pervec": norm,
            },
            attrs,
            {
                "base_noise_index": [base_idx],
                "hp_index": [hp_start],
            },
            metadata_datasets=basis_metadata(),
        )


def test_emt_finalize_streams_shards_and_embeds_ringed_kinetic(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    shard_dir = tmp_path / "EMTc" / "shards"
    _write_synthetic_base(shard_dir, tag, 0, configured_n_vec=1)
    _write_synthetic_base(shard_dir, tag, 1, configured_n_vec=2)

    assert finalize_emt_quark_1pt_shards(shard_dir, tag, 2) == tag + ".h5"

    with h5py.File(tag + ".h5", "r") as emt:
        assert emt["raw/local_bilinear_pervec"].shape == (4, 16, 2, 2, 3)
        assert emt["raw/derivative_bilinear_pervec"].shape == (4, 16, 4, 2, 2, 3)
        assert "raw/Tmunu_pervec" not in emt
        assert "raw/CHI_pervec" not in emt
        assert "avg/CHI" not in emt
        assert emt["raw/flowed_noise_norm_pervec"].shape == (4, 2, 2, 3)
        assert "raw/source_index" not in emt
        np.testing.assert_array_equal(emt["raw/base_noise_index"][()], [0, 0, 1, 1])
        np.testing.assert_array_equal(emt["raw/hp_index"][()], [0, 1, 0, 1])
        raw_derivative = emt["raw/derivative_bilinear_pervec"][()]
        raw_t = emt_tensor_from_derivative_bilinear(raw_derivative)
        expected_k = ringed_kinetic_pervec_from_derivative(raw_derivative, 0, 8)
        np.testing.assert_allclose(emt["derived/ringed/kinetic_pervec"][()], expected_k)
        np.testing.assert_allclose(
            emt["derived/ringed/kinetic_spacetime"][()],
            np.mean(expected_k, axis=(0, -1)),
        )
        assert not emt["derived/ringed"].attrs["ringed_factors_stored"]
        assert emt.attrs["n_base_noise"] == 2
        assert emt.attrs["effective_n_inversions"] == 4
        assert emt["physical_from_pyquda"].shape == (16, 16)
        for mu in range(4):
            for nu in range(mu, 4):
                np.testing.assert_allclose(
                    emt[f"avg/Tmunu/T{mu+1}{nu+1}"][()],
                    np.mean(raw_t[:, mu, nu], axis=0) / 8,
                )


def test_finalize_rejects_partial_base_and_preserves_old_canonical(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    Path(tag).parent.mkdir(parents=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"
    shard_dir = tmp_path / "EMTc" / "shards"
    _write_synthetic_base(shard_dir, tag, 0, configured_n_vec=2)

    with pytest.raises(ValueError, match="filename coverage mismatch"):
        finalize_emt_quark_1pt_shards(shard_dir, tag, 2)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"


def test_finalizer_rejects_obsolete_persisted_source_index(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    shard_dir = tmp_path / "EMTc" / "shards"
    _write_synthetic_base(shard_dir, tag, 0, configured_n_vec=1)
    first = shard_part_path(shard_dir, tag, 0, 0, 0, 1)
    with h5py.File(first, "r+") as h5:
        h5["raw"].create_dataset("source_index", data=[0])
    with pytest.raises(ValueError, match="obsolete raw/source_index"):
        finalize_emt_quark_1pt_shards(shard_dir, tag, 1)


def test_emt_finalizer_rejects_one_sided_schema_v4_and_preserves_canonical(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    shard_dir = tmp_path / "EMTc" / "shards"
    _write_synthetic_base(shard_dir, tag, 0, configured_n_vec=1)
    first = shard_part_path(shard_dir, tag, 0, 0, 0, 1)
    with h5py.File(first, "r+") as h5:
        h5.attrs["emt_operator_schema_version"] = 4
    Path(tag).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"
    with pytest.raises(ValueError, match="require emt_operator_schema_version=5"):
        finalize_emt_quark_1pt_shards(shard_dir, tag, 1)
    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"


def test_emt_finalizer_rejects_old_tmunu_only_shard_schema(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    Path(tag).parent.mkdir(parents=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"
    shard_dir = tmp_path / "EMTc" / "shards"
    path = shard_part_path(shard_dir, tag, 0, 0, 0, 1)
    attrs = shard_part_attrs(_common_attrs(1), 0, 0, 0, 1, 1)
    write_raw_part_hdf5(
        path,
        {
            "Tmunu_pervec": np.zeros((1, 4, 4, 2, 2, 3), dtype=np.complex128),
            "CHI_pervec": np.zeros((1, 2, 2, 2, 3), dtype=np.complex128),
        },
        attrs,
        {"base_noise_index": [0], "hp_index": [0]},
    )
    with pytest.raises(ValueError, match="missing metadata dataset|missing raw"):
        finalize_emt_quark_1pt_shards(shard_dir, tag, 1)
    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"


def _write_synthetic_qtmd_base(shard_dir, tag, base_idx, legacy=False):
    path = shard_part_path(shard_dir, tag, base_idx, 0, 0, 1)
    attrs = shard_part_attrs({
        "measurement": "disconnected_qTMD_1pt",
        "output_kind": "disconnected_qTMD_1pt",
        "shard_schema": SHARD_SCHEMA,
        "block_interval_solves": 64,
        "operator_kind": "GI_PDF",
        "qext": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "W_index_list": np.asarray([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
        "gamma_list": np.asarray(["5", "T"], dtype="S"),
        "volume_norm": 8,
        "mass": 0.1,
        "csw": 1.0,
        "gauge_preprocessing": "test",
        "t_boundary": -1,
        "n_zn": 4,
        "config_num": 9,
        "noise_stream": 2,
        "noise_generator": "splitmix64_global_coordinate_v1",
        "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
        "noise_scheme": "zn",
        "hp_num_vectors": 1,
        "hp_ordering": "global_xyzt_gray_projected_to_evenodd",
        "gi_qtmd_staple_mode": "link_cache",
        "schema_version": 1 if legacy else QTMD_SCHEMA_VERSION,
        "loop_convention": "eta_dagger_Gamma_O_b_xi" if legacy else QTMD_LOOP_CONVENTION,
        "trace_target": "legacy_incorrect_trace" if legacy else QTMD_TRACE_TARGET,
    }, base_idx, 0, 0, 1, 1)
    metadata = {
        "gamma_list": np.asarray(["5", "T"], dtype="S"),
        "momentum_list": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "W_index_list": np.asarray([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
    }
    loop = np.full((1, 2, 2, 1, 3), base_idx + 1, dtype=np.complex128)
    write_raw_part_hdf5(
        path, {"loop_pervec": loop}, attrs,
        {"base_noise_index": [base_idx], "hp_index": [0]},
        metadata_datasets=metadata,
    )


def test_sample_log_is_exact_hdf5_independent_and_nvec_extensible(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    log = tmp_path / "sample.log"
    attrs_one = _common_attrs(1)
    attrs_many = _common_attrs(128)
    assert sample_log_fingerprint(attrs_one) == sample_log_fingerprint(attrs_many)

    assert prepare_sample_log(log, tag, attrs_one) == set()
    assert append_completed_base(log, tag, attrs_one, 1)
    assert append_completed_base(log, tag, attrs_one, 10)
    assert not append_completed_base(log, tag, attrs_one, 1)
    assert prepare_sample_log(log, tag, attrs_many) == {1, 10}

    lines = log.read_text().splitlines()
    assert lines[1:] == ["base000001", "base000010"]
    assert not list(tmp_path.rglob("*.h5"))


def test_sample_log_header_mismatch_fails_without_hdf5_probe(tmp_path):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    log = tmp_path / "sample.log"
    attrs = _common_attrs(1)
    prepare_sample_log(log, tag, attrs)
    changed = dict(attrs)
    changed["noise_stream"] = 99
    with pytest.raises(ValueError, match="header mismatch"):
        prepare_sample_log(log, tag, changed)


def test_sample_log_falls_back_to_posix_lock_when_flock_is_unsupported(
    tmp_path, monkeypatch
):
    tag = str(tmp_path / "EMTc" / "lat.EMTc.9.0.sm")
    log = tmp_path / "sample.log"
    attrs = _common_attrs(1)
    original_lockf = shard_module.fcntl.lockf
    lockf_calls = []

    def unsupported_flock(*args):
        raise OSError(524, "DVS does not support flock")

    def recorded_lockf(*args):
        lockf_calls.append(args[1])
        return original_lockf(*args)

    monkeypatch.setattr(shard_module.fcntl, "flock", unsupported_flock)
    monkeypatch.setattr(shard_module.fcntl, "lockf", recorded_lockf)
    assert prepare_sample_log(log, tag, attrs) == set()
    assert append_completed_base(log, tag, attrs, 0)
    assert prepare_sample_log(log, tag, attrs) == {0}
    assert shard_module.fcntl.LOCK_EX in lockf_calls
    assert shard_module.fcntl.LOCK_UN in lockf_calls


def test_sample_log_parallel_nonoverlapping_base_appends(tmp_path):
    tag = str(tmp_path / "qTMD1pt" / "lat.qTMD1pt.9.0.sm")
    log = tmp_path / "sample.log"
    attrs = _common_attrs(8)
    prepare_sample_log(log, tag, attrs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda base: append_completed_base(log, tag, attrs, base),
            range(8),
        ))
    assert all(results)
    assert prepare_sample_log(log, tag, attrs) == set(range(8))


def test_qtmd_finalize_streams_source_independent_canonical(tmp_path):
    tag = str(tmp_path / "qTMD1pt" / "lat.qTMD1pt.9.0.sm")
    shard_dir = tmp_path / "qTMD1pt" / "shards"
    _write_synthetic_qtmd_base(shard_dir, tag, 0)
    _write_synthetic_qtmd_base(shard_dir, tag, 1)

    finalize_disconnected_qtmd_1pt_shards(shard_dir, tag, 2)

    with h5py.File(tag + ".h5", "r") as h5:
        assert "raw/source_index" not in h5
        np.testing.assert_array_equal(h5["raw/base_noise_index"][()], [0, 1])
        np.testing.assert_allclose(h5["raw/loop_pervec"][0], 1.0)
        np.testing.assert_allclose(h5["raw/loop_pervec"][1], 2.0)
        np.testing.assert_allclose(h5["avg/SS/5/PX0PY0PZ0/b_X/eta0/bT0/bz0"][()], 1.5 / 8.0)
        assert h5.attrs["n_zn"] == 4
        assert h5.attrs["noise_generator"] == "splitmix64_global_coordinate_v1"
        assert h5.attrs["schema_version"] == QTMD_SCHEMA_VERSION
        assert h5.attrs["loop_convention"] == QTMD_LOOP_CONVENTION
        assert h5.attrs["trace_target"] == QTMD_TRACE_TARGET
        assert "rand_seed" not in h5.attrs
        assert "tol" not in h5.attrs
        assert "maxiter" not in h5.attrs


def test_qtmd_finalize_rejects_legacy_trace_and_preserves_canonical(tmp_path):
    tag = str(tmp_path / "qTMD1pt" / "lat.qTMD1pt.9.0.sm")
    shard_dir = tmp_path / "qTMD1pt" / "shards"
    _write_synthetic_qtmd_base(shard_dir, tag, 0, legacy=True)
    Path(tag).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"

    with pytest.raises(ValueError, match="old disconnected qTMD data"):
        finalize_disconnected_qtmd_1pt_shards(shard_dir, tag, 1)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"


def test_qtmd_finalize_rejects_duplicate_part_interval(tmp_path):
    tag = str(tmp_path / "qTMD1pt" / "lat.qTMD1pt.9.0.sm")
    shard_dir = tmp_path / "qTMD1pt" / "shards"
    _write_synthetic_qtmd_base(shard_dir, tag, 0)
    duplicate = shard_dir / (
        Path(tag).name + ".base000000.part0000.hp0000-0001.h5"
    )
    with h5py.File(duplicate, "w"):
        pass
    with pytest.raises(ValueError, match="exactly one base-0 part-0"):
        finalize_disconnected_qtmd_1pt_shards(shard_dir, tag, 1)


def test_qtmd_finalize_rejects_mixed_metadata_during_streaming(tmp_path):
    tag = str(tmp_path / "qTMD1pt" / "lat.qTMD1pt.9.0.sm")
    shard_dir = tmp_path / "qTMD1pt" / "shards"
    _write_synthetic_qtmd_base(shard_dir, tag, 0)
    _write_synthetic_qtmd_base(shard_dir, tag, 1)
    second = shard_part_path(shard_dir, tag, 1, 0, 0, 1)
    with h5py.File(second, "r+") as h5:
        h5["W_index_list"][0, 1] = 99
    Path(tag).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tag + ".h5", "w") as h5:
        h5.attrs["sentinel"] = "old"

    with pytest.raises(ValueError, match="incompatible metadata"):
        finalize_disconnected_qtmd_1pt_shards(shard_dir, tag, 2)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["sentinel"] == "old"
