from pathlib import Path

import h5py
import numpy as np

from application.analysis_helper.emt_proton_t44_analysis import (
    deterministic_stratified_sources,
    disconnected_ratio,
    load_t44_base_loops,
    optimized_ratio,
    pplus_unpolarized_c2,
    source_relative_loops,
    translation_covariance,
    two_way_bootstrap_disconnected,
)


def test_deterministic_sources_are_unique_and_time_stratified():
    first = deterministic_stratified_sources(config_num=1, stream=7, per_time=16)
    second = deterministic_stratified_sources(config_num=1, stream=7, per_time=16)
    changed = deterministic_stratified_sources(config_num=2, stream=7, per_time=16)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len({tuple(site) for site in first}) == 128
    np.testing.assert_array_equal(np.bincount(first[:, 3], minlength=8), np.full(8, 16))
    assert not np.array_equal(first, changed)


def test_pplus_projection_is_identity_plus_temporal_over_four():
    identity = np.asarray([2 + 3j, 4 - 1j])
    temporal = np.asarray([6 - 3j, -4 + 5j])
    np.testing.assert_allclose(
        pplus_unpolarized_c2(identity, temporal),
        0.25 * (identity + temporal),
    )


def test_zero_momentum_optimized_ratio_reduces_to_c3_over_c2():
    c2 = np.asarray([10.0, 8.0, 6.0, 4.0, 3.0], dtype=np.complex128)
    taus = np.asarray([1, 2])
    c3 = np.asarray([[2.0 + 1j, 3.0 - 2j]])
    ratio = optimized_ratio(c3, c2, c2[None, :], 3, taus)
    np.testing.assert_allclose(ratio, c3 / c2[3])


def test_loop_source_rephasing_and_time_roll():
    qext = np.asarray([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    loop = np.stack([np.arange(8), 10 + np.arange(8)]).astype(np.complex128)
    sources = np.asarray([[2, 0, 0, 3]], dtype=np.int32)
    result = source_relative_loops(loop, qext, sources, (8, 8, 8, 8))[0]
    np.testing.assert_allclose(result[0], -1j * np.roll(loop[0], -3))
    np.testing.assert_allclose(result[1], np.roll(loop[1], -3))


def test_translation_covariance_uses_unbiased_sample_factor():
    c2 = np.asarray([1.0, 2.0, 4.0])
    loop = np.asarray([[3.0], [5.0], [8.0]])
    observed = translation_covariance(c2, loop)[0]
    expected = np.cov(c2, loop[:, 0], ddof=1)[0, 1]
    np.testing.assert_allclose(observed, expected)


def _write_t44_file(path, incomplete=False):
    n_base = 3
    values = np.broadcast_to(np.arange(1, 4)[:, None, None, None]/512, (3, 2, 2, 8))
    with h5py.File(path, "w") as h5:
        h5.attrs.update(dict(output_kind="emt_complete_base_statistics_v2",
            shard_schema="disconnected_1pt_base_parts_v4", emt_operator_schema_version=5,
            base_values_volume_normalized=True, n_base_noise=n_base, vectors_per_base=2,
            spin_color_dilution="none", effective_n_inversions=6,
            qext=[[-1, 0, 0, 0], [0, 0, 0, 0]], flow_times=[0., 0.207936],
            global_lattice_size=[8, 8, 8, 8]))
        h5.create_dataset("base/base_noise_index", data=np.arange(n_base))
        h5.create_dataset("base/Tmunu/T44", data=values[:-1] if incomplete else values)


def test_t44_loader_groups_only_complete_hp_bases(tmp_path):
    complete = tmp_path / "complete.h5"
    _write_t44_file(complete)
    loops = load_t44_base_loops(complete)
    assert loops.base_values.shape == (3, 2, 2, 8)
    np.testing.assert_allclose(loops.base_values[:, 0, 0, 0], [1, 2, 3] / np.asarray(512))

    incomplete = tmp_path / "incomplete.h5"
    _write_t44_file(incomplete, incomplete=True)
    try:
        load_t44_base_loops(incomplete)
    except ValueError as err:
        assert "complete-base axes" in str(err)
    else:
        raise AssertionError("partial HP base should be rejected")


def test_disconnected_bootstrap_shapes_and_seed_reproducibility():
    sources = np.asarray([[idx, 0, 0, idx % 8] for idx in range(8)], dtype=np.int32)
    qext = np.asarray([[0, 0, 0, 0]], dtype=np.int32)
    c2_pf = np.asarray([5 + 0.1 * idx + np.arange(8) for idx in range(8)], np.complex128)
    c2_pi = c2_pf[:, None, :]
    base = np.asarray([
        np.full((1, 2, 8), 2 + 0.05 * idx, np.complex128)
        + np.arange(8)[None, None, :] * 0.01
        for idx in range(4)
    ])
    expected = disconnected_ratio(c2_pf, c2_pi, base, qext, sources, (8, 8, 8, 8), 3)
    first, errors_first = two_way_bootstrap_disconnected(
        c2_pf, c2_pi, base, qext, sources, (8, 8, 8, 8), 3,
        n_resamples=20, seed=9,
    )
    second, errors_second = two_way_bootstrap_disconnected(
        c2_pf, c2_pi, base, qext, sources, (8, 8, 8, 8), 3,
        n_resamples=20, seed=9,
    )
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(first, second)
    for key in ("source", "stochastic", "combined"):
        assert errors_first[key]["real"].shape == (1, 2)
        np.testing.assert_allclose(errors_first[key]["real"], errors_second[key]["real"])
