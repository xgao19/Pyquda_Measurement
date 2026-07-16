import numpy as np
import pytest

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    VALID_HP_ORDERINGS,
    apply_hierarchical_probe,
    ceil_log2,
    counter_zn_phase_indices,
    effective_n_inversions,
    hierarchical_probe_pattern,
    is_power_of_two,
    iter_noise_base_hp_interval,
    normalize_noise_scheme,
    part_source_bookkeeping,
    reconstruct_source_indices,
    validate_hierarchical_probing_options,
)


class TinyLatticeInfo:
    global_size = [2, 2, 2, 2]

    def coordinate(self, mu=None):
        axes = [np.arange(size) for size in self.global_size]
        coords = np.meshgrid(*axes, indexing="ij")
        if mu is None:
            return coords
        return coords[mu]


class FakeFermion:
    def __init__(self, data, latt_info=None):
        self.data = np.asarray(data, dtype=np.complex128)
        self.latt_info = latt_info or TinyLatticeInfo()

    def copy(self):
        return FakeFermion(self.data.copy(), self.latt_info)


class PartitionedTinyLatticeInfo(TinyLatticeInfo):
    def __init__(self, x_start, x_stop):
        self.x_start = int(x_start)
        self.x_stop = int(x_stop)

    def coordinate(self, mu=None):
        axes = [
            np.arange(self.x_start, self.x_stop),
            np.arange(self.global_size[1]),
            np.arange(self.global_size[2]),
            np.arange(self.global_size[3]),
        ]
        coords = np.meshgrid(*axes, indexing="ij")
        if mu is None:
            return coords
        return coords[mu]


class S8LatticeInfo(TinyLatticeInfo):
    global_size = [8, 8, 8, 8]


def _hp_displacement_correlation(latt_info, hp_count, ordering, mu, distance):
    patterns = np.asarray([
        hierarchical_probe_pattern(latt_info, hp_idx, ordering)
        for hp_idx in range(hp_count)
    ])
    shifted = np.roll(patterns, -int(distance), axis=int(mu) + 1)
    return np.mean(patterns * shifted)


def test_noise_scheme_and_hp_validation():
    assert normalize_noise_scheme(" ZN ") == "zn"
    assert normalize_noise_scheme("Hierarchical_Probing") == "hierarchical_probing"
    assert is_power_of_two(1)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(6)
    assert ceil_log2(1) == 0
    assert ceil_log2(9) == 4

    for ordering in VALID_HP_ORDERINGS:
        validate_hierarchical_probing_options(4, ordering)

def test_counter_z4_is_deterministic_and_uses_all_counter_fields():
    latt_info = TinyLatticeInfo()
    reference = counter_zn_phase_indices(latt_info, 17, 3, stream_seed=5, n=4)
    repeated = counter_zn_phase_indices(latt_info, 17, 3, stream_seed=5, n=4)

    np.testing.assert_array_equal(reference, repeated)
    assert set(np.unique(reference)) <= {0, 1, 2, 3}
    phases = np.asarray([1.0, 1.0j, -1.0, -1.0j])[reference]
    assert set(np.unique(phases)) <= {1.0, 1.0j, -1.0, -1.0j}
    assert np.unique(reference[0, 0, 0, 0]).size > 1

    assert not np.array_equal(reference, counter_zn_phase_indices(latt_info, 18, 3, stream_seed=5, n=4))
    assert not np.array_equal(reference, counter_zn_phase_indices(latt_info, 17, 4, stream_seed=5, n=4))
    assert not np.array_equal(reference, counter_zn_phase_indices(latt_info, 17, 3, stream_seed=6, n=4))


def test_counter_z4_is_independent_of_lattice_partitioning():
    full = counter_zn_phase_indices(TinyLatticeInfo(), 23, 2, stream_seed=7, n=4)
    left = counter_zn_phase_indices(PartitionedTinyLatticeInfo(0, 1), 23, 2, stream_seed=7, n=4)
    right = counter_zn_phase_indices(PartitionedTinyLatticeInfo(1, 2), 23, 2, stream_seed=7, n=4)

    np.testing.assert_array_equal(np.concatenate([left, right], axis=0), full)
    assert not np.array_equal(left, right)


def test_site_only_counter_z4_is_independent_of_lattice_partitioning():
    full = counter_zn_phase_indices(
        TinyLatticeInfo(), 23, 2, stream_seed=7, n=4, spin_count=1, color_count=1
    )[..., 0, 0]
    left = counter_zn_phase_indices(
        PartitionedTinyLatticeInfo(0, 1), 23, 2, stream_seed=7, n=4,
        spin_count=1, color_count=1,
    )[..., 0, 0]
    right = counter_zn_phase_indices(
        PartitionedTinyLatticeInfo(1, 2), 23, 2, stream_seed=7, n=4,
        spin_count=1, color_count=1,
    )[..., 0, 0]

    np.testing.assert_array_equal(np.concatenate([left, right], axis=0), full)
    assert set(np.unique(full)) <= {0, 1, 2, 3}


def test_noise_iterator_requires_counter_configuration():
    with pytest.raises(ValueError, match="config_num is required"):
        next(iter_noise_base_hp_interval(
            TinyLatticeInfo(), 0, 0, 1, 4, "zn", 1,
            next(iter(VALID_HP_ORDERINGS)), config_num=None,
        ))


def test_array_to_numpy_accepts_numpy_and_get_backends():
    from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy

    values = np.asarray([1.0, 2.0])
    assert array_to_numpy(values) is values

    class GetArray:
        def get(self):
            return values.copy()

    np.testing.assert_array_equal(array_to_numpy(GetArray()), values)


def test_direct_part_bookkeeping_plain_and_hp():
    plain = part_source_bookkeeping(3, 2, 5, 8)
    np.testing.assert_array_equal(plain["base_noise_index"], [3, 3, 3])
    np.testing.assert_array_equal(plain["hp_index"], [2, 3, 4])
    np.testing.assert_array_equal(
        reconstruct_source_indices(
            plain["base_noise_index"], plain["hp_index"], 8
        ),
        [26, 27, 28],
    )

def test_hierarchical_probes_reuse_one_counter_base_source():
    latt_info = TinyLatticeInfo()
    indices = counter_zn_phase_indices(latt_info, 31, 6, stream_seed=4, n=4)
    base = FakeFermion(np.asarray([1.0, 1.0j, -1.0, -1.0j])[indices], latt_info)
    hp0 = apply_hierarchical_probe(base, 0, "interleaved_xyzt_binary_projected_to_evenodd")
    hp1 = apply_hierarchical_probe(base, 1, "interleaved_xyzt_binary_projected_to_evenodd")
    pattern = hierarchical_probe_pattern(latt_info, 1, "interleaved_xyzt_binary_projected_to_evenodd")

    np.testing.assert_array_equal(hp0.data, base.data)
    np.testing.assert_array_equal(hp1.data, base.data * pattern[..., None, None])


def test_effective_inversions_and_reconstructed_source_indices():
    assert effective_n_inversions(3, "zn", 8) == 3
    assert effective_n_inversions(3, "hierarchical_probing", 8) == 24
    np.testing.assert_array_equal(
        reconstruct_source_indices([0, 0, 1, 1], [0, 1, 0, 1], 2),
        [0, 1, 2, 3],
    )


def test_hierarchical_probe_patterns_are_rademacher():
    latt_info = TinyLatticeInfo()
    for ordering in VALID_HP_ORDERINGS:
        pattern0 = hierarchical_probe_pattern(latt_info, 0, ordering)
        pattern1 = hierarchical_probe_pattern(latt_info, 1, ordering)

        np.testing.assert_array_equal(pattern0, np.ones_like(pattern0))
        assert set(np.unique(pattern1)) <= {-1.0, 1.0}
        assert np.any(pattern1 == -1.0)
        assert np.any(pattern1 == 1.0)


def test_interleaved_xyz_hp_pattern_is_time_independent():
    latt_info = TinyLatticeInfo()
    for hp_idx in range(8):
        pattern = hierarchical_probe_pattern(latt_info, hp_idx, "interleaved_xyz_binary_projected_to_evenodd")
        np.testing.assert_array_equal(pattern[:, :, :, 0], pattern[:, :, :, 1])


def test_interleaved_xyz_hp_pattern_resolves_spatial_bits_first():
    latt_info = TinyLatticeInfo()

    pattern_x = hierarchical_probe_pattern(latt_info, 1, "interleaved_xyz_binary_projected_to_evenodd")
    pattern_y = hierarchical_probe_pattern(latt_info, 2, "interleaved_xyz_binary_projected_to_evenodd")
    pattern_z = hierarchical_probe_pattern(latt_info, 4, "interleaved_xyz_binary_projected_to_evenodd")

    np.testing.assert_array_equal(pattern_x[0, :, :, :], -pattern_x[1, :, :, :])
    np.testing.assert_array_equal(pattern_y[:, 0, :, :], -pattern_y[:, 1, :, :])
    np.testing.assert_array_equal(pattern_z[:, :, 0, :], -pattern_z[:, :, 1, :])


@pytest.mark.parametrize("hp_count,max_cancelled_distance", [(16, 1), (256, 3)])
def test_interleaved_xyzt_complete_shell_cancels_four_dimensional_neighbors(
    hp_count, max_cancelled_distance
):
    latt_info = S8LatticeInfo()
    ordering = "interleaved_xyzt_binary_projected_to_evenodd"
    for mu in range(4):
        for distance in range(1, max_cancelled_distance + 1):
            assert _hp_displacement_correlation(
                latt_info, hp_count, ordering, mu, distance
            ) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("hp_count", [16, 256])
def test_interleaved_xyz_leaves_all_temporal_displacements_uncancelled(hp_count):
    latt_info = S8LatticeInfo()
    ordering = "interleaved_xyz_binary_projected_to_evenodd"
    for distance in range(1, 5):
        assert _hp_displacement_correlation(
            latt_info, hp_count, ordering, 3, distance
        ) == pytest.approx(1.0, abs=1e-15)
