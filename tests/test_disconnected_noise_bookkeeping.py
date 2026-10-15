import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_utils_vibe_develop as noise_helpers

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    VALID_HP_ORDERINGS,
    NESTED_HP_ORDERING,
    SPIN_COLOR_POINT_FACTOR,
    apply_hierarchical_probe,
    apply_spin_color_point_dilution,
    counter_zn_phase_indices,
    effective_n_inversions,
    hierarchical_probe_pattern,
    iter_noise_base_probe_interval,
    normalize_noise_scheme,
    normalize_spin_color_dilution,
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


def test_noise_scheme_and_hp_validation():
    assert normalize_noise_scheme(" ZN ") == "zn"
    assert normalize_noise_scheme("Hierarchical_Probing") == "hierarchical_probing"

    for ordering in VALID_HP_ORDERINGS:
        validate_hierarchical_probing_options(2 if ordering == NESTED_HP_ORDERING else 4, ordering)

    assert normalize_spin_color_dilution(" POINT ") == "point"
    assert SPIN_COLOR_POINT_FACTOR == 12
    with pytest.raises(ValueError, match="spin_color_dilution"):
        normalize_spin_color_dilution("interlace")

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
        next(iter_noise_base_probe_interval(
            TinyLatticeInfo(), 0, 0, 1, 4, "zn", 1,
            next(iter(VALID_HP_ORDERINGS)), config_num=None,
        ))


def test_array_to_numpy_accepts_numpy_and_get_backends():
    from pyquda_measurement_utils.tools import array_to_numpy

    values = np.asarray([1.0, 2.0])
    assert array_to_numpy(values) is values

    class GetArray:
        def get(self):
            return values.copy()

    np.testing.assert_array_equal(array_to_numpy(GetArray()), values)


def test_direct_part_bookkeeping_plain_and_hp():
    plain = part_source_bookkeeping(3, 2, 5, 8)
    np.testing.assert_array_equal(plain["base_noise_index"], [3, 3, 3])
    np.testing.assert_array_equal(plain["probe_index"], [2, 3, 4])
    np.testing.assert_array_equal(
        reconstruct_source_indices(
            plain["base_noise_index"], plain["probe_index"], 8
        ),
        [26, 27, 28],
    )


def test_point_spin_color_bookkeeping_is_complete_and_ordered():
    bookkeeping = part_source_bookkeeping(2, 3, 5, 8, "point")
    assert len(bookkeeping["probe_index"]) == 24
    np.testing.assert_array_equal(bookkeeping["base_noise_index"], np.full(24, 2))
    np.testing.assert_array_equal(bookkeeping["probe_index"], np.repeat([3, 4], 12))
    np.testing.assert_array_equal(bookkeeping["spin_index"], np.tile(np.repeat(range(4), 3), 2))
    np.testing.assert_array_equal(bookkeeping["color_index"], np.tile(range(3), 8))


def test_point_spin_color_projectors_partition_a_source():
    values = np.arange(16 * 4 * 3).reshape(2, 2, 2, 2, 4, 3) + 1j
    source = FakeFermion(values)
    pieces = [
        apply_spin_color_point_dilution(source, spin, color).data
        for spin in range(4) for color in range(3)
    ]
    np.testing.assert_array_equal(np.sum(pieces, axis=0), source.data)
    assert all(np.count_nonzero(piece) == 16 for piece in pieces)


def test_point_spin_color_iterator_reuses_each_spatial_probe(monkeypatch):
    latt_info = TinyLatticeInfo()
    values = np.arange(16 * 4 * 3).reshape(2, 2, 2, 2, 4, 3) + 1j
    base = FakeFermion(values, latt_info)
    monkeypatch.setattr(
        noise_helpers,
        "make_counter_zn_noise_fermion",
        lambda *_args, **_kwargs: base.copy(),
    )
    records = list(iter_noise_base_probe_interval(
        latt_info, 1, 0, 2, 4, "hierarchical_probing", 2,
        NESTED_HP_ORDERING, config_num=7, spin_color_dilution="point",
    ))
    assert len(records) == 24
    assert [record[0] for record in records] == list(range(24, 48))
    for probe_idx in range(2):
        selected = records[probe_idx * 12:(probe_idx + 1) * 12]
        assert [(record[4], record[5]) for record in selected] == [
            (spin, color) for spin in range(4) for color in range(3)
        ]
        expected = apply_hierarchical_probe(base, probe_idx, NESTED_HP_ORDERING)
        np.testing.assert_array_equal(
            np.sum([record[3].data for record in selected], axis=0),
            expected.data,
        )

def test_hierarchical_probes_reuse_one_counter_base_source():
    latt_info = TinyLatticeInfo()
    indices = counter_zn_phase_indices(latt_info, 31, 6, stream_seed=4, n=4)
    base = FakeFermion(np.asarray([1.0, 1.0j, -1.0, -1.0j])[indices], latt_info)
    hp0 = apply_hierarchical_probe(base, 0, NESTED_HP_ORDERING)
    hp1 = apply_hierarchical_probe(base, 1, NESTED_HP_ORDERING)
    pattern = hierarchical_probe_pattern(latt_info, 1, NESTED_HP_ORDERING)

    np.testing.assert_array_equal(hp0.data, base.data)
    np.testing.assert_array_equal(hp1.data, base.data * pattern[..., None, None])


def test_effective_inversions_and_reconstructed_source_indices():
    assert effective_n_inversions(3, "zn", 8) == 3
    assert effective_n_inversions(3, "hierarchical_probing", 32) == 96
    assert effective_n_inversions(
        3, "hierarchical_probing", 32, spin_color_dilution="point"
    ) == 1152
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



@pytest.mark.parametrize("ordering", [
    "global_xyzt_gray_projected_to_evenodd",
    "interleaved_xyzt_binary_projected_to_evenodd",
    "interleaved_xyz_binary_projected_to_evenodd",
    "spatial_xyz_then_t_gray_projected_to_evenodd",
])
def test_retired_orderings_are_rejected_even_for_probe_zero(ordering):
    with pytest.raises(ValueError):
        hierarchical_probe_pattern(TinyLatticeInfo(), 0, ordering)
