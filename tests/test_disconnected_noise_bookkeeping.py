import numpy as np

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    VALID_HP_ORDERINGS,
    ceil_log2,
    effective_n_inversions,
    hierarchical_probe_pattern,
    is_power_of_two,
    normalize_noise_scheme,
    source_bookkeeping_arrays,
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


def test_effective_inversions_and_source_bookkeeping_arrays():
    assert effective_n_inversions(3, "zn", 8) == 3
    assert effective_n_inversions(3, "hierarchical_probing", 8) == 24

    bookkeeping = source_bookkeeping_arrays(4)
    np.testing.assert_array_equal(bookkeeping["source_index"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(bookkeeping["base_noise_index"], np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(bookkeeping["hp_index"], np.zeros(4, dtype=np.int32))


def test_hierarchical_probe_patterns_are_rademacher():
    latt_info = TinyLatticeInfo()
    for ordering in VALID_HP_ORDERINGS:
        pattern0 = hierarchical_probe_pattern(latt_info, 0, ordering)
        pattern1 = hierarchical_probe_pattern(latt_info, 1, ordering)

        np.testing.assert_array_equal(pattern0, np.ones_like(pattern0))
        assert set(np.unique(pattern1)) <= {-1.0, 1.0}
        assert np.any(pattern1 == -1.0)
        assert np.any(pattern1 == 1.0)
