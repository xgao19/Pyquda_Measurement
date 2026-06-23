import numpy as np

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    VALID_HP_ORDERINGS,
    apply_hierarchical_probe,
    apply_spin_color_point_dilution,
    ceil_log2,
    effective_n_inversions,
    hierarchical_probe_pattern,
    is_power_of_two,
    normalize_noise_scheme,
    normalize_spin_color_dilution,
    source_bookkeeping_arrays,
    spin_color_dilution_factor,
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


def test_noise_scheme_and_hp_validation():
    assert normalize_noise_scheme(" ZN ") == "zn"
    assert normalize_noise_scheme("Hierarchical_Probing") == "hierarchical_probing"
    assert normalize_spin_color_dilution(" Point ") == "point"
    assert spin_color_dilution_factor("none") == 1
    assert spin_color_dilution_factor("point") == 12
    assert is_power_of_two(1)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(6)
    assert ceil_log2(1) == 0
    assert ceil_log2(9) == 4

    for ordering in VALID_HP_ORDERINGS:
        validate_hierarchical_probing_options(4, ordering)

    try:
        normalize_spin_color_dilution("spin")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid spin-color dilution mode should fail")


def test_effective_inversions_and_source_bookkeeping_arrays():
    assert effective_n_inversions(3, "zn", 8) == 3
    assert effective_n_inversions(3, "hierarchical_probing", 8) == 24
    assert effective_n_inversions(3, "zn", 8, "point") == 36
    assert effective_n_inversions(3, "hierarchical_probing", 8, "point") == 288

    bookkeeping = source_bookkeeping_arrays(4)
    np.testing.assert_array_equal(bookkeeping["source_index"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(bookkeeping["base_noise_index"], np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(bookkeeping["hp_index"], np.zeros(4, dtype=np.int32))
    sc_bookkeeping = source_bookkeeping_arrays(4, include_spin_color=True)
    np.testing.assert_array_equal(sc_bookkeeping["spin_index"], -np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(sc_bookkeeping["color_index"], -np.ones(4, dtype=np.int32))


def test_hierarchical_probe_patterns_are_rademacher():
    latt_info = TinyLatticeInfo()
    for ordering in VALID_HP_ORDERINGS:
        pattern0 = hierarchical_probe_pattern(latt_info, 0, ordering)
        pattern1 = hierarchical_probe_pattern(latt_info, 1, ordering)

        np.testing.assert_array_equal(pattern0, np.ones_like(pattern0))
        assert set(np.unique(pattern1)) <= {-1.0, 1.0}
        assert np.any(pattern1 == -1.0)
        assert np.any(pattern1 == 1.0)


def test_spin_color_point_dilution_keeps_one_channel():
    data = np.ones((2, 2, 2, 2, 4, 3), dtype=np.complex128)
    diluted = apply_spin_color_point_dilution(FakeFermion(data), spin_idx=2, color_idx=1)

    mask = np.zeros_like(data, dtype=bool)
    mask[..., 2, 1] = True
    assert np.all(diluted.data[mask] == 1)
    assert np.all(diluted.data[~mask] == 0)


def test_hp_pattern_is_site_only_before_spin_color_point_dilution():
    latt_info = TinyLatticeInfo()
    data = np.ones((2, 2, 2, 2, 4, 3), dtype=np.complex128)
    hp_source = apply_hierarchical_probe(FakeFermion(data, latt_info), 1, "interleaved_xyzt_binary_projected_to_evenodd")
    diluted = apply_spin_color_point_dilution(hp_source, spin_idx=1, color_idx=2)
    pattern = hierarchical_probe_pattern(latt_info, 1, "interleaved_xyzt_binary_projected_to_evenodd")

    np.testing.assert_allclose(diluted.data[..., 1, 2], pattern)
    assert np.count_nonzero(diluted.data[..., :1, :]) == 0
    assert np.count_nonzero(diluted.data[..., 2:, :]) == 0
    assert np.count_nonzero(diluted.data[..., 1, :2]) == 0
