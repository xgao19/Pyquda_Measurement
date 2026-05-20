import numpy as np

import pyquda_measurement_utils.boosted_smearing_pyquda as boosted


class TinyLatticeInfo:
    size = [4, 4, 4, 2]
    global_size = [4, 4, 4, 2]

    def evenodd(self, arr, _dagger):
        return arr


class FakeLatticeComplex:
    def __init__(self, latt_info):
        self.latt_info = latt_info
        self.data = None


def test_exp_complex_matches_numpy_complex_exponential():
    real = np.array([0.0, -1.0, -2.0])
    imag = np.array([0.0, 0.5, -0.25])
    np.testing.assert_allclose(boosted._exp_complex(np, real, imag), np.exp(real + 1j * imag))


def test_boosted_smearing_kernel_zero_boost_is_real_and_time_independent():
    old_lattice_complex = boosted.LatticeComplex
    old_get_rank = boosted.getMPIRank
    old_get_coord = boosted.getCoordFromRank
    had_asnumpy = hasattr(np, "asnumpy")
    old_asnumpy = getattr(np, "asnumpy", None)
    try:
        boosted.LatticeComplex = FakeLatticeComplex
        boosted.getMPIRank = lambda: 0
        boosted.getCoordFromRank = lambda _rank: [0, 0, 0, 0]
        np.asnumpy = np.asarray

        kernel = boosted._build_kernel_realspace_distributed(np, TinyLatticeInfo(), 2.0, [0, 0, 0]).data
    finally:
        boosted.LatticeComplex = old_lattice_complex
        boosted.getMPIRank = old_get_rank
        boosted.getCoordFromRank = old_get_coord
        if had_asnumpy:
            np.asnumpy = old_asnumpy
        else:
            delattr(np, "asnumpy")

    assert kernel.shape == (2, 4, 4, 4)
    np.testing.assert_allclose(kernel.imag, 0.0)
    np.testing.assert_allclose(kernel[0], kernel[1])


def test_boosted_smearing_kernel_opposite_boosts_are_conjugates():
    old_lattice_complex = boosted.LatticeComplex
    old_get_rank = boosted.getMPIRank
    old_get_coord = boosted.getCoordFromRank
    had_asnumpy = hasattr(np, "asnumpy")
    old_asnumpy = getattr(np, "asnumpy", None)
    try:
        boosted.LatticeComplex = FakeLatticeComplex
        boosted.getMPIRank = lambda: 0
        boosted.getCoordFromRank = lambda _rank: [0, 0, 0, 0]
        np.asnumpy = np.asarray

        plus = boosted._build_kernel_realspace_distributed(np, TinyLatticeInfo(), 2.0, [1, 0, 0]).data
        minus = boosted._build_kernel_realspace_distributed(np, TinyLatticeInfo(), 2.0, [-1, 0, 0]).data
    finally:
        boosted.LatticeComplex = old_lattice_complex
        boosted.getMPIRank = old_get_rank
        boosted.getCoordFromRank = old_get_coord
        if had_asnumpy:
            np.asnumpy = old_asnumpy
        else:
            delattr(np, "asnumpy")

    np.testing.assert_allclose(plus, minus.conj())
