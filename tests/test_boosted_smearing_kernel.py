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


def test_wait_sycl_queue_waits_only_when_queue_is_present():
    class FakeQueue:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    class FakeData:
        def __init__(self, queue):
            self.sycl_queue = queue

    queue = FakeQueue()
    boosted._wait_sycl_queue(FakeData(queue))
    boosted._wait_sycl_queue(np.zeros(1))

    assert queue.wait_count == 1


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


def test_propagator_smearing_batches_all_spin_color_columns(monkeypatch):
    class FakeInfo:
        pass

    class FakePropagator:
        field_shape = [4, 4, 3, 3]

        def __init__(self, data):
            self.latt_info = FakeInfo()
            self.data = data

    class FakeComplex:
        def __init__(self, data):
            self.data = data

    source_data = np.ones((2, 2, 2, 2, 2, 4, 4, 3, 3), dtype=np.complex128)
    kernel_data = np.full((2, 2, 2, 2, 2), 2.0, dtype=np.complex128)
    source = FakePropagator(source_data.copy())
    events = []

    def fake_wait(field):
        events.append(("wait", field))

    def fake_fft(field, **kwargs):
        events.append(("fft", field))
        return field

    def fake_ifft(field, **kwargs):
        events.append(("ifft", field))
        return field

    monkeypatch.setattr(boosted, "LatticeFermion", type("FakeFermion", (), {}))
    monkeypatch.setattr(boosted, "_wait_sycl_queue", fake_wait)
    monkeypatch.setattr(boosted, "LatticePropagator", FakePropagator)
    monkeypatch.setattr(boosted, "fft", fake_fft)
    monkeypatch.setattr(boosted, "ifft", fake_ifft)
    monkeypatch.setattr(
        boosted,
        "_build_kernel_realspace_distributed",
        lambda *_args, **_kwargs: FakeComplex(kernel_data),
    )
    monkeypatch.setattr(boosted, "_get_xp_from_array", lambda _data: np)
    monkeypatch.setattr(boosted, "mpi_timer_print", lambda *_args, **_kwargs: None)

    result = boosted.boosted_smearing(source, w=9.0, boost=[0, 0, 0])

    assert result is source
    assert [name for name, _field in events] == [
        "wait", "fft", "wait", "fft", "wait", "ifft", "wait"
    ]
    np.testing.assert_allclose(result.data, 2.0)
