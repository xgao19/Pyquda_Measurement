import numpy as np

import pyquda_measurement_utils.bw_seq_pyquda as bw_module


class _FakePropagator:
    def __init__(self, data=None, latt_info=None):
        self.data = data
        self.latt_info = latt_info


def _epsilon(dtype):
    eps = np.zeros((3, 3, 3), dtype=dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1
    return eps


def _reference_down(data, gamma_matrix, projector):
    original_shape = data.shape
    flat = data.reshape(-1, 4, 4, 3, 3)
    gamma_transpose = gamma_matrix.T
    pdu = np.einsum("ij,...jiab->...ab", projector, flat)
    gtdg = np.einsum(
        "ij,...jkab,kl->...ilab", gamma_transpose, flat, gamma_matrix
    )
    gtd = np.einsum("ij,...jkab->...ikab", gamma_transpose, flat)
    pdg = np.einsum("ij,...jkab,kl->...ilab", projector, flat, gamma_matrix)
    eps = _epsilon(data.dtype)
    term1 = np.einsum("abc,def,...fc,...uveb->...uvad", eps, eps, pdu, gtdg)
    term2 = np.einsum("abc,def,...ujec,...jkfb->...ukad", eps, eps, gtd, pdg)
    return np.swapaxes(term2 - term1, -4, -3).reshape(original_shape)


def _reference_up(up_data, down_data, gamma_matrix, projector):
    original_shape = up_data.shape
    up_flat = up_data.reshape(-1, 4, 4, 3, 3)
    down_flat = down_data.reshape(-1, 4, 4, 3, 3)
    gamma_transpose = gamma_matrix.T
    gtdg = np.einsum(
        "ij,...jkab,kl->...ilab", gamma_transpose, down_flat, gamma_matrix
    )
    pdu = np.einsum("ij,...jkab->...ikab", projector, up_flat)
    dup = np.einsum("...jkab,kl->...jlab", up_flat, projector)
    trdup = np.einsum("...kjab,jk->...ab", up_flat, projector)
    eps = _epsilon(up_data.dtype)
    scalar = np.einsum("...mnbe,...mnad->...bead", gtdg, up_flat)
    r1_pre = np.einsum("abc,def,...bead->...cf", eps, eps, scalar)
    r1 = np.einsum("ij,...cf->...ijcf", projector, r1_pre)
    r2 = np.einsum("abc,def,...ad,...jibe->...ijcf", eps, eps, trdup, gtdg)
    r3 = np.einsum("abc,def,...ikad,...jkbe->...ijcf", eps, eps, pdu, gtdg)
    r4 = np.einsum("abc,def,...kiad,...klbe->...ilcf", eps, eps, gtdg, dup)
    result = -(((r1 + r2) + r3) + r4)
    return np.swapaxes(result, -1, -2).reshape(original_shape)


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _patch_numpy_backend(monkeypatch):
    monkeypatch.setattr(bw_module, "_get_xp_from_array", lambda _data: np)
    monkeypatch.setattr(
        bw_module,
        "_asarray_on_queue",
        lambda value, _xp, _reference: np.asarray(value),
    )
    monkeypatch.setattr(
        bw_module.core,
        "LatticePropagator",
        lambda latt_info: _FakePropagator(latt_info=latt_info),
    )


def test_up_insertion_matches_original_ordered_formula(monkeypatch):
    _patch_numpy_backend(monkeypatch)
    rng = np.random.default_rng(1234)
    shape = (2, 4, 4, 3, 3)
    up_data = _random_complex(rng, shape)
    down_data = _random_complex(rng, shape)
    gamma_matrix = _random_complex(rng, (4, 4))
    projector = _random_complex(rng, (4, 4))
    expected = _reference_up(up_data, down_data, gamma_matrix, projector)
    actual = bw_module.up_quark_insertion_pyquda(
        _FakePropagator(up_data, object()),
        _FakePropagator(down_data, object()),
        gamma_matrix,
        projector,
    ).data
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_down_insertion_matches_original_formula(monkeypatch):
    _patch_numpy_backend(monkeypatch)
    rng = np.random.default_rng(5678)
    shape = (2, 4, 4, 3, 3)
    data = _random_complex(rng, shape)
    gamma_matrix = _random_complex(rng, (4, 4))
    projector = _random_complex(rng, (4, 4))
    expected = _reference_down(data, gamma_matrix, projector)
    actual = bw_module.down_quark_insertion_pyquda(
        _FakePropagator(data, object()), gamma_matrix, projector
    ).data
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_wait_sycl_queue_and_numpy_noop():
    class Queue:
        waits = 0

        def wait(self):
            self.waits += 1

    queue = Queue()
    field = _FakePropagator(
        data=type("Array", (), {"sycl_queue": queue})()
    )
    bw_module._wait_sycl_queue(field)
    assert queue.waits == 1
    bw_module._wait_sycl_queue(np.zeros(1))
