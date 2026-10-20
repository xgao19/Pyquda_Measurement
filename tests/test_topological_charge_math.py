"""CPU identities fixing the local normalization, orientation and phases."""

from itertools import permutations
import numpy as np
import pytest
from pyquda_measurement_utils.topological_charge_1pt import topological_density
import pyquda_measurement_utils.flowed_gauge_utils as gauge_utils


def random_fields(seed=42, shape=(7,)):
    rng = np.random.default_rng(seed)
    fields = [[None] * 4 for _ in range(4)]
    for mu in range(4):
        for nu in range(mu + 1, 4):
            a = rng.normal(size=shape + (3, 3)) + 1j * rng.normal(size=shape + (3, 3))
            a = a + a.swapaxes(-1, -2).conj()
            a -= np.trace(a, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            fields[mu][nu], fields[nu][mu] = a, -a
    return fields


def epsilon_reference(fields):
    result = 0
    for p in permutations(range(4)):
        sign = (-1)**sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4))
        result += sign * np.einsum("...ab,...ba->...", fields[p[0]][p[1]], fields[p[2]][p[3]])
    return result.real / (32 * np.pi**2)


def test_epsilon_reduction_and_normalization():
    fields = random_fields()
    np.testing.assert_allclose(topological_density(fields), epsilon_reference(fields), atol=1e-14)
    zero = [[np.zeros((3, 3), complex) for _ in range(4)] for _ in range(4)]
    h = np.diag([0.5, -0.5, 0.])
    zero[0][1] = zero[2][3] = h
    np.testing.assert_allclose(topological_density(zero), 1 / (8 * np.pi**2))


def test_color_gauge_invariance_and_reflection():
    fields = random_fields()
    rng = np.random.default_rng(11)
    omega, _ = np.linalg.qr(rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3)))
    transformed = [[None if a is None else omega @ a @ omega.swapaxes(-1, -2).conj()
                    for a in row] for row in fields]
    np.testing.assert_allclose(topological_density(transformed), topological_density(fields), atol=1e-14)
    reflected = [[None if a is None else (-a if (mu == 0) != (nu == 0) else a)
                  for nu, a in enumerate(row)] for mu, row in enumerate(fields)]
    np.testing.assert_allclose(topological_density(reflected), -topological_density(fields), atol=1e-14)


@pytest.mark.parametrize("identity", [False, True])
def test_clover_projection(monkeypatch, identity):
    rng = np.random.default_rng(17)
    raw = np.broadcast_to(4 * np.eye(3), (4, 2, 3, 3)).astype(complex).copy()
    if not identity:
        raw += rng.normal(size=raw.shape) + 1j * rng.normal(size=raw.shape)

    class Field:
        data = raw.copy()
        location = "numpy"

    class Gauge:
        def loop(self, paths, coeff):
            assert paths[0] == [[0, 1, 4, 5], [1, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 4]]
            assert coeff == [1.] * 4
            return Field()

    monkeypatch.setattr(gauge_utils, "arrayIdentity", lambda n, dtype, location: np.eye(n, dtype=dtype))
    h = gauge_utils.clover_traceless(Gauge(), 0, 1).data
    anti = (raw - raw.swapaxes(-1, -2).conj()) / 8
    expected = -1j * (anti - np.trace(anti, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3)
    np.testing.assert_allclose(h, expected, atol=1e-14)
    np.testing.assert_allclose(h, h.swapaxes(-1, -2).conj(), atol=1e-14)
    np.testing.assert_allclose(np.trace(h, axis1=-2, axis2=-1), 0, atol=1e-14)
    if identity:
        np.testing.assert_array_equal(h, 0)


def test_fourier_source_alignment_and_total():
    from application.analysis_helper.emt_disconnected_analysis import source_relative_loops
    rng = np.random.default_rng(4)
    density = rng.normal(size=(6, 4, 4, 4))
    z, y, x = np.indices((4, 4, 4))
    momenta = np.array([[0, 0, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0]])
    phases = np.exp(2j * np.pi * (momenta[:, 0, None, None, None] * x +
                                 momenta[:, 1, None, None, None] * y) / 4)
    loop = np.einsum("qzyx,tzyx->qt", phases, density)[:, None, :] / 64
    np.testing.assert_allclose(loop[2], loop[1].conj())
    np.testing.assert_allclose(64 * loop[0].sum(), density.sum())
    source = np.array([1, 2, 3, 4])
    aligned = source_relative_loops(loop, momenta, [source], [4, 4, 4, 6], q_axis=0)
    expected = np.roll(loop, -source[3], axis=-1) * np.exp(-2j*np.pi*(momenta[:, :3] @ source[:3])/4)[:, None, None]
    np.testing.assert_allclose(aligned[0], expected, atol=1e-14)
