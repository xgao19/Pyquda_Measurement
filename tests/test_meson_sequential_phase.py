import numpy as np


def spatial_phase(momentum, coords, lattice_size):
    momentum = np.asarray(momentum, dtype=int)
    coords = np.asarray(coords, dtype=int)
    lattice_size = np.asarray(lattice_size, dtype=int)
    angle = 2j * np.pi * np.sum(momentum * coords / lattice_size, axis=-1)
    return np.exp(angle)


def gamma5_hermitian_sink_gamma(gamma5, sink_gamma):
    return gamma5 @ sink_gamma.conj().T @ gamma5


def test_meson_sequential_gamma_source_uses_gamma5_hermitian_sink():
    gamma5 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    sink_gamma = np.array(
        [
            [0, 1 + 2j, 0, 0],
            [3 - 1j, 0, 0, 0],
            [0, 0, 0, -2j],
            [0, 0, 4, 0],
        ],
        dtype=np.complex128,
    )

    expected = gamma5 @ sink_gamma.conj().T @ gamma5
    actual = gamma5_hermitian_sink_gamma(gamma5, sink_gamma)

    np.testing.assert_array_equal(actual, expected)


def test_meson_sequential_rhs_phase_becomes_conjugate_sink_phase_after_backward_line():
    lattice_size = (8, 8, 8)
    axes = [np.arange(size) for size in lattice_size]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pf = [0, 0, 2]

    rhs_phase = spatial_phase(pf, coords, lattice_size)
    backward_line_sink_phase = rhs_phase.conj()
    direct_sink_factor = np.conjugate(spatial_phase(pf, coords, lattice_size))

    np.testing.assert_allclose(backward_line_sink_phase, direct_sink_factor, atol=0, rtol=0)


def test_meson_sequential_sink_phase_selects_final_momentum_pf():
    lattice_size = (8, 8, 8)
    axes = [np.arange(size) for size in lattice_size]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pf = [0, 0, 2]
    sink_factor = np.conjugate(spatial_phase(pf, coords, lattice_size))

    best_momentum = None
    best_overlap = -1.0
    for px in range(-lattice_size[0] // 2, lattice_size[0] // 2):
        for py in range(-lattice_size[1] // 2, lattice_size[1] // 2):
            for pz in range(-lattice_size[2] // 2, lattice_size[2] // 2):
                trial = spatial_phase([px, py, pz], coords, lattice_size)
                overlap = abs(np.sum(sink_factor * trial))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_momentum = [px, py, pz]

    assert best_momentum == pf
