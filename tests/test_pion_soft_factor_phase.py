import numpy as np


def spatial_phase(momentum, coords, lattice_size):
    momentum = np.asarray(momentum, dtype=int)
    coords = np.asarray(coords, dtype=int)
    lattice_size = np.asarray(lattice_size, dtype=int)
    angle = 2j * np.pi * np.sum(momentum * coords / lattice_size, axis=-1)
    return np.exp(angle)


def dominant_sink_momentum_after_soft_factor_phase(pion_momentum, lattice_size=(8, 8, 8)):
    axes = [np.arange(size) for size in lattice_size]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    sink_phase = spatial_phase([-2 * p for p in pion_momentum], coords, lattice_size)
    best_momentum = None
    best_overlap = -1.0
    for px in range(-lattice_size[0] // 2, lattice_size[0] // 2):
        for py in range(-lattice_size[1] // 2, lattice_size[1] // 2):
            for pz in range(-lattice_size[2] // 2, lattice_size[2] // 2):
                trial_phase = spatial_phase([px, py, pz], coords, lattice_size)
                overlap = abs(np.sum(sink_phase * trial_phase))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_momentum = [px, py, pz]
    return best_momentum


def test_soft_factor_sink_phase_selects_plus_two_pion_momentum():
    cases = [
        ([0, 0, 1], [0, 0, 2]),
        ([0, 0, -1], [0, 0, -2]),
        ([1, -1, 0], [2, -2, 0]),
    ]

    for pion_momentum, expected_momentum in cases:
        assert dominant_sink_momentum_after_soft_factor_phase(pion_momentum) == expected_momentum


def test_soft_factor_pion_momentum_from_quark_and_antiquark_momenta():
    quark_mom_fw = [0, 0, 2]
    quark_mom_bw = [0, 0, -1]
    pion_momentum = [fw + bw for fw, bw in zip(quark_mom_fw, quark_mom_bw)]

    assert pion_momentum == [0, 0, 1]
    assert dominant_sink_momentum_after_soft_factor_phase(pion_momentum) == [0, 0, 2]
