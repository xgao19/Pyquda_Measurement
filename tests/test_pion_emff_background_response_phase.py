import numpy as np


def spatial_phase(momentum, coords, lattice_size):
    momentum = np.asarray(momentum, dtype=int)
    coords = np.asarray(coords, dtype=int)
    lattice_size = np.asarray(lattice_size, dtype=int)
    angle = 2j * np.pi * np.sum(momentum * coords / lattice_size, axis=-1)
    return np.exp(angle)


def dominant_source_momentum_for_response(pf, qext, lattice_size=(8, 8, 8)):
    axes = [np.arange(size) for size in lattice_size]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    current_phase = spatial_phase(qext, coords, lattice_size)
    sink_phase = np.conjugate(spatial_phase(pf, coords, lattice_size))

    best_momentum = None
    best_overlap = -1.0
    for px in range(-lattice_size[0] // 2, lattice_size[0] // 2):
        for py in range(-lattice_size[1] // 2, lattice_size[1] // 2):
            for pz in range(-lattice_size[2] // 2, lattice_size[2] // 2):
                source_phase = spatial_phase([px, py, pz], coords, lattice_size)
                overlap = abs(np.sum(sink_phase * current_phase * source_phase))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_momentum = [px, py, pz]
    return best_momentum


def test_background_response_current_phase_selects_pi_equal_pf_minus_qext():
    cases = [
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 1], [0, 0, 1]),
        ([0, 0, 1], [0, 0, 2]),
        ([0, 0, 2], [0, 0, 4]),
    ]
    for pf, qext in cases:
        expected_pi = [pf_i - q_i for pf_i, q_i in zip(pf, qext)]
        assert dominant_source_momentum_for_response(pf, qext) == expected_pi
