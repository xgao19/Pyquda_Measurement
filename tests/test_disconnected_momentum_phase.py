import numpy as np


def _coords(lattice_size):
    axes = [np.arange(size) for size in lattice_size]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _momentum_phase(momentum, coords, lattice_size, origin=(0, 0, 0)):
    momentum = np.asarray(momentum, dtype=int)
    lattice_size = np.asarray(lattice_size, dtype=int)
    origin = np.asarray(origin, dtype=int)
    angle = 2j * np.pi * np.sum(momentum * (coords - origin) / lattice_size, axis=-1)
    return np.exp(angle)


def _project(momentum, field, coords, lattice_size, origin=(0, 0, 0)):
    return np.sum(_momentum_phase(momentum, coords, lattice_size, origin=origin) * field)


def test_disconnected_loop_phase_selects_negative_plane_wave():
    lattice_size = (8, 8, 8)
    coords = _coords(lattice_size)
    q_target = (1, -2, 3)
    source_origin = (0, 0, 0)
    volume = np.prod(lattice_size)

    field = np.conjugate(_momentum_phase(q_target, coords, lattice_size, origin=source_origin))

    selected = _project(q_target, field, coords, lattice_size, origin=source_origin)
    rejected = _project(tuple(-q for q in q_target), field, coords, lattice_size, origin=source_origin)

    np.testing.assert_allclose(selected, volume, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rejected, 0.0, atol=1e-12, rtol=0.0)


def test_disconnected_loop_phase_origin_shift():
    lattice_size = (8, 8, 8)
    coords = _coords(lattice_size)
    q_target = (0, 1, -2)
    origin = (2, 3, 1)
    volume = np.prod(lattice_size)

    field = np.conjugate(_momentum_phase(q_target, coords, lattice_size, origin=origin))

    selected = _project(q_target, field, coords, lattice_size, origin=origin)
    wrong_origin = _project(q_target, field, coords, lattice_size, origin=(0, 0, 0))

    np.testing.assert_allclose(selected, volume, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(abs(wrong_origin), volume, atol=1e-12, rtol=1e-12)
    assert not np.allclose(wrong_origin, selected)


if __name__ == "__main__":
    test_disconnected_loop_phase_selects_negative_plane_wave()
    test_disconnected_loop_phase_origin_shift()
    print("[disconnected q != 0 momentum phase sanity check]")
    print("phase(q,x) = exp(+2*pi*i*q.(x-x0)/L)")
    print("projection at q selects local_loop momentum -q")
