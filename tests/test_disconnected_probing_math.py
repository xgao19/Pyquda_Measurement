"""Independent geometry and algebra checks; no GPU or Dirac solver required."""

from itertools import product

import numpy as np
import pytest

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    NESTED_HP_ORDERING,
    hierarchical_probe_pattern,
    hierarchical_nested_index,
    validate_nested_hp_shape,
    validate_multiplier_plan,
    multiplier_probe_pattern,
    broadcast_multiplier_plan,
    vectors_per_base,
    effective_n_inversions,
)


class Grid:
    def __init__(self, shape=(8, 8, 8, 8)):
        self.global_size = shape

    def coordinate(self, mu=None):
        coords = np.indices(self.global_size, dtype=np.int64)
        return coords if mu is None else coords[mu]


def reference_colors(grid, q):
    coords = grid.coordinate()
    residues = sum((coords[mu] % q) * q**mu for mu in range(4))
    return residues + q**4 * (sum(coord // q for coord in coords) % 2)


def assert_periodic_coloring(colors, distance):
    shape = colors.shape
    # Enumerate actual wrapped neighbors independently of production checkers.
    ranges = [range(-min(distance, L // 2), min(distance, L // 2) + 1) for L in shape]
    for delta in product(*ranges):
        length = sum(min(abs(dx), L - abs(dx)) for dx, L in zip(delta, shape))
        if 0 < length <= distance:
            assert np.all(colors != np.roll(colors, delta, axis=(0, 1, 2, 3))), delta


def test_nested_complete_levels_match_independent_coloring():
    grid = Grid()
    label = hierarchical_nested_index(grid)
    for count, q in ((2, 1), (32, 2), (512, 4)):
        actual = label % count
        reference = reference_colors(grid, q)
        pairs = np.unique(np.stack((actual.ravel(), reference.ravel()), axis=1), axis=0)
        assert len(pairs) == count
        assert len(np.unique(pairs[:, 0])) == count
        assert len(np.unique(pairs[:, 1])) == count
        assert_periodic_coloring(actual, 2 * q - 1)


def test_complete_walsh_orthogonality():
    grid = Grid()
    label = hierarchical_nested_index(grid)
    for count in (2, 32, 512):
        _, representatives = np.unique(label.ravel() % count, return_index=True)
        patterns = np.asarray([
            hierarchical_probe_pattern(grid, k, NESTED_HP_ORDERING).ravel()[representatives]
            for k in range(count)
        ])
        np.testing.assert_array_equal(patterns.T @ patterns, count * np.eye(count))


def test_legacy_complete_counts_have_short_distance_collisions():
    grid = Grid()
    # Independent reference for the retired interleaved ordering on 8^4.
    label = sum(((coord >> bit) & 1) << (4 * bit + mu)
                for bit in range(3) for mu, coord in enumerate(grid.coordinate()))
    for count, offset in ((2, 1), (32, 2), (512, 4)):
        assert label[0, 0, 0, 0] % count == label[0, offset, 0, 0] % count


def test_nested_boundaries():
    for count in (2, 32, 512):
        validate_nested_hp_shape(Grid().global_size, count)
        validate_nested_hp_shape((80, 80, 80, 80), count)
    for count in (0, 1, 4, 16, 256, 8192, 2.0, True):
        with pytest.raises((TypeError, ValueError)):
            validate_nested_hp_shape(Grid().global_size, count)
    for shape in ((8, 8, 8), (1, 8, 8, 8), (8, 8, 8, 9)):
        with pytest.raises(ValueError):
            validate_nested_hp_shape(shape, 2)
    with pytest.raises(ValueError):
        validate_nested_hp_shape((4, 4, 4, 4), 512)
    for index in (-1, 512, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            hierarchical_probe_pattern(Grid(), index, NESTED_HP_ORDERING)
    with pytest.raises(ValueError):
        hierarchical_probe_pattern(Grid((3, 4, 4, 4)), 0, NESTED_HP_ORDERING)


def test_nested_complete_levels_on_80_torus():
    class CoordinateList:
        global_size = (80, 80, 80, 80)

        def __init__(self, coordinates):
            self.coordinates = np.asarray(coordinates, dtype=np.int64).T

        def coordinate(self, mu=None):
            return self.coordinates if mu is None else self.coordinates[mu]

    for count, q in ((2, 1), (32, 2), (512, 4)):
        period = 2 * q
        points = np.asarray(list(product(range(period), repeat=4)), dtype=np.int64)
        labels = hierarchical_nested_index(CoordinateList(points)) % count
        assert len(np.unique(labels)) == count
        distance = period - 1
        for delta in product(range(-distance, distance + 1), repeat=4):
            if not 0 < sum(map(abs, delta)) <= distance:
                continue
            neighbors = (points + np.asarray(delta)) % 80
            neighbor_labels = hierarchical_nested_index(CoordinateList(neighbors)) % count
            assert np.all(labels != neighbor_labels), (count, delta)


def test_small_matrix_shared_noise_complete_basis():
    rng = np.random.default_rng(17)
    grid = Grid((2, 2, 2, 2))
    colors = reference_colors(grid, 1).ravel()
    A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    A *= (colors[:, None] != colors[None, :]) | np.eye(16, dtype=bool)
    xi = rng.choice([1, -1, 1j, -1j], size=16)
    estimates = []
    for k in range(2):
        eta = xi * hierarchical_probe_pattern(grid, k, NESTED_HP_ORDERING).ravel()
        estimates.append(eta.conj() @ A @ eta)
    np.testing.assert_allclose(np.mean(estimates), np.trace(A), atol=1e-13)


def plan(shape=(8, 8, 8, 8), distance=2, colors=16, sigma=(1, 2, 3, 4)):
    return dict(shape=list(shape), distance=distance, colors=colors, sigma=list(sigma), version=1)


def test_multiplier_fixed_plans_and_wrap_rejection():
    for candidate in (plan(distance=1, colors=2, sigma=(1, 1, 1, 1)), plan()):
        verified = validate_multiplier_plan(candidate)
        coords = Grid().coordinate()
        colors = sum(s*c for s, c in zip(candidate["sigma"], coords)) % candidate["colors"]
        assert_periodic_coloring(colors, candidate["distance"])
        assert verified == candidate
    large = plan((64,)*4, 4, 64, (1, 4, 10, 17))
    validate_multiplier_plan(large)
    # Divisibility makes the independent signed-displacement check sufficient.
    assert all(s*L % 64 == 0 for s, L in zip(large["sigma"], large["shape"]))
    for delta in product(range(-4, 5), repeat=4):
        if 0 < sum(abs(dx) for dx in delta) <= 4:
            assert sum(s*dx for s, dx in zip(large["sigma"], delta)) % 64 != 0
    with pytest.raises(ValueError, match="periodic"):
        validate_multiplier_plan(plan(colors=9))
    with pytest.raises(ValueError, match="shape"):
        validate_multiplier_plan(plan(), (16,)*4)
    for updates in ({"version": 2}, {"distance": 0}, {"colors": 2.0}, {"sigma": [1, 2]}):
        with pytest.raises((TypeError, ValueError)):
            validate_multiplier_plan(dict(plan(), **updates))
    with pytest.raises(ValueError):
        vectors_per_base("multiplier_probing", 1)
    assert effective_n_inversions(3, "multiplier_probing", 1, plan()) == 48


def test_multiplier_fourier_orthogonality_and_checkerboard_identity():
    grid = Grid()
    colors = reference_colors(grid, 1)
    m1 = plan(distance=1, colors=2, sigma=(1, 1, 1, 1))
    for k in range(2):
        np.testing.assert_allclose(multiplier_probe_pattern(grid, k, m1),
                                   hierarchical_probe_pattern(grid, k, NESTED_HP_ORDERING), atol=1e-14)
    m2 = plan()
    labels = sum(s*c for s, c in zip(m2["sigma"], grid.coordinate())) % 16
    _, representatives = np.unique(labels.ravel(), return_index=True)
    patterns = np.asarray([multiplier_probe_pattern(grid, k, m2).ravel()[representatives] for k in range(16)])
    np.testing.assert_allclose(patterns.conj().T @ patterns / 16, np.eye(16), atol=1e-14)
    for idx in (-1, 16, 1.5):
        with pytest.raises((TypeError, ValueError)):
            multiplier_probe_pattern(grid, idx, m2)


def test_fourier_average_equals_mask_sum_and_exact_noise_expectation():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    colors = np.array([0, 1, 0, 1])
    patterns = np.exp(2j*np.pi*np.arange(2)[:, None]*colors[None, :]/2)
    means = []
    for noise in product((1, -1, 1j, -1j), repeat=4):
        xi = np.array(noise)
        fourier = np.mean([(p*xi).conj() @ A @ (p*xi) for p in patterns])
        masks = sum((xi*(colors == c)).conj() @ A @ (xi*(colors == c)) for c in range(2))
        np.testing.assert_allclose(fourier, masks, atol=1e-13)
        means.append(fourier)
    np.testing.assert_allclose(np.mean(means), np.trace(A), atol=1e-13)


def test_plan_broadcast_including_root_failure(tmp_path):
    class Root:
        Get_rank = staticmethod(lambda: 0)
        bcast = staticmethod(lambda value, root=0: value)
    assert broadcast_multiplier_plan(plan(), Root()) == plan()
    with pytest.raises(ValueError, match="initialization failed"):
        broadcast_multiplier_plan(tmp_path / "missing.json", Root())


def test_m6_8torus_all_same_color_pairs():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "application/EMT_disconnected_1pt/probing_plans/m6_8x8x8x8.json"
    candidate = json.loads(path.read_text())
    assert candidate["colors"] == 411 and candidate["distance"] == 6
    validate_multiplier_plan(candidate)
    coords = np.indices((8,)*4).reshape(4, -1).T
    colors = coords @ np.asarray(candidate["sigma"]) % candidate["colors"]
    assert len(np.unique(colors)) == 411
    nearest = 17
    for color in np.unique(colors):
        points = coords[colors == color]
        delta = abs(points[:, None, :] - points[None, :, :])
        distances = np.minimum(delta, 8-delta).sum(axis=2)
        np.fill_diagonal(distances, 99)
        nearest = min(nearest, int(distances.min()))
    assert nearest == 7
    ball = coords[np.minimum(coords, 8-coords).sum(axis=1) <= 3]
    assert len(ball) == 129
    delta = abs(ball[:, None, :] - ball[None, :, :])
    assert np.minimum(delta, 8-delta).sum(axis=2).max() == 6


def test_m8_8torus_all_same_color_pairs():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "application/EMT_disconnected_1pt/probing_plans/m8_8x8x8x8.json"
    candidate = json.loads(path.read_text())
    assert candidate["colors"] == 829 and candidate["distance"] == 8
    validate_multiplier_plan(candidate)
    coords = np.indices((8,)*4).reshape(4, -1).T
    colors = coords @ np.asarray(candidate["sigma"]) % candidate["colors"]
    assert len(np.unique(colors)) == 829
    nearest = 17
    for color in np.unique(colors):
        points = coords[colors == color]
        delta = abs(points[:, None, :] - points[None, :, :])
        distances = np.minimum(delta, 8-delta).sum(axis=2)
        np.fill_diagonal(distances, 99)
        nearest = min(nearest, int(distances.min()))
    assert nearest == 9
    ball = coords[np.minimum(coords, 8-coords).sum(axis=1) <= 4]
    assert len(ball) == 317
    delta = abs(ball[:, None, :] - ball[None, :, :])
    assert np.minimum(delta, 8-delta).sum(axis=2).max() == 8
