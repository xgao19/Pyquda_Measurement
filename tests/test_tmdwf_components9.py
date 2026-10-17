"""Compact TMDWF algebra, directional halo exchange and periodic wrap tests."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft


def test_generic_gamma_components_match_full_bilinear():
    rng = np.random.default_rng(630)
    shape = (2, 3, 2, 4, 4, 4, 3, 3)
    fw, bw = [rng.normal(size=shape) + 1j*rng.normal(size=shape) for _ in range(2)]
    src = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    before = [v.copy() for v in (fw, bw)]
    g5 = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    left_c, forward_c = soft._tmdwf_components(fw, bw, src, g5)
    assert left_c.shape == forward_c.shape == shape[:4] + (3, 3)
    for direction in range(3):
        for b in (0, 1, 5):
            for z in (0, 1, 4):
                shifted = np.roll(np.roll(bw, -b, axis=3-direction), -z, axis=1)
                bar = np.einsum('ij,tzyxmlca,kl->tzyxkjca', g5, shifted.conj(), g5)
                left = np.einsum('ij,tzyxjlca->tzyxilca', src, bar)
                expected = np.einsum('tzyxjiab,tzyxilba->t', left, fw)
                compact = np.roll(np.roll(left_c, -b, axis=3-direction), -z, axis=1)
                actual = np.einsum('tzyxab,tzyxab->t', compact, forward_c)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    for value, old in zip((fw, bw), before):
        np.testing.assert_array_equal(value, old)


@pytest.mark.parametrize('direction', [0, 1, 2])
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('components', [(3, 3), (4, 4, 3, 3)])
def test_lexico_halo_against_global_field(monkeypatch, direction, distributed, components):
    grid = (2, 3, 2, 2) if distributed else (1, 1, 1, 2)
    local = (2, 2, 3, 2)
    shape = tuple(n*g for n, g in zip(local, grid[::-1])) + components
    rng = np.random.default_rng(631)
    global_field = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    axis = 3-direction
    monkeypatch.setattr(soft, 'getGridSize', lambda: grid)
    monkeypatch.setattr(soft, 'getRankFromCoord', lambda c: np.ravel_multi_index(tuple(c), grid))

    def tile(field, coord):
        return field[tuple(slice(c*n, (c+1)*n) for c, n in zip(coord[::-1], local))]

    for coord in np.ndindex(grid):
        monkeypatch.setattr(soft, 'getGridCoord', lambda: coord)
        current = tile(global_field, coord).copy()
        original = current.copy()
        walk = soft._shifted_lexico_array(current, 7, direction)
        input_field = next(walk)
        assert input_field is current
        seen_buffers = set()
        for step in range(7):
            before = current.copy()
            full_current = np.roll(global_field, -step, axis=axis)
            events = []

            def exchange(sendbuf, dest, recvbuf, source):
                lower, upper = list(coord), list(coord)
                lower[direction] = (coord[direction]-1) % grid[direction]
                upper[direction] = (coord[direction]+1) % grid[direction]
                assert dest == np.ravel_multi_index(tuple(lower), grid)
                assert source == np.ravel_multi_index(tuple(upper), grid)
                assert sendbuf.flags.c_contiguous
                np.testing.assert_array_equal(sendbuf, np.take(before, 0, axis=axis))
                recvbuf[:] = np.take(tile(full_current, upper), 0, axis=axis)
                events.append(True)

            monkeypatch.setattr(soft, 'getMPIComm', lambda: SimpleNamespace(Sendrecv=exchange))
            current = next(walk)
            seen_buffers.add(id(current))
            expected = tile(np.roll(global_field, -(step+1), axis=axis), coord)
            np.testing.assert_array_equal(current, expected)
            np.testing.assert_array_equal(input_field, original)
            np.testing.assert_array_equal(before, tile(full_current, coord))
            assert len(events) == int(distributed)
        assert len(seen_buffers) == 2
        with pytest.raises(StopIteration):
            next(walk)


@pytest.mark.parametrize('length', [0, 1, 6])
def test_nested_walks_preserve_transverse_state(monkeypatch, length):
    monkeypatch.setattr(soft, 'getGridSize', lambda: [1, 1, 1, 1])
    data = np.arange(2*3*4*5*9).reshape(2, 3, 4, 5, 3, 3)
    before = data.copy()
    for b, transverse in enumerate(soft._shifted_lexico_array(data, length, 1)):
        saved = transverse.copy()
        for z, shifted in enumerate(soft._shifted_lexico_array(transverse, length, 2)):
            np.testing.assert_array_equal(shifted, np.roll(np.roll(data, -b, axis=2), -z, axis=1))
        np.testing.assert_array_equal(transverse, saved)
    np.testing.assert_array_equal(data, before)
