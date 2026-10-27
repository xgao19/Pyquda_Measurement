"""Algebra and MPI ownership checks; no timing assertions or GPU imports."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft
from soft_factor_reference import adjoint, four_point, matrices, spin, tmdwf


class Prop:
    def __init__(self, data, shifts=None):
        self.data = data
        self.shifts = [] if shifts is None else shifts

    def copy(self):
        return Prop(self.data.copy(), self.shifts)

    def lexico(self, multi):
        assert multi is False
        return self.data

    def shift(self, steps, direction):
        self.shifts.append((steps, direction))
        return Prop(np.roll(self.data, -steps, axis=3 - direction), self.shifts)


def measurement():
    return soft.pion_soft_factor({
        'quark_mom': [[1, -1, 2]], 'bT_dir': [0, 1], 'bT_length': 3,
        'bz_length': 3, 'tsep_list': [6, 8],
        'pion_channel_pairs': {
            '5': (soft.G5, soft.G5),
            'XY': (soft._raw_gamma_by_label['X'], soft._raw_gamma_by_label['Y']),
        },
        'gamma_channel_pairs': {'5': ('5', '5'), 'XY': ('X', 'Y'), 'YX': ('Y', 'X')},
    })


@pytest.mark.parametrize('local', [True, False])
def test_shift_walk_matches_pyquda_sign_and_wraparound(monkeypatch, local):
    monkeypatch.setattr(soft, 'getGridSize', lambda: [1, 1, 1, 1] if local else [2, 2, 2, 1])
    monkeypatch.setattr(soft, 'getGridCoord', lambda: [0, 0, 0, 0])
    monkeypatch.setattr(soft, 'getRankFromCoord', lambda coord: 0)
    monkeypatch.setattr(soft, 'getMPIComm', lambda: SimpleNamespace(
        Sendrecv=lambda sendbuf, dest, recvbuf, source: np.copyto(recvbuf, sendbuf)))
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    for direction in range(3):
        prop = Prop(data.copy())
        for b, value in enumerate(soft._shifted_lexico(prop, 7, direction)):
            np.testing.assert_array_equal(value, np.roll(data, -b, axis=3 - direction))
        assert prop.shifts == []
        np.testing.assert_array_equal(prop.data, data)


@pytest.mark.parametrize('grid', [[1, 1, 1, 1], [1, 5, 4, 5]])
def test_full_contractions_match_matrix_traces_with_changing_inputs(monkeypatch, grid):
    rng = np.random.default_rng(410)
    shape = (2, 2, 2, 2, 4, 4, 3, 3)
    props = [Prop(rng.normal(size=shape) + 1j*rng.normal(size=shape)) for _ in range(4)]
    m = measurement()
    info = SimpleNamespace(size=[2, 2, 2, 2], global_size=[2, 2, 2, 2], mpi_rank=0)
    monkeypatch.setattr(soft, 'getGridSize', lambda: grid)
    # This algebra fixture represents identical periodic data on every rank.
    monkeypatch.setattr(soft, 'getGridCoord', lambda: [0, 0, 0, 0])
    monkeypatch.setattr(soft, 'getRankFromCoord', lambda coord: 0)
    monkeypatch.setattr(soft, 'getMPIComm', lambda: SimpleNamespace(
        Sendrecv=lambda sendbuf, dest, recvbuf, source: np.copyto(recvbuf, sendbuf)))
    monkeypatch.setattr(soft, 'mpi_print', lambda *args: None)
    monkeypatch.setattr(m, 'apply_phase', lambda prop, *args: prop.copy())
    reductions = []

    def reduce(local):
        reductions.append(local.shape)
        return local.copy()

    monkeypatch.setattr(soft, '_reduce_spatial_projected', reduce)
    for _ in range(2):
        before = [prop.data.copy() for prop in props]
        actual, plabels, glabels = m.contract_soft_factor(info, *props, [0, 0, 0])
        expected = four_point(m, [p.data for p in props], soft.G5, soft._raw_gamma_by_label, [0, 0, 0])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        assert plabels == list(m.pion_channel_pairs) and glabels == list(m.gamma_channel_pairs)
        for label, (src, _) in m.pion_channel_pairs.items():
            actual = m.contract_tmdwf_check(info, *props[:2], [0, 0, 0], label)
            reference = tmdwf(props[0].data, props[1].data, src, soft.G5,
                              list(soft._raw_gamma_by_label.values()), [0, 0, 0], m.bT_dir, 3, 3)
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-10)
        for prop, old in zip(props, before):
            np.testing.assert_array_equal(prop.data, old)
        # In-place source/sink changes must not hit a stale persistent cache.
        props[0].data *= 1.0 + 0.2j
        props[3].data *= 0.9 - 0.1j
    assert len(reductions) == 6


def test_prepared_matrix_blocks_match_independent_matrix_products():
    rng = np.random.default_rng(411)
    shape = (2, 1, 2, 3, 4, 4, 3, 3)
    left = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    gammas = rng.normal(size=(3, 4, 4)) + 1j*rng.normal(size=(3, 4, 4))
    g5 = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    a = soft._soft_factor_block_from_left(soft._prepare_soft_factor_left(left, gammas, g5), right.conj())
    b = soft._soft_factor_block_from_right(left, soft._prepare_soft_factor_right(gammas, g5, right.conj()))
    for index, gamma in enumerate(gammas):
        ref = matrices(left) @ spin(gamma) @ spin(g5) @ adjoint(matrices(right)) @ spin(g5)
        np.testing.assert_allclose(matrices(a[index]), ref, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(matrices(b[index]), ref, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fail', [False, True])
def test_wall_attributes_have_one_writer_and_collective_error(monkeypatch, tmp_path, rank, fail):
    events = []
    error = 'OSError: simulated attribute failure' if fail else None

    def broadcast(value, root):
        assert root == 0
        assert value == (error if rank == 0 else None)
        events.append('broadcast')
        return error

    comm = SimpleNamespace(Get_rank=lambda: rank, Barrier=lambda: events.append('barrier'), bcast=broadcast)
    monkeypatch.setattr(soft, 'getMPIComm', lambda: comm)
    path = tmp_path / 'prop.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('propagator', data=np.arange(4))
    real_open = h5py.File

    def open_file(*args, **kwargs):
        assert rank == 0
        events.append('attributes')
        if fail:
            raise OSError('simulated attribute failure')
        return real_open(*args, **kwargs)

    monkeypatch.setattr(soft.h5py, 'File', open_file)
    prop = SimpleNamespace(saveH5=lambda *args: events.append('collective_save'))
    if fail:
        with pytest.raises(RuntimeError, match='simulated attribute failure'):
            measurement().save_wall_propagator(prop, str(path.with_suffix('')), {'test': 7})
    else:
        measurement().save_wall_propagator(prop, str(path.with_suffix('')), {'test': 7})
        with real_open(path) as handle:
            np.testing.assert_array_equal(handle['propagator'][()], np.arange(4))
            if rank == 0:
                assert handle.attrs['test'] == 7
    assert events == ['collective_save', 'barrier'] + (['attributes'] if rank == 0 else []) + ['broadcast']
