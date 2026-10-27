"""Production contractions checked against independent 12x12 matrix algebra."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft
from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS, PHYSICAL_FROM_PYQUDA
from soft_factor_reference import (
    adjoint, field_from_matrices, four_point, matrices, phased, spin,
    tmdwf, trace_spatial, wall_2pt,
)


class Prop:
    def __init__(self, data):
        self.data = data
        self.latt_info = SimpleNamespace()

    def copy(self):
        return Prop(self.data.copy())

    def lexico(self, force_numpy):
        assert force_numpy is False
        return self.data


def setup(monkeypatch, shape=(2, 2, 3, 4), seed=170):
    rng = np.random.default_rng(seed)
    fields = [rng.normal(size=shape + (4, 4, 3, 3))
              + 1j * rng.normal(size=shape + (4, 4, 3, 3)) for _ in range(4)]
    arbitrary = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(2)]
    m = soft.pion_soft_factor({
        'quark_mom': [[1, -1, 2], [-1, 0, 1]],
        'bT_dir': [0, 1, 2], 'bT_length': 2, 'bz_length': 2, 'tsep_list': [2],
        'pion_channel_pairs': {**soft.soft_factor_pion_channel_pairs,
                               '5': (soft.G5, soft.G5), 'arbitrary': tuple(arbitrary)},
        'gamma_channel_pairs': {**soft.soft_factor_gamma_channel_pairs,
                                'XY': ('X', 'Y'), 'YX': ('Y', 'X')},
    })
    monkeypatch.setattr(soft, 'getGridSize', lambda: [1, 1, 1, 1])
    monkeypatch.setattr(soft, 'mpi_print', lambda *args: None)
    monkeypatch.setattr(soft, '_reduce_spatial_projected', lambda a: a.copy())
    monkeypatch.setattr(m, 'apply_phase', lambda p, k, sign: Prop(phased(p.data, [sign*v for v in k])))
    info = SimpleNamespace(size=list(shape[::-1]), global_size=list(shape[::-1]), mpi_rank=0)
    return m, info, [Prop(f) for f in fields]


@pytest.mark.parametrize('seed', [170, 171, 172])
def test_all_contractions_match_independent_traces_with_momenta_and_shifts(monkeypatch, seed):
    m, info, props = setup(monkeypatch, seed=seed)
    fields = [p.data for p in props]
    before = [f.copy() for f in fields]
    kf, kb = [1, -1, 2], [-1, 0, 1]
    momentum = [a+b for a, b in zip(kf, kb)]
    for label, (src, sink) in m.pion_channel_pairs.items():
        actual = m.contract_wall_2pt(info, *props[:2], kf, kb, label)
        expected = wall_2pt(*fields[:2], src, sink, soft.G5, kf, kb)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        actual = m.contract_tmdwf_check(info, *props[:2], momentum, label)
        expected = tmdwf(*fields[:2], src, soft.G5, list(soft._raw_gamma_by_label.values()),
                         momentum, m.bT_dir, m.bT_length, m.bz_length)
        assert actual.shape == (16, 27, info.global_size[3])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
    actual, plabels, glabels = m.contract_soft_factor(info, *props, momentum)
    expected = four_point(m, fields, soft.G5, soft._raw_gamma_by_label, momentum)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    assert plabels == list(m.pion_channel_pairs) and glabels == list(m.gamma_channel_pairs)
    for value, old in zip(fields, before):
        np.testing.assert_array_equal(value, old)


def test_ss_preserves_cross_site_and_cross_rank_products(monkeypatch):
    m, info, props = setup(monkeypatch, shape=(2, 1, 1, 4))
    expected = wall_2pt(*[p.data for p in props[:2]], soft.G5, soft.G5, soft.G5, [0]*3, [0]*3)
    local = [Prop(p.data[:, :, :, :2]) for p in props[:2]]
    remote = [p.data[:, :, :, 2:] for p in props[:2]]
    info.size[0] = 2
    calls = []

    def reduce_open_lines(value):
        calls.append(value.shape)
        other = np.stack([f.sum(axis=(1, 2, 3)).transpose(1, 2, 3, 4, 0) for f in remote])
        return value + other

    monkeypatch.setattr(soft, '_reduce_spatial_projected', reduce_open_lines)
    actual = m.contract_wall_2pt(info, *local, [0]*3, [0]*3, '5')
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    assert calls == [(2, 4, 4, 3, 3, 2)]
    fw, bw = [matrices(p.data) for p in props[:2]]
    sp = trace_spatial(adjoint(bw) @ fw)
    assert not np.allclose(expected, sp)


@pytest.mark.parametrize('diagonal', [[1, 1, 1, 1], [1, -1, 1, -1]])
def test_gamma5_two_point_is_exact_norm_not_independent_spin_sums(monkeypatch, diagonal):
    m, info, _ = setup(monkeypatch, shape=(1, 1, 1, 1))
    matrix = np.kron(np.diag(diagonal), np.eye(3)).astype(complex)
    prop = Prop(field_from_matrices(matrix[None, None, None, None]))
    actual = m.contract_wall_2pt(info, prop, prop, [0]*3, [0]*3, '5')
    np.testing.assert_allclose(actual, [12], rtol=0, atol=0)
    sp = m.contract_tmdwf_check(info, prop, prop, [0]*3, '5')
    np.testing.assert_allclose(sp[GAMMA_LABELS.index('5'), 0], [12], rtol=0, atol=0)


def test_four_point_offdiagonal_color_closes_matrix_trace(monkeypatch):
    m, info, _ = setup(monkeypatch, shape=(1, 1, 1, 1))
    m.pion_channel_pairs = {'I': (np.eye(4), np.eye(4))}
    m.gamma_channel_pairs = {'I': ('I', 'I')}
    color = np.zeros((3, 3))
    color[0, 1] = 1
    offdiag = np.kron(np.eye(4), color).astype(complex)
    identity = np.eye(12, dtype=complex)
    props = [Prop(field_from_matrices(v[None, None, None, None]))
             for v in (offdiag, identity, offdiag, identity)]
    result = m.contract_soft_factor(info, *props, [0]*3)[0]
    np.testing.assert_allclose(result, 0, rtol=0, atol=0)


def test_four_point_is_invariant_under_complex_global_color_rotation(monkeypatch):
    m, info, props = setup(monkeypatch)
    baseline = m.contract_soft_factor(info, *props, [0]*3)[0]
    rng = np.random.default_rng(173)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))
    u *= np.linalg.det(u)**(-1/3)
    rotation = np.kron(np.eye(4), u)
    rotated = [Prop(field_from_matrices(rotation @ matrices(p.data) @ rotation.conj().T)) for p in props]
    result = m.contract_soft_factor(info, *rotated, [0]*3)[0]
    np.testing.assert_allclose(result, baseline, rtol=1e-12, atol=1e-9)


def test_tmdwf_raw_basis_maps_to_gpt_physical_sink_channels(monkeypatch):
    m, info, props = setup(monkeypatch)
    actual = m.contract_tmdwf_check(info, *props[:2], [0]*3, '5')
    physical = (PHYSICAL_FROM_PYQUDA @ actual.reshape(16, -1)).reshape(actual.shape)
    physical_sinks = list(soft._raw_gamma_by_label.values())
    for label, axis in [('T5', 'T'), ('Y5', 'Y')]:
        physical_sinks[GAMMA_LABELS.index(label)] = soft._raw_gamma_by_label[axis] @ soft.G5
    expected = tmdwf(*[p.data for p in props[:2]], soft.G5, soft.G5,
                     physical_sinks, [0]*3, m.bT_dir, m.bT_length, m.bz_length)
    np.testing.assert_allclose(physical, expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('sign', [1, -1])
def test_apply_phase_preserves_input_and_checkerboard_axes(monkeypatch, sign):
    rng = np.random.default_rng(174)
    data = rng.normal(size=(2, 2, 2, 2, 2, 4, 4, 3, 3)).astype(complex)
    prop = Prop(data)
    original = data.copy()
    phase = np.exp(1j * rng.normal(size=data.shape[:5]))
    calls = []

    def get_phase(momentum, x0):
        calls.append((momentum, x0))
        return phase

    monkeypatch.setattr(soft.phase, 'MomentumPhase',
                        lambda info: SimpleNamespace(getPhase=get_phase))
    m = soft.pion_soft_factor({'quark_mom': [[0]*3], 'bT_dir': [0],
                              'bT_length': 0, 'tsep_list': [1]})
    result = m.apply_phase(prop, [1, -1, 2], sign=sign, x0=[1, 2, 3, 0])
    expected_phase = phase if sign == 1 else phase.conj()
    np.testing.assert_allclose(result.data, original * expected_phase[..., None, None, None, None])
    np.testing.assert_array_equal(prop.data, original)
    assert calls == [([1, -1, 2], [1, 2, 3, 0])]
