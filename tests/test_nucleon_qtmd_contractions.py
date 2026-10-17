"""CPU algebra and halo bookkeeping; GPU/MPI validation is separate."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyquda_measurement_utils import nucleon_qtmd_contractions as kernel
from pyquda_measurement_utils.qtmd_operator_utils import (
    create_cg_qtmd_wilsonline_index_lists, create_pdf_wilsonline_index_list,
)


class Info:
    size = [4, 2, 2, 2]
    global_size = size

    def evenodd(self, data, multi=False):
        if not multi:
            return self.evenodd(data[None], True)[0]
        parity = np.indices(self.size[::-1]).sum(axis=0).ravel() % 2
        flat = data.reshape(data.shape[0], -1, *data.shape[5:])
        return np.stack([flat[:, parity == p] for p in (0, 1)], axis=1).reshape(
            data.shape[0], 2, *self.size[:0:-1], self.size[0] // 2, *data.shape[5:])

    def lexico(self, data, multi=False, backend="numpy"):
        if not multi:
            return self.lexico(data[None], True)[0]
        parity = np.indices(self.size[::-1]).sum(axis=0).ravel() % 2
        flat = data.reshape(data.shape[0], 2, -1, *data.shape[6:])
        out = np.empty((data.shape[0], len(parity), *data.shape[6:]), dtype=data.dtype)
        for p in (0, 1):
            out[:, parity == p] = flat[:, p]
        return out.reshape(data.shape[0], *self.size[::-1], *data.shape[6:])


class Prop:
    def __init__(self, info, lexico):
        self.info = info
        self.data = info.evenodd(lexico)

    def lexico(self, force_numpy=False):
        return self.info.lexico(self.data)

    def copy(self):
        return Prop(self.info, self.lexico())


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(kernel, "getGridCoord", lambda: [0] * 4)
    monkeypatch.setattr(kernel, "getGridSize", lambda: [1] * 4)
    monkeypatch.setattr(kernel, "getMPIComm", lambda: SimpleNamespace(Get_rank=lambda: 0))
    monkeypatch.setattr(kernel, "mpi_print", lambda *a: None)


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", [-1, 1])
def test_local_shift(local, direction, sign):
    field = np.arange(2 * 4 * 6 * 8 * 3).reshape(2, 4, 6, 8, 3).astype(complex)
    out = np.empty_like(field)
    kernel._shift_one(field, out, direction, sign, {})
    np.testing.assert_array_equal(out, np.roll(field, -sign, axis=3 - direction))


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_distributed_face(local, monkeypatch, direction, sign):
    field = np.arange(4**4 * 2).reshape(4, 4, 4, 4, 2).astype(complex)
    grid, coord = [3] * 4, [1] * 4
    monkeypatch.setattr(kernel, "getGridCoord", lambda: coord)
    monkeypatch.setattr(kernel, "getGridSize", lambda: grid)
    monkeypatch.setattr(kernel, "getRankFromCoord", tuple)
    def exchange(sendbuf, dest, recvbuf, source):
        expected_dest, expected_src = coord.copy(), coord.copy()
        expected_dest[direction] -= sign
        expected_src[direction] += sign
        assert dest == tuple(expected_dest) and source == tuple(expected_src)
        np.testing.assert_array_equal(sendbuf, np.take(field, 0 if sign > 0 else -1, axis=3-direction))
        recvbuf[:] = 987 + 2j
    monkeypatch.setattr(kernel, "getMPIComm", lambda: SimpleNamespace(Sendrecv=exchange))
    out = np.empty_like(field)
    kernel._shift_one(field, out, direction, sign, {})
    expected = np.roll(field, -sign, axis=3-direction)
    edge = [slice(None)] * 5
    edge[3-direction] = -1 if sign > 0 else 0
    expected[tuple(edge)] = 987 + 2j
    np.testing.assert_array_equal(out, expected)


def test_complete_cg_walk(local):
    field = np.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8).astype(complex)
    d0, d1 = create_cg_qtmd_wilsonline_index_lists(24, 24)
    original = field.copy()
    for index, shifted in zip(d0 + d1, kernel._cg_walk(field, d0 + d1)):
        expected = np.roll(np.roll(field, -index[0], 3-index[3]), -index[1], 1)
        np.testing.assert_array_equal(shifted, expected)
    np.testing.assert_array_equal(field, original)


@pytest.mark.parametrize("operator", ["CG_qTMD", "CG_PDF", "GI_PDF"])
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_full_contraction_against_legacy_einsum(local, monkeypatch, operator, block_size):
    rng = np.random.default_rng(72)
    def rand(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    info = Info()
    forward_lex = rand((*info.size[::-1], 4, 4, 3, 3))
    forward = Prop(info, forward_lex)
    seq_lex = [rand((2, *info.size[::-1], 4, 4, 3, 3)) for _ in range(2)]
    sequential = [info.evenodd(value, True) for value in seq_lex]
    gamma = rand((16, 4, 4))
    qz = [0] if operator == "CG_qTMD" else [-1, 0, 1]
    momenta = [[x, y, z, 0] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in qz]
    origin = [3, 1, 1, 0]
    info.global_size = [8, 6, 4, 2]
    coord = [1, 2, 1, 0]
    monkeypatch.setattr(kernel, "getGridCoord", lambda: coord)
    positions = np.indices(info.size[::-1])[::-1]
    phase_lex = np.asarray([np.exp(sum(2j * np.pi * q[d] *
        (positions[d] + coord[d] * info.size[d] - origin[d]) / info.global_size[d]
        for d in range(3))) for q in momenta])
    phases = info.evenodd(phase_lex, True)
    indices = sum(create_cg_qtmd_wilsonline_index_lists(2, 1), []) if operator == "CG_qTMD" else create_pdf_wilsonline_index_list(2)
    calls = []
    monkeypatch.setattr(kernel, "_reduce_spatial_projected", lambda a: calls.append(a.shape) or a.copy())
    # GI transport remains the existing helper: this stub checks call/reset order,
    # not SU(3) gauge transport (covered by the real-gauge integration run).
    gi_calls = []
    def gi_shift(gauge, prop, index, previous):
        delta = index[1] - previous[1]
        assert delta in (-1, 0, 1)
        gi_calls.append((index[1], previous[1]))
        return Prop(info, np.roll(prop.lexico(), -delta, axis=1))
    monkeypatch.setattr(kernel, "shift_propagator_pdf_gi", gi_shift)
    actual = kernel.contract_nucleon_qtmd(latt_info=info, gauge=None, forward_prop=forward,
        sequential_down=sequential[0], sequential_up=sequential[1], gamma_stack_device=gamma,
        momentum_list=momenta, source_position=origin, wilson_indices=indices,
        operator_kind=operator, block_size=block_size)
    for flavor in range(2):
        expected = []
        for index in indices:
            shifted = np.roll(np.roll(forward_lex, -index[0], 3-index[3]), -index[1], 1)
            scalar = np.einsum("pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx", sequential[flavor], gamma, info.evenodd(shifted), optimize=True)
            expected.append(np.einsum("qwtzyx,pgwtzyx->pqgt", phases, scalar, optimize=True))
        np.testing.assert_allclose(actual[flavor], expected, rtol=3e-12, atol=3e-11)
    assert len(calls) == (len(indices) + block_size - 1) // block_size
    assert calls[-1][0] == (len(indices)-1) % block_size + 1
    np.testing.assert_array_equal(forward.lexico(), forward_lex)
    for before, after in zip(seq_lex, sequential):
        np.testing.assert_array_equal(info.lexico(after, True), before)
    if operator == "GI_PDF":
        assert gi_calls == [(0, 0), (1, 0), (2, 1), (-1, 0), (-2, -1)]


def test_nonroot(local, monkeypatch):
    info = Info()
    field = np.zeros((*info.size[::-1], 4, 4, 3, 3), complex)
    seq = info.evenodd(field[None], True)
    monkeypatch.setattr(kernel, "getMPIComm", lambda: SimpleNamespace(Get_rank=lambda: 1))
    monkeypatch.setattr(kernel, "_reduce_spatial_projected", lambda a: None)
    assert kernel.contract_nucleon_qtmd(latt_info=info, gauge=None, forward_prop=Prop(info, field),
        sequential_down=seq, sequential_up=seq, gamma_stack_device=np.zeros((16, 4, 4), complex),
        momentum_list=[[0, 0, 0, 0]], source_position=[0]*4, wilson_indices=[[0]*4], operator_kind="CG_PDF") == (None, None)
