"""CPU tests for spatial sums with a globally ordered time axis."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt


@pytest.mark.parametrize("grid", [(1, 1, 1, 1), (2, 2, 2, 1), (1, 1, 1, 4), (2, 2, 2, 3), (2, 2, 4, 4)])
@pytest.mark.parametrize("prefix", [(3,), (16, 3)])
def test_projected_allreduce_preserves_time_and_sums_space(monkeypatch, grid, prefix):
    rng = np.random.default_rng(27)
    local_nt = 2
    shape = grid + prefix + (local_nt,)
    local = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    spatial_sum = local.sum(axis=(0, 1, 2))
    expected = np.concatenate(list(spatial_sum), axis=-1)
    monkeypatch.setattr(emt, "getGridSize", lambda: grid)
    for coord in np.ndindex(grid):
        # Also exercise a non-contiguous local input.
        backing = np.zeros(prefix + (2 * local_nt,), dtype=np.complex128)
        projected = backing[..., ::2]
        projected[...] = local[coord]
        submitted = np.zeros(prefix + (grid[3] * local_nt,), dtype=np.complex128)
        start = coord[3] * local_nt
        submitted[..., start:start + local_nt] = projected
        calls = []

        def allreduce(sendbuf, recvbuf, op):
            assert sendbuf is emt.MPI.IN_PLACE
            assert op is emt.MPI.SUM
            assert recvbuf.flags.c_contiguous
            assert recvbuf.dtype == np.complex128
            np.testing.assert_array_equal(recvbuf, submitted)
            recvbuf[...] = expected
            calls.append(1)

        monkeypatch.setattr(emt, "getGridCoord", lambda: coord)
        monkeypatch.setattr(emt, "getMPIComm", lambda: SimpleNamespace(Allreduce=allreduce))
        actual = emt._sum_spatial_projected(projected)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(projected, local[coord])
        assert calls == [1]


@pytest.mark.parametrize("gamma_axis", [False, True])
def test_emt_projection_uses_spatial_allreduce(monkeypatch, gamma_axis):
    rng = np.random.default_rng(29)
    sites = (2, 2, 1, 1, 2)
    phases = np.exp(1j * rng.normal(size=(3,) + sites))
    shape = ((16,) if gamma_axis else ()) + sites
    fields = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    equation = "qwtzyx,gwtzyx->gqt" if gamma_axis else "qwtzyx,wtzyx->qt"
    expected_local = np.einsum(equation, phases, fields)
    calls = []

    def reduce(projected):
        np.testing.assert_allclose(projected, expected_local, rtol=1e-13, atol=1e-13)
        calls.append(1)
        return projected * 2

    monkeypatch.setattr(emt, "_sum_spatial_projected", reduce)
    method = (emt.EMTDisconnectedQuark1pt._project_gamma_fields if gamma_axis
              else emt.EMTDisconnectedQuark1pt._impose_P_Breit_slice)
    np.testing.assert_allclose(method(fields, phases), 2 * expected_local, rtol=1e-13, atol=1e-13)
    assert calls == [1]
