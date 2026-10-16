"""CPU checks for root-only projected sums and their contraction callers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyquda

from pyquda_measurement_utils import tools
import pyquda_measurement_utils.pion_utils_vibe_develop as pion
import pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop as disconnected


@pytest.mark.parametrize('grid', [(1, 1, 1, 1), (2, 1, 1, 3), (1, 5, 4, 5)])
@pytest.mark.parametrize('prefix', [(3,), (16, 3), (2, 3, 16)])
def test_reduce_preserves_global_time_and_returns_only_on_root(monkeypatch, grid, prefix):
    rng = np.random.default_rng(71)
    local = rng.normal(size=grid + prefix + (2,)) + 1j * rng.normal(size=grid + prefix + (2,))
    expected = np.concatenate(list(local.sum(axis=(0, 1, 2))), axis=-1)
    monkeypatch.setattr(pyquda, 'getGridSize', lambda: grid)
    for rank, coord in enumerate(np.ndindex(grid)):
        backing = np.zeros(prefix + (4,), complex)
        projected = backing[..., ::2]
        projected[...] = local[coord]
        submitted = np.zeros_like(expected)
        submitted[..., 2 * coord[3]:2 * coord[3] + 2] = projected
        calls = []

        def reduce(send, recv, op, root):
            assert root == 0 and op is pyquda.MPI.SUM
            if rank == 0:
                assert send is pyquda.MPI.IN_PLACE
                np.testing.assert_array_equal(recv, submitted)
                assert recv.flags.c_contiguous
                recv[...] = expected
            else:
                assert recv is None
                np.testing.assert_array_equal(send, submitted)
                assert send.flags.c_contiguous
            calls.append(1)

        monkeypatch.setattr(pyquda, 'getGridCoord', lambda: coord)
        monkeypatch.setattr(pyquda, 'getMPIComm', lambda: SimpleNamespace(Get_rank=lambda: rank, Reduce=reduce))
        actual = tools._reduce_spatial_projected(projected)
        if rank == 0:
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual is None
        np.testing.assert_array_equal(projected, local[coord])
        assert calls == [1]


@pytest.mark.parametrize('rank', [0, 1])
def test_contraction_callers_keep_root_only_results(monkeypatch, rank):
    rng = np.random.default_rng(73)
    shape = (2, 2, 1, 1, 1, 4, 4, 3, 3)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    info = SimpleNamespace(mpi_rank=rank, size=[2, 1, 1, 2])

    class Prop:
        def __init__(self, values):
            self.data = values

        def copy(self):
            return Prop(self.data.copy())

        def shift(self, steps, direction):
            assert steps == 0
            return self.copy()

    phases = np.ones((3,) + shape[:5], complex)
    reduced = []

    def reduce(local):
        reduced.append(local.copy())
        return local * 2 if rank == 0 else None

    monkeypatch.setattr(tools, '_reduce_spatial_projected', reduce)
    monkeypatch.setattr(pion, '_reduce_spatial_projected', reduce)
    monkeypatch.setattr(disconnected, '_reduce_spatial_projected', reduce)
    monkeypatch.setattr(pyquda, 'getMPIComm', lambda: SimpleNamespace(Get_rank=lambda: rank))
    corr = pion.contract_pion_gamma_scan_from_backward_line(info, Prop(data), data.conj(), phases, ['5', 'X'])
    for i, label in enumerate(['5', 'X']):
        if rank == 0:
            np.testing.assert_array_equal(corr[label], reduced[i] * 2)
        else:
            assert corr[label] is None

    measurement = object.__new__(disconnected.DisconnectedQuarkqTMD1pt)
    fermion = Prop(data[..., 0, :, 0])
    result = measurement._contract_one_operator_list(
        info, None, fermion, fermion, phases, [[0, 0, 0, 0]], 'CG_PDF')
    if rank == 0:
        np.testing.assert_array_equal(result[0], reduced[-1] * 2)
    else:
        assert result is None

    path = Path(__file__).resolve().parents[1] / 'application/nucleon_TMD/shared_runner.py'
    spec = importlib.util.spec_from_file_location('reduce_test_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    down, up = runner._contract_operator_list(
        latt_info=info, gauge=None, prop_f=Prop(data), seq_down=data[None],
        seq_up=data.conj()[None], gamma_ls=pion.gamma_stack(data), phases=phases,
        wilson_indices=[[0, 0, 0, 0]], operator_kind='CG_PDF')
    if rank == 0:
        np.testing.assert_array_equal(down[0], reduced[-2] * 2)
        np.testing.assert_array_equal(up[0], reduced[-1] * 2)
    else:
        assert down is None and up is None


@pytest.mark.parametrize('rank', [0, 1])
def test_disconnected_shards_do_not_broadcast_or_allocate_on_nonroot(monkeypatch, rank, tmp_path):
    info = SimpleNamespace(mpi_rank=rank, global_size=[2, 2, 2, 4])
    gauge = SimpleNamespace(latt_info=info)
    measurement = object.__new__(disconnected.DisconnectedQuarkqTMD1pt)
    measurement.noise_scheme = 'zn'
    measurement.hp_num_vectors = 1
    measurement.hp_ordering = 'unused_for_zn'
    monkeypatch.setattr(disconnected, 'getMPIComm', lambda: SimpleNamespace(Barrier=lambda: None))
    monkeypatch.setattr(disconnected, 'iter_noise_base_probe_interval',
        lambda *a, **kw: iter([(0, 0, 0, object(), None, None)]))
    expected = np.ones((1, 16, 1, 4), complex)
    monkeypatch.setattr(measurement, '_contract_one_operator_list',
        lambda *a, **kw: expected if rank == 0 else None)
    writes = []
    monkeypatch.setattr(disconnected, 'write_raw_part_hdf5',
        lambda path, datasets, *a, **kw: writes.append(datasets['loop_pervec'].copy()))
    monkeypatch.setattr(disconnected, 'append_completed_base', lambda *a, **kw: None)
    allocations = []
    zeros = np.zeros

    def tracked_zeros(shape, *a, **kw):
        allocations.append(shape)
        return zeros(shape, *a, **kw)

    monkeypatch.setattr(disconnected.np, 'zeros', tracked_zeros)
    measurement._measure_base_shards(
        gauge, SimpleNamespace(invert=lambda value: value), [1, 4, 0], str(tmp_path / 'loop'),
        'CG_PDF', None, [[0, 0, 0, 0]], None,
        {'qext': [[0, 0, 0, 0]], 'config_num': 600, 'noise_stream': 0},
        tmp_path, 0, 1, 1, tmp_path / 'samples', set())
    if rank == 0:
        assert len(writes) == 1
        np.testing.assert_array_equal(writes[0], expected[None])
        assert (1, 1, 16, 1, 4) in allocations
    else:
        assert writes == []
        assert (1, 1, 16, 1, 4) not in allocations
