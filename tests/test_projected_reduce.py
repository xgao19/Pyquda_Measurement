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
import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft_factor
import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt


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


@pytest.mark.parametrize('rank', [0, 1])
def test_soft_factor_reductions_preserve_axes_and_root_results(monkeypatch, rank):
    rng = np.random.default_rng(82)
    shape = (2, 1, 2, 2, 4, 4, 3, 3)
    info = SimpleNamespace(mpi_rank=rank, size=[2, 2, 1, 2], global_size=[2, 2, 1, 2])

    class Prop:
        def __init__(self, data):
            self.data = data

        def copy(self):
            return Prop(self.data.copy())

        def lexico(self, multi):
            assert multi is False
            return self.data

        def shift(self, steps, direction):
            return Prop(np.roll(self.data, -steps, axis=3 - direction))

    props = [Prop(rng.normal(size=shape) + 1j * rng.normal(size=shape)) for _ in range(4)]
    measurement = soft_factor.pion_soft_factor({
        'quark_mom': [[0, 0, 1]], 'bT_dir': [0, 1], 'bT_length': 1,
        'bz_length': 1, 'tsep_list': [3],
        'pion_channel_pairs': {'5': (soft_factor.G5, soft_factor.G5)},
        'gamma_channel_pairs': {'5': ('5', '5'), 'XY': ('X', 'Y')},
    })
    monkeypatch.setattr(measurement, 'apply_phase', lambda prop, *args: prop.copy())
    monkeypatch.setattr(soft_factor, 'mpi_print', lambda *args: None)
    monkeypatch.setattr(soft_factor, 'getGridSize', lambda: [1, 1, 1, 1])
    projected = []

    def reduce(local):
        projected.append(local.copy() * 3)
        return projected[-1] if rank == 0 else None

    monkeypatch.setattr(soft_factor, '_reduce_spatial_projected', reduce)
    wall = measurement.contract_wall_2pt(info, *props[:2], [0, 0, 1], '5')
    tmdwf = measurement.contract_tmdwf_check(info, *props[:2], [0, 0, 1], '5')
    corr, pion_labels, gamma_labels = measurement.contract_soft_factor(info, *props, [0, 0, 1])
    assert len(projected) == 3
    assert pion_labels == ['5'] and gamma_labels == ['5', 'XY']
    if rank == 0:
        np.testing.assert_array_equal(wall, projected[0])
        np.testing.assert_array_equal(tmdwf, projected[1].reshape(8, 2))
        np.testing.assert_array_equal(corr, projected[2])
    else:
        assert wall is None and tmdwf is None and corr is None


@pytest.mark.parametrize('rank', [0, 1])
def test_gluon_reduce_keeps_normalization_flow_and_nonroot_zeros(monkeypatch, rank, tmp_path):
    info = SimpleNamespace(mpi_rank=rank, size=[2, 1, 1, 2],
                           global_size=[6, 1, 1, 2], t_boundary=-1)
    gauge = SimpleNamespace(latt_info=info, data=np.zeros(1, complex), location='numpy')
    gauge.copy = lambda: gauge
    shape = (2, 2, 1, 1, 1)
    rng = np.random.default_rng(83)
    fields = [[rng.normal(size=shape + (3, 3)) for _ in range(4)] for _ in range(4)]
    phases = rng.normal(size=(3,) + shape) + 1j * rng.normal(size=(3,) + shape)
    monkeypatch.setattr(emt.phase, 'MomentumPhase',
                        lambda info: SimpleNamespace(getPhases=lambda *args: phases))
    monkeypatch.setattr(emt, 'arrayZeros', lambda shape, dtype, location: np.zeros(shape, dtype))
    monkeypatch.setattr(emt, 'mpi_print', lambda *args: None)
    measurement = emt.EMTDisconnectedGluon1pt({
        'qext': [[0, 0, 0, 0], [1, 0, 0, 0], [-1, 0, 0, 0]],
        'flow_type': 'symanzik', 'flow_epsilon': 0.1, 'flow_steps': 1,
        'config_num': 600,
    })
    monkeypatch.setattr(measurement, '_all_F_clover_traceless', lambda gauge: fields)
    advances = []
    monkeypatch.setattr(measurement, '_advance_flowed_gauge', lambda *args: advances.append(args[1:]))
    projected = []

    def reduce(local):
        projected.append(local.copy() * 3)
        return projected[-1] if rank == 0 else None

    monkeypatch.setattr(emt, '_reduce_spatial_projected', reduce)
    writes = []
    monkeypatch.setattr(emt, 'save_emt_gluon_1pt_hdf5',
                        lambda tag, data, attrs: writes.append((data.copy(), attrs)))
    result = measurement.flowed_1pt(gauge, str(tmp_path / 'gluon'))
    assert len(projected) == 20
    assert advances == [(0, 0.1, 1, 1), (1, 0.1, 1, 1)]
    expected = np.zeros((4, 4, 3, 2, 2), complex)
    if rank == 0:
        index = 0
        for step in range(2):
            for mu in range(4):
                for nu in range(mu, 4):
                    expected[mu, nu, :, step] = 2 * projected[index] / 6
                    index += 1
        assert len(writes) == 1
        np.testing.assert_array_equal(writes[0][0], expected)
        assert writes[0][1]['volume_norm'] == 6
        np.testing.assert_array_equal(writes[0][1]['flow_times'], [0., 0.1])
    else:
        assert writes == []
    np.testing.assert_array_equal(result, expected)
