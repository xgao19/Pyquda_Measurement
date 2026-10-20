"""CPU mocks cover scheduling, ownership and persistent provenance."""

import numpy as np
import h5py
import pytest
from types import SimpleNamespace
import pyquda_measurement_utils.topological_charge_1pt as topology
from pyquda_measurement_utils.io_corr import get_topological_charge_loop_file_tag


def parameters(**overrides):
    return dict(flow_type="wilson", flow_epsilon=.1, flow_steps=2,
                flow_substeps_per_interval=10, qext=[[0, 0, 0, 0]],
                ensemble="small", stream="b", config_num=600, setup="test",
                gauge_path="identity", gauge_preprocessing="none", code_version="test", **overrides)


@pytest.mark.parametrize("key,value", [
    ("flow_type", "unknown"), ("flow_epsilon", 0), ("flow_epsilon", float("nan")),
    ("flow_steps", -1), ("flow_steps", 1.5), ("flow_steps", True),
    ("flow_substeps_per_interval", 0), ("qext", [[1, 0, 0, 0]]),
    ("flow_first_interval_substeps", 0), ("flow_first_interval_substeps", True),
    ("flow_first_interval_substeps", 1.5),
    ("qext", [[0., 0., 0., 0.]]), ("qext", [[0, 0, 0, 1]]),
    ("qext", [[0, 0, 0, 0]] * 2), ("qext", []),
])
def test_invalid_parameters(key, value):
    params = parameters()
    params[key] = value
    with pytest.raises(ValueError):
        topology.TopologicalCharge1pt(params)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("flow_type", ["wilson", "symanzik"])
@pytest.mark.parametrize("first_substeps,later_substeps", [(None, 10), (10, 3)])
def test_schedule_input_ownership_and_hdf5(monkeypatch, tmp_path, rank, flow_type, first_substeps, later_substeps):
    calls = []
    info = SimpleNamespace(global_size=[2, 2, 2, 4], mpi_rank=rank)

    class Gauge:
        latt_info = info
        def __init__(self):
            self.time = 0
        def copy(self):
            return Gauge()
        def toDevice(self):
            pass
        def wilsonFlow(self, n, epsilon):
            calls.append(("wilson", n, epsilon))
            self.time += n * epsilon
        def symanzikFlow(self, n, epsilon):
            calls.append(("symanzik", n, epsilon))
            self.time += n * epsilon

    monkeypatch.setattr(topology, "all_clover_traceless", lambda gauge: gauge.time)
    monkeypatch.setattr(topology, "topological_density", lambda time: np.full((2, 4, 2, 2, 1), 1 + time))
    monkeypatch.setattr(topology.phase, "MomentumPhase", lambda _: SimpleNamespace(getPhases=lambda *args: np.ones((1, 2, 4, 2, 2, 1))))
    monkeypatch.setattr(topology, "_reduce_spatial_projected", lambda a: a if rank == 0 else None)
    monkeypatch.setattr(topology, "getGridSize", lambda: [1, 1, 1, 1])
    params = parameters()
    params["flow_type"] = flow_type
    params["flow_substeps_per_interval"] = later_substeps
    if first_substeps is not None:
        params["flow_first_interval_substeps"] = first_substeps
    gauge = Gauge()
    tag = get_topological_charge_loop_file_tag(tmp_path, "small", "b", 600, "test")
    result = topology.TopologicalCharge1pt(params).flowed_1pt(gauge, tag)
    assert gauge.time == 0
    first = later_substeps if first_substeps is None else first_substeps
    assert calls == [(flow_type, first, .1 / first), (flow_type, later_substeps, .1 / later_substeps)]
    if rank:
        assert result is None
        assert not list(tmp_path.rglob("*.h5"))
    else:
        np.testing.assert_allclose(result["loop"][0], np.repeat([[1.], [1.1], [1.2]], 4, axis=1))
        np.testing.assert_allclose(result["Q_total"], [32, 35.2, 38.4])
        with h5py.File(tag + ".h5") as handle:
            assert handle["loop"].dtype == np.complex128
            assert handle["loop"].shape == (1, 3, 4)
            assert handle.attrs["stream"] == "b"
            assert handle.attrs["volume_norm"] == 8
            assert handle.attrs["schema"] == "topological_charge_loop_v1"
            assert handle.attrs["flow_first_interval_substeps"] == first
            assert handle.attrs["flow_substeps_per_interval"] == later_substeps
            assert handle.attrs["first_integration_epsilon"] == .1 / first
            assert handle.attrs["integration_epsilon"] == .1 / later_substeps
            np.testing.assert_allclose(handle["flow_times_lattice"][:], [0, .1, .2])
            np.testing.assert_allclose(handle["diagnostics/Q_total"][:], result["Q_total"])


def test_emt_preserves_first_interval_schedule():
    from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import EMTDisconnectedGluon1pt
    calls = []
    gauge = SimpleNamespace(wilsonFlow=lambda n, epsilon: calls.append((n, epsilon)))
    measurement = EMTDisconnectedGluon1pt(parameters())
    for step in range(4):
        measurement._advance_flowed_gauge(gauge, step, .1, 3, 2)
    assert calls == [(10, .01), (2, .05), (2, .05)]
