import inspect
from contextlib import contextmanager

import numpy as np
import pytest

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import (
    FlowedQuarkRingedNorm,
    _iter_source_batches,
    _positive_flow_batch_size,
)
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT
from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_ringed_flow_batch_size_rejects_non_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        _positive_flow_batch_size(value)


def test_ringed_source_batch_iterator_preserves_order_and_tail():
    assert list(_iter_source_batches(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_ringed_batch_uses_one_context_per_flow_time(monkeypatch):
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "0")
    measurement = FlowedQuarkRingedNorm.__new__(FlowedQuarkRingedNorm)
    measurement.flow_steps = 2
    context_entries = []
    contractions = []
    advances = []

    class Gauge:
        class LattInfo:
            global_size = [2, 2, 2, 3]

        latt_info = LattInfo()

        def copy(self):
            return self

        def setAntiPeriodicT(self):
            pass

        @contextmanager
        def use(self):
            token = object()
            context_entries.append(token)
            yield token

    def kinetic(_u, gauge_dirac, xi, eta, _phase, _volume):
        contractions.append((gauge_dirac, xi, eta))
        return np.full(3, xi + eta, dtype=np.complex128)

    def advance(_u, xis, etas, step):
        advances.append((list(xis), list(etas), step))
        return (
            object(),
            [value + 10 for value in xis],
            [value + 20 for value in etas],
        )

    measurement._kinetic_per_time_for_source = kinetic
    measurement._advance_flowed_batch = advance
    timers = {"contract": np.zeros(3), "flow": np.zeros(2)}
    result = measurement._measure_source_batch(
        Gauge(), [1, 2], [4, 5], None, 8, timers
    )

    assert result.shape == (2, 3, 3)
    assert len(context_entries) == 3
    assert len(contractions) == 6
    assert all(
        contractions[2 * step][0] is contractions[2 * step + 1][0]
        for step in range(3)
    )
    assert [entry[2] for entry in advances] == [0, 1]


def test_ringed_batch_restores_thin_gauge_once(monkeypatch):
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "0")
    measurement = FlowedQuarkRingedNorm.__new__(FlowedQuarkRingedNorm)
    captured = {}

    class Dirac:
        def __init__(self):
            self.loads = []
            self.inverted = []

        def loadGauge(self, gauge, thin_update_only=False):
            self.loads.append((gauge, thin_update_only))

        def invert(self, source):
            self.inverted.append(source)
            return source + 100

    def measure(_u, xis, etas, _phase, _volume, _timers):
        captured["xis"] = xis
        captured["etas"] = etas
        return np.asarray(etas)

    measurement._measure_source_batch = measure
    gauge, dirac = object(), Dirac()
    records = [(0, 0, 0, -1, -1, 1), (1, 1, 0, -1, -1, 2)]
    timers = {"restore": 0.0, "invert": 0.0}
    measurement._invert_and_measure_batch(
        gauge, dirac, records, None, 8, timers
    )

    assert dirac.loads == [(gauge, True)]
    assert dirac.inverted == [1, 2]
    assert captured == {"xis": [1, 2], "etas": [101, 102]}


def test_connected_helpers_require_caller_owned_gauge_context():
    assert list(inspect.signature(
        EMTDisconnectedQuark1pt._covdev_sym_prop
    ).parameters) == ["gauge_dirac", "prop", "mu"]
    assert "loadGauge" not in inspect.getsource(
        EMTDisconnectedQuark1pt._covdev_sym_prop
    )
    assert "gauge_dirac" in inspect.signature(
        QuarkEMT.get_C3_primitive_bilinears
    ).parameters
    assert "gauge_dirac" in inspect.signature(
        ProtonQuarkEMT.get_C3_primitive_bilinears_proton
    ).parameters


@pytest.mark.parametrize("measurement", [QuarkEMT, ProtonQuarkEMT])
def test_connected_production_uses_thin_restore_and_one_caller_context(measurement):
    source = inspect.getsource(measurement)
    assert "dirac.loadGauge(U, thin_update_only=True)" in source
    assert "with U_f.use() as gauge_dirac" in source


def test_connected_production_has_no_branch_batching_interface():
    assert "flow_branch_batch" not in inspect.getsource(QuarkEMT)
    assert "flow_branch_batch" not in inspect.getsource(ProtonQuarkEMT)
