import inspect

import pyquda_measurement_utils.pion_EMT_vibe_develop as pion_emt
import pyquda_measurement_utils.pion_utils_vibe_develop as pion_utils
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT


class _TaggedField:
    def __init__(self, identity):
        self.identity = identity

    def copy(self):
        return _TaggedField(self.identity)


class _Gauge:
    class _LatticeInfo:
        mpi_rank = 0

    latt_info = _LatticeInfo()


def _measurement(pos_boost, neg_boost):
    measurement = QuarkEMT.__new__(QuarkEMT)
    measurement.CG_GaussSmear = True
    measurement.pos_boost = list(pos_boost)
    measurement.neg_boost = list(neg_boost)
    measurement.width = 1.0
    return measurement


def _install_source_fakes(monkeypatch):
    source_calls = []
    inversion_calls = []

    def point_source(_latt_info, source_type, position):
        assert source_type == "point"
        field = _TaggedField(("point", len(source_calls), tuple(position)))
        source_calls.append(field.identity)
        return field

    def smear(field, *, w, boost):
        return _TaggedField(("smeared", field.identity, float(w), tuple(boost)))

    def invert(_dirac, field, _source_type, _solution_type):
        inversion_calls.append(field.identity)
        return _TaggedField(("propagator", field.identity))

    monkeypatch.setattr(pion_utils.source, "propagator", point_source)
    monkeypatch.setattr(pion_utils, "boosted_smearing", smear)
    monkeypatch.setattr(pion_utils.core, "invertPropagator", invert)
    return source_calls, inversion_calls


def test_equal_boost_reuses_one_source_inversion(monkeypatch):
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "0")
    source_calls, inversion_calls = _install_source_fakes(monkeypatch)
    measurement = _measurement([0, 0, 1], [0, 0, 1])

    spectator, active = measurement._make_meson_source_props(
        object(), _Gauge(), [0, 0, 0, 0], restore_original_gauge=False
    )

    assert len(source_calls) == 1
    assert len(inversion_calls) == 1
    assert spectator is not active
    assert spectator.identity == active.identity


def test_unequal_boost_builds_independent_spectator_and_active_sources(monkeypatch):
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "0")
    source_calls, inversion_calls = _install_source_fakes(monkeypatch)
    measurement = _measurement([0, 0, 1], [0, 0, -1])

    spectator, active = measurement._make_meson_source_props(
        object(), _Gauge(), [0, 0, 0, 0], restore_original_gauge=False
    )

    assert len(source_calls) == 2
    assert len(inversion_calls) == 2
    assert inversion_calls[0][-1] == (0, 0, 1)
    assert inversion_calls[1][-1] == (0, 0, -1)
    assert spectator.identity != active.identity


def test_connected_3pt_routes_positive_spectator_and_negative_active_lines():
    source = inspect.getsource(QuarkEMT.connected_3pt)

    assert "prop_fw_SS = boosted_smearing(prop_fw_SP.copy()" in source
    assert "dirac,\n                prop_fw_SS," in source
    assert "self.neg_boost if self.CG_GaussSmear else None" in source
    assert "active_prop_flow = prop_bw_SP.copy()" in source
    assert "gauge_dirac, active_prop_flow, seq_bw_prop_flow" in source
    assert "prop_fw_flow = prop_fw_SP.copy()" not in source


def test_pion_boost_provenance_uses_line_not_endpoint_names():
    source = inspect.getsource(QuarkEMT.connected_3pt)

    assert '"pos_boost"' in source
    assert '"neg_boost"' in source
    assert '"operator_insertion_line": "neg_boost"' in source
    assert '"boost_line_convention": "pos_spectator_neg_active"' in source
    assert '"source_boost"' not in source
    assert '"sink_boost"' not in source
