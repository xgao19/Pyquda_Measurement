from pyquda_measurement_utils.pion_qTMD_vibe_develop import pion_TMD
from pyquda_measurement_utils.proton_qTMD_pyquda import proton_TMD


class FakeFermion:
    def __init__(self, path=None):
        self.path = list(path or [])

    def copy(self):
        return FakeFermion(self.path)


class FakePureGauge:
    def covDev(self, fermion, direction):
        return FakeFermion(fermion.path + [direction])


class FakeGauge:
    pure_gauge = FakePureGauge()


class FakePropagator:
    def __init__(self):
        self.values = {(spin, color): FakeFermion() for spin in range(4) for color in range(3)}

    def copy(self):
        copied = FakePropagator()
        copied.values = {key: val.copy() for key, val in self.values.items()}
        return copied

    def getFermion(self, spin, color):
        return self.values[(spin, color)]

    def setFermion(self, fermion, spin, color):
        self.values[(spin, color)] = fermion


def _all_paths(prop):
    return {tuple(value.path) for value in prop.values.values()}


def test_connected_pion_and_proton_pdf_gi_one_step_helpers_match():
    gauge = FakeGauge()
    pion = object.__new__(pion_TMD)
    proton = object.__new__(proton_TMD)

    pion_prop = pion.create_fw_prop_PDF_GI(gauge, FakePropagator(), [0, 1, 0, 0], [0, 0, 0, 0])
    proton_prop = proton.create_fw_prop_PDF_GI(gauge, FakePropagator(), [0, 1, 0, 0], [0, 0, 0, 0])
    assert _all_paths(pion_prop) == {(2,)}
    assert _all_paths(proton_prop) == {(2,)}

    pion_prop = pion.create_fw_prop_PDF_GI(gauge, FakePropagator(), [0, -1, 0, 0], [0, 0, 0, 0])
    proton_prop = proton.create_fw_prop_PDF_GI(gauge, FakePropagator(), [0, -1, 0, 0], [0, 0, 0, 0])
    assert _all_paths(pion_prop) == {(6,)}
    assert _all_paths(proton_prop) == {(6,)}


def test_connected_pdf_gi_helpers_reject_non_incremental_jumps():
    gauge = FakeGauge()
    pion = object.__new__(pion_TMD)
    proton = object.__new__(proton_TMD)

    for measurement in (pion, proton):
        try:
            measurement.create_fw_prop_PDF_GI(gauge, FakePropagator(), [0, 2, 0, 0], [0, 0, 0, 0])
        except ValueError:
            continue
        raise AssertionError("PDF GI helper should reject jumps larger than one lattice spacing")


def test_connected_gi_qtmd_production_requires_link_cache():
    gauge = FakeGauge()
    pion = object.__new__(pion_TMD)
    proton = object.__new__(proton_TMD)
    w_index = [2, 4, 3, 0]

    for measurement in (pion, proton):
        with pytest.raises(TypeError):
            if measurement is proton:
                measurement.create_fw_prop_TMD_GI(
                    FakePropagator(), w_index
                )
            else:
                measurement.create_fw_prop_TMD_GI(
                    gauge, FakePropagator(), w_index
                )
import pytest
