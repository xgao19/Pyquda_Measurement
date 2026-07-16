import inspect

import numpy as np

from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT
from pyquda_measurement_utils.pion_qTMD_vibe_develop import pion_TMD
import pyquda_measurement_utils.pion_qTMD_vibe_develop as pion_qtmd_module
import pyquda_measurement_utils.pion_EMT_vibe_develop as pion_emt_module
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt,
    contract_pion_2pt_multi_src_gamma,
)


def test_qtmdwf_2pt_contracts_one_sink_gamma_at_a_time():
    wrapper_source = inspect.getsource(pion_TMDWF_measurement.contract_2pt_pion)
    source = inspect.getsource(contract_pion_2pt_multi_src_gamma)

    assert "for gamma_idx" in source
    assert "latt_info.size[3]" in source
    assert "latt_info.global_size[3]" not in source
    assert "contract_pion_2pt(" in wrapper_source
    assert "contract_2pt_pion_multi_src_gamma" not in wrapper_source
    assert "sink_inserted" not in wrapper_source


def test_pion_emt_c2_has_no_gamma_times_propagator_intermediate():
    source = inspect.getsource(QuarkEMT.contract_meson_2pt)

    assert '"wtzyxjicf,gim->gwtzyxjmcf"' not in source.replace(" ", "")
    assert "for gamma_idx, sink_gamma" in source
    assert "latt_info.size[3]" in source


def test_pion_qtmd_has_no_gamma_times_propagator_intermediate():
    source = inspect.getsource(pion_TMD._contract_qTMD_one_shift)

    assert '"wtzyxjicf,gim->gwtzyxjmcf"' not in source.replace(" ", "")
    assert "for gamma_idx, sink_gamma" in source
    assert "shifted_prop.latt_info.size[3]" in source


def test_pion_qtmd_channel_loop_matches_stacked_reference(monkeypatch):
    rng = np.random.default_rng(17)
    lattice_shape = (2, 3, 1, 1, 1)
    color = 2
    seq_bw_line = rng.normal(size=lattice_shape + (4, 4, color, color)) + 1j * rng.normal(
        size=lattice_shape + (4, 4, color, color)
    )
    shifted_data = rng.normal(size=lattice_shape + (4, 4, color, color)) + 1j * rng.normal(
        size=lattice_shape + (4, 4, color, color)
    )
    sink_gamma_ls = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    source_gamma_ls = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    phases = rng.normal(size=(2,) + lattice_shape) + 1j * rng.normal(size=(2,) + lattice_shape)

    class FakeLatticeInfo:
        size = [1, 1, 1, lattice_shape[1]]

    class FakePropagator:
        data = shifted_data
        latt_info = FakeLatticeInfo()

    monkeypatch.setattr(pion_qtmd_module.core, "gatherLattice", lambda values, axes: values)
    result = pion_TMD._contract_qTMD_one_shift(
        object(),
        seq_bw_line,
        FakePropagator(),
        sink_gamma_ls,
        source_gamma_ls,
        phases,
    )

    sink_inserted = np.einsum(
        "wtzyxjicf,gim->gwtzyxjmcf", seq_bw_line, sink_gamma_ls, optimize=True
    )
    corr_site = np.einsum(
        "gwtzyxjiab,wtzyxilba,glj->gwtzyx",
        sink_inserted,
        shifted_data,
        source_gamma_ls,
        optimize=True,
    )
    reference = np.einsum("qwtzyx,gwtzyx->gqt", phases, corr_site, optimize=True)
    np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-13)


def test_pion_emt_c2_channel_loop_matches_stacked_reference(monkeypatch):
    rng = np.random.default_rng(31)
    lattice_shape = (2, 2, 1, 1, 1)
    color = 2
    data_shape = lattice_shape + (4, 4, color, color)
    forward = rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape)
    backward = rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape)
    sink_gammas = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    gamma5 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    source_gamma = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    phases = rng.normal(size=(2,) + lattice_shape) + 1j * rng.normal(
        size=(2,) + lattice_shape
    )

    class FakeLatticeInfo:
        size = [1, 1, 1, lattice_shape[1]]
        mpi_rank = 0

    class FakePropagator:
        def __init__(self, data):
            self.data = data

    class FakeMomentumPhase:
        def __init__(self, latt_info):
            pass

        def getPhases(self, momenta, source_position):
            assert len(momenta) == phases.shape[0]
            return phases

    class FakeComm:
        @staticmethod
        def bcast(values, root=0):
            return values

    class FakeMeasurement:
        CG_GaussSmear = False
        pilist = [[0, 0, 0, 0], [1, 0, 0, 0]]

        @staticmethod
        def _gamma_stack_for(reference):
            return sink_gammas

        @staticmethod
        def _gamma5_for(reference):
            return gamma5

    monkeypatch.setattr(pion_emt_module.phase, "MomentumPhase", FakeMomentumPhase)
    monkeypatch.setattr(pion_emt_module.core, "gatherLattice", lambda values, axes: values)
    monkeypatch.setattr(pion_emt_module, "getMPIComm", lambda: FakeComm())

    result = QuarkEMT.contract_meson_2pt(
        FakeMeasurement(),
        FakeLatticeInfo(),
        FakePropagator(forward),
        FakePropagator(backward),
        source_gamma,
        [0, 0, 0, 0],
    )

    backward_line = np.einsum(
        "ij,wtzyxilab,kl->wtzyxkjba", gamma5, backward.conj(), gamma5, optimize=True
    )
    sink_inserted = np.einsum(
        "wtzyxjicf,gim->gwtzyxjmcf", backward_line, sink_gammas, optimize=True
    )
    corr_site = np.einsum(
        "gwtzyxjiab,wtzyxilba,lj->gwtzyx",
        sink_inserted,
        forward,
        source_gamma,
        optimize=True,
    )
    reference = np.einsum("qwtzyx,gwtzyx->gqt", phases, corr_site, optimize=True)
    np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-13)
