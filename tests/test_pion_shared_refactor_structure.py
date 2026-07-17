import inspect

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
)
from pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop import (
    FlowedFermionBilinearKernel,
)
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT
from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT
from pyquda_measurement_utils.pion_soft_factor_vibe_develop import pion_soft_factor
import pyquda_measurement_utils.pion_current_background_response_vibe_develop as response
from application.qTMDWF_CG import qTMDWF_runner


def test_connected_emt_uses_lightweight_bilinear_base():
    assert issubclass(QuarkEMT, FlowedFermionBilinearKernel)
    assert issubclass(ProtonQuarkEMT, FlowedFermionBilinearKernel)
    assert not issubclass(QuarkEMT, EMTDisconnectedQuark1pt)
    assert not issubclass(ProtonQuarkEMT, EMTDisconnectedQuark1pt)


def test_soft_factor_uses_global_propagator_shift():
    source = inspect.getsource(pion_soft_factor.contract_soft_factor)
    assert "xp.roll" not in source
    assert "phased_sink_backward.shift(bT, bT_dir)" in source
    assert "prop_bw_src.shift(bT, bT_dir)" in source


def test_qtmdwf_runner_uses_exact_log_and_one_root_writer():
    source = inspect.getsource(qTMDWF_runner.run_qtmdwf_sources)
    assert "read_sample_log_entries" in source
    assert "if entry in completed" in source
    assert source.count("save_qTMDWF_hdf5_noRoll(") == 1
    assert "if latt_info.mpi_rank == 0" in source
    assert "append_sample_log_entry" in source


def test_response_production_module_contains_no_analysis_or_writers():
    assert not hasattr(response, "tau_window_list")
    assert not hasattr(response, "response_ratio")
    assert not hasattr(response, "save_pion_EMFF_background_response_hdf5")
    assert not hasattr(response, "current_current_response_toy")
