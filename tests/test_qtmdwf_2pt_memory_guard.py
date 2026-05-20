import inspect

from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement


def test_qtmdwf_2pt_contracts_one_sink_gamma_at_a_time():
    source = inspect.getsource(pion_TMDWF_measurement.contract_2pt_pion)

    assert "for gamma_idx" in source
    assert 'glj' not in source
    assert "src_gamma_ls = [G5_backend] * n_gamma" in source
    assert '"wtzyxjiab, wtzyxilba, lj -> wtzyx"' in source
