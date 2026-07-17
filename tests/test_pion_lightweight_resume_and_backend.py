from pathlib import Path

import numpy as np

from pyquda_measurement_utils.pion_utils_vibe_develop import (
    array_to_numpy,
    zeros_on_backend,
)
from pyquda_measurement_utils.tools import (
    append_sample_log_entry,
    read_sample_log_entries,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_sample_log_uses_exact_lines_and_durable_dedup(tmp_path):
    log = tmp_path / "sample.log"
    assert read_sample_log_entries(log) == set()
    assert append_sample_log_entry(log, "base1")
    assert append_sample_log_entry(log, "base10")
    assert not append_sample_log_entry(log, "base1")
    assert read_sample_log_entries(log) == {"base1", "base10"}
    assert log.read_text(encoding="utf-8").splitlines() == ["base1", "base10"]


def test_lightweight_sample_log_does_not_probe_hdf5(tmp_path):
    log = tmp_path / "sample.log"
    append_sample_log_entry(log, "finished-source")
    assert not list(tmp_path.glob("*.h5"))
    assert read_sample_log_entries(log) == {"finished-source"}


def test_qtmd_and_emff_read_resume_before_source_inversion():
    qtmd = (
        REPO_ROOT / "application/pion_TMD/perlmutter/Pyquda_pion_TMD.py"
    ).read_text(encoding="utf-8")
    emff = (
        REPO_ROOT / "application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py"
    ).read_text(encoding="utf-8")

    assert qtmd.index("read_sample_log_entries(sample_log_file)") < qtmd.index(
        "build_pion_source_propagators("
    )
    assert qtmd.index("append_sample_log_entry(sample_log_file, sample_log_tag)") > qtmd.index(
        "save_qTMD_pion_hdf5_noRoll("
    )

    assert emff.index("read_sample_log_entries(sample_log_file)") < emff.index(
        "core.invertPropagator(dirac, srcD_pos"
    )
    assert "if not pending_tseps:" in emff
    assert "if sample_log_tag in completed_by_tsep[t_insert]:" in emff
    assert emff.index("append_sample_log_entry(sample_log_file, sample_log_tag)") > emff.index(
        "save_pion_EMFF_hdf5_noRoll("
    )


def test_shared_host_conversion_and_queue_aware_zeros():
    values = np.arange(4)
    assert array_to_numpy(values) is values

    class GetArray:
        def get(self):
            return values

    np.testing.assert_array_equal(array_to_numpy(GetArray()), values)

    queue = object()
    calls = []

    class FakeDPNP:
        def __init__(self):
            self.__name__ = "dpnp"

        def zeros(self, shape, *, dtype, sycl_queue):
            calls.append((shape, dtype, sycl_queue))
            return np.zeros(shape, dtype=dtype)

    class Reference:
        sycl_queue = queue

    result = zeros_on_backend((2, 3), np.complex128, FakeDPNP(), Reference())
    assert result.shape == (2, 3)
    assert calls == [((2, 3), np.complex128, queue)]


def test_emff_and_soft_factor_have_no_backend_specific_host_conversion():
    emff = (
        REPO_ROOT / "pyquda_measurement_utils/pion_EMFF_vibe_develop.py"
    ).read_text(encoding="utf-8")
    soft = (
        REPO_ROOT / "pyquda_measurement_utils/pion_soft_factor_vibe_develop.py"
    ).read_text(encoding="utf-8")

    assert "xp.asnumpy" not in emff
    assert "xp.zeros" not in emff
    assert "xp.asnumpy" not in soft
    assert "def _to_numpy" not in soft
    assert "first_gamma.device" not in soft
    assert "array_to_numpy" in emff
    assert "array_to_numpy" in soft


def test_response_apps_require_explicit_source_relative_conversion_and_tagging():
    first_order = (
        REPO_ROOT
        / "application/EMFF_pion_background_response/perlmutter"
        / "Pyquda_pion_EMFF_background_response.py"
    ).read_text(encoding="utf-8")
    current_current = (
        REPO_ROOT
        / "application/pion_current_current_response/perlmutter"
        / "Pyquda_pion_current_current_response.py"
    ).read_text(encoding="utf-8")

    for source in (first_order, current_current):
        assert 'parser.add_argument("--src_pos"' in source
        assert "source_time=src_pos[3]" in source
        assert "roll_to_source_relative(" in source
        assert '"source_position": np.asarray(src_pos' in source
        assert '"time_axis": "source_relative"' in source
        assert "f\".x{src_pos[0]}y{src_pos[1]}z{src_pos[2]}t{src_pos[3]}\"" in source
        assert "getMPIComm().bcast(c2_corr, root=0)" in source

    assert "getMPIComm().bcast(c3_by_src[src_gamma], root=0)" in first_order
    assert "getMPIComm().bcast(response_corr, root=0)" in first_order
    assert "getMPIComm().bcast(cc_response_corr, root=0)" in current_current
