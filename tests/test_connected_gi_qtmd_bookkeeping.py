from pathlib import Path

from pyquda_measurement_utils.qtmd_operator_utils import (
    create_gi_qtmd_wilsonline_index_lists,
)


def test_connected_gi_qtmd_uses_shared_wilson_index_bookkeeping():
    repo_root = Path(__file__).resolve().parents[1]
    pion_application = (
        repo_root
        / "application/pion_TMD/perlmutter/Pyquda_pion_TMD.py"
    ).read_text()
    proton_runner = (
        repo_root / "application/nucleon_TMD/shared_runner.py"
    ).read_text()
    assert "create_gi_qtmd_wilsonline_index_lists" in pion_application
    assert "create_gi_qtmd_wilsonline_index_lists" in proton_runner


def test_connected_gi_qtmd_fixed_length_pdf_limit_indices():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([0, 1, 2], 4, 1)

    assert [0, 0, 0, 0] in dir0
    assert [0, 2, 1, 0] in dir0
    assert [0, -2, 1, 0] in dir0
    assert [0, 4, 2, 0] in dir0
    assert [0, -4, 2, 0] in dir0
    assert [0, 4, 1, 0] not in dir0
    assert [0, 2, 1, 1] in dir1
    assert [0, -2, 1, 1] in dir1


def test_connected_production_has_no_disconnected_module_dependency():
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        "pyquda_measurement_utils/pion_qTMD_vibe_develop.py",
        "pyquda_measurement_utils/proton_qTMD_pyquda.py",
        "pyquda_measurement_utils/pion_EMT_vibe_develop.py",
        "pyquda_measurement_utils/proton_EMT_vibe_develop.py",
        "pyquda_measurement_utils/proton_utils_vibe_develop.py",
        "application/nucleon_TMD/shared_runner.py",
    ]
    for relative_path in paths:
        assert "pyquda_measurement_utils.Disconnected_" not in (
            repo_root / relative_path
        ).read_text()


def test_backend_conversion_and_operator_utility_are_neutral():
    repo_root = Path(__file__).resolve().parents[1]
    definitions = []
    for path in (repo_root / "pyquda_measurement_utils").glob("*.py"):
        if "def array_to_numpy(" in path.read_text():
            definitions.append(path.name)
    assert definitions == ["tools.py"]

    operator_source = (
        repo_root
        / "pyquda_measurement_utils/qtmd_operator_utils.py"
    ).read_text()
    assert "Disconnected_" not in operator_source
    assert "pion_qTMD" not in operator_source
    assert "proton_qTMD" not in operator_source


if __name__ == "__main__":
    test_connected_gi_qtmd_uses_shared_wilson_index_bookkeeping()
    test_connected_gi_qtmd_fixed_length_pdf_limit_indices()
    print("[connected GI qTMD bookkeeping sanity check]")
    print("PASS")
