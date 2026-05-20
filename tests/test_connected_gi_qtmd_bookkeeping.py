import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import create_gi_qtmd_wilsonline_index_lists


def _source_uses_shared_helper(path, method_name):
    text = Path(path).read_text()
    marker = f"def {method_name}(self):"
    start = text.index(marker)
    end = text.find("\n    def ", start + len(marker))
    body = text[start:] if end == -1 else text[start:end]
    return "create_gi_qtmd_wilsonline_index_lists(self.eta, self.b_z, self.b_T)" in body


def test_connected_gi_qtmd_uses_shared_wilson_index_bookkeeping():
    repo_root = Path(__file__).resolve().parents[1]

    assert _source_uses_shared_helper(
        repo_root / "pyquda_measurement_utils" / "pion_qTMD_vibe_develop.py",
        "create_TMD_Wilsonline_index_list_GI",
    )
    assert _source_uses_shared_helper(
        repo_root / "pyquda_measurement_utils" / "proton_qTMD_pyquda.py",
        "create_TMD_Wilsonline_index_list_GI",
    )


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


if __name__ == "__main__":
    test_connected_gi_qtmd_uses_shared_wilson_index_bookkeeping()
    test_connected_gi_qtmd_fixed_length_pdf_limit_indices()
    print("[connected GI qTMD bookkeeping sanity check]")
    print("PASS")
