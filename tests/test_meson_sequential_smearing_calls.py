import ast
from pathlib import Path


CALL_SITES = [
    "application/EMFF_pion/perlmutter/Pyquda_pion_EMFF.py",
    "application/pion_TMD_CG/perlmutter/Pyquda_pion_TMD_CG.py",
    "application/pion_TMD/perlmutter/Pyquda_pion_TMD.py",
    "pyquda_measurement_utils/pion_EMT_vibe_develop.py",
]


def _meson_seq_calls(path):
    tree = ast.parse(Path(path).read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_meson_bw_seq_pyquda":
            calls.append(node)
    return calls


def test_meson_sequential_source_calls_pass_smearing_arguments():
    repo_root = Path(__file__).resolve().parents[1]

    for relpath in CALL_SITES:
        calls = _meson_seq_calls(repo_root / relpath)
        assert calls, f"No create_meson_bw_seq_pyquda call found in {relpath}"
        for call in calls:
            assert len(call.args) >= 8, f"Sequential source call in {relpath} does not pass smearing width and boost"


def test_meson_sequential_source_helper_keeps_optional_smearing_interface():
    repo_root = Path(__file__).resolve().parents[1]
    helper = repo_root / "pyquda_measurement_utils" / "bw_seq_pyquda.py"
    tree = ast.parse(helper.read_text())

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_meson_bw_seq_pyquda":
            arg_names = [arg.arg for arg in node.args.args]
            assert arg_names[-2:] == ["sm_width", "sm_boost"]
            return

    raise AssertionError("create_meson_bw_seq_pyquda was not found")
