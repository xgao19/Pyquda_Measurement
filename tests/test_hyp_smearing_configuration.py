import ast
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ROOT = REPO_ROOT / "application"


def _python_sources():
    return sorted(APPLICATION_ROOT.rglob("*.py"))


def _tree(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(path.read_text())


def test_all_active_hyp_smearing_uses_fixed_four_dimensional_dir_ignore():
    calls = []
    for path in _python_sources():
        tree = _tree(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "hypSmear":
                continue
            calls.append((path, node))

    assert calls
    for path, call in calls:
        assert len(call.args) >= 5, path
        dir_ignore = call.args[4]
        assert (
            isinstance(dir_ignore, ast.UnaryOp)
            and isinstance(dir_ignore.op, ast.USub)
            and isinstance(dir_ignore.operand, ast.Constant)
            and dir_ignore.operand.value == 1
        ), f"{path} must call hypSmear(..., dir_ignore=-1)"


def test_hyp_dir_ignore_has_no_environment_or_cli_override():
    combined_source = "\n".join(path.read_text() for path in _python_sources())
    assert "FLOWED_RINGED_HYP_PROJECT" not in combined_source
    assert "EMT_PROTON_HYP_PROJECT" not in combined_source
    assert "--hyp-project" not in combined_source
    assert "--hyp-dir-ignore" not in combined_source


def test_hyp_provenance_names_dir_ignore_instead_of_projection():
    hyp_metadata = []
    for path in _python_sources():
        tree = _tree(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("HYP(1,0.75,0.6,0.3,")
            ):
                hyp_metadata.append((path, node.value))

    assert hyp_metadata
    for path, value in hyp_metadata:
        assert value == "HYP(1,0.75,0.6,0.3,dir_ignore=-1)", (
            path,
            value,
        )
