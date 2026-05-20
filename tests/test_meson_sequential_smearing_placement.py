import ast
import inspect

from pyquda_measurement_utils.bw_seq_pyquda import create_meson_bw_seq_pyquda


def _call_positions(source, function_name):
    tree = ast.parse(source)
    positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == function_name:
                positions.append(node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == function_name:
                positions.append(node.lineno)
    return positions


def test_meson_sequential_source_smearing_is_applied_to_rhs_before_inversion():
    source = inspect.getsource(create_meson_bw_seq_pyquda)
    seq_line = _call_positions(source, "sequential12")[0]
    phase_line = _call_positions(source, "getPhases")[0]
    smearing_line = _call_positions(source, "boosted_smearing")[0]
    invert_line = _call_positions(source, "invertPropagator")[0]

    assert seq_line < phase_line < smearing_line < invert_line
    assert "src_seq = boosted_smearing(src_seq, w=sm_width, boost=sm_boost)" in source
    assert "return core.invertPropagator(dirac, src_seq, 1, 0)" in source


def test_meson_sequential_source_does_not_smear_current_insertion_line_inside_helper():
    source = inspect.getsource(create_meson_bw_seq_pyquda)

    assert "boosted_smearing(prop" not in source
    assert source.count("boosted_smearing(") == 1
