import ast
import sys
from pathlib import Path
from unittest import SkipTest

try:
    from pyquda_measurement_utils.pion_soft_factor_vibe_develop import as_momentum_3, momentum_tag
except Exception as err:
    raise SkipTest(f"pion soft-factor helpers require the PyQUDA Python environment: {err}") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relpath):
    return (REPO_ROOT / relpath).read_text()


def test_pion_soft_factor_momentum_tags_are_stable():
    assert as_momentum_3([1, 2, 3, 0]) == [1, 2, 3]
    assert as_momentum_3([-1, 0, 5]) == [-1, 0, 5]
    assert momentum_tag([0, 0, -4]) == "qx0qy0qz-4"


def test_pion_soft_factor_prop_stage_saves_plus_and_minus_quark_momenta():
    tree = ast.parse(_source("application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_prop.py"))
    text = _source("application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_prop.py")

    assert "momenta_to_save.append(mom)" in text
    assert "momenta_to_save.append([-mom[0], -mom[1], -mom[2]])" in text
    assert "dict.fromkeys(tuple(mom) for mom in momenta_to_save)" in text

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "momenta_to_save" in names


def test_pion_soft_factor_contract_stage_loads_required_source_and_sink_momenta():
    text = _source("application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_contract.py")

    assert "prop_fw_tag = get_pion_soft_factor_prop_file_tag" in text
    assert "prop_bw_src_tag = get_pion_soft_factor_prop_file_tag" in text
    assert "prop_sink_fw_tag = get_pion_soft_factor_prop_file_tag" in text
    assert "prop_sink_bw_tag = get_pion_soft_factor_prop_file_tag" in text
    assert "[-quark_mom_bw[0], -quark_mom_bw[1], -quark_mom_bw[2]]" in text
    assert "[-quark_mom_fw[0], -quark_mom_fw[1], -quark_mom_fw[2]]" in text
