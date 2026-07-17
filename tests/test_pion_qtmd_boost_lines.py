import ast
import inspect
from pathlib import Path

import numpy as np

import pyquda_measurement_utils.pion_utils_vibe_develop as pion_utils
from pyquda_measurement_utils.pion_qTMD_vibe_develop import pion_TMD


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "application/pion_TMD/perlmutter/Pyquda_pion_TMD.py"


class _TaggedField:
    def __init__(self, identity):
        self.identity = identity

    def copy(self):
        return _TaggedField(self.identity)


def _install_source_fakes(monkeypatch):
    source_calls = []
    inversion_calls = []

    def point_source(_latt_info, source_type, position):
        assert source_type == "point"
        field = _TaggedField(("point", len(source_calls), tuple(position)))
        source_calls.append(field.identity)
        return field

    def smear(field, *, w, boost):
        return _TaggedField(("smeared", field.identity, float(w), tuple(boost)))

    def invert(_dirac, field, _source_type, _solution_type):
        inversion_calls.append(field.identity)
        return _TaggedField(("propagator", field.identity))

    monkeypatch.setattr(pion_utils.source, "propagator", point_source)
    monkeypatch.setattr(pion_utils, "boosted_smearing", smear)
    monkeypatch.setattr(pion_utils.core, "invertPropagator", invert)
    return source_calls, inversion_calls


def test_shared_source_builder_reuses_equal_boost_inversion(monkeypatch):
    source_calls, inversion_calls = _install_source_fakes(monkeypatch)
    spectator, active = pion_utils.build_pion_source_propagators(
        object(),
        object(),
        [1, 2, 3, 4],
        gaussian_smearing=True,
        width=2.0,
        pos_boost=[0, 0, 1],
        neg_boost=[0, 0, 1],
    )

    assert len(source_calls) == 1
    assert len(inversion_calls) == 1
    assert spectator is not active
    assert spectator.identity == active.identity


def test_shared_source_builder_inverts_unequal_boost_lines(monkeypatch):
    source_calls, inversion_calls = _install_source_fakes(monkeypatch)
    spectator, active = pion_utils.build_pion_source_propagators(
        object(),
        object(),
        [1, 2, 3, 4],
        gaussian_smearing=True,
        width=2.0,
        pos_boost=[0, 0, 1],
        neg_boost=[0, 0, -1],
    )

    assert len(source_calls) == 2
    assert len(inversion_calls) == 2
    assert inversion_calls[0][-1] == (0, 0, 1)
    assert inversion_calls[1][-1] == (0, 0, -1)
    assert spectator.identity != active.identity


def _named_calls(source, function_name):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == function_name
    ]


def test_canonical_driver_routes_spectator_and_active_lines():
    source = DRIVER.read_text()

    assert "spectator_prop, active_prop = build_pion_source_propagators(" in source
    assert "spectator_prop,\n        active_prop," in source
    assert "spectator_sink_prop = boosted_smearing(\n        spectator_prop.copy()," in source
    assert 'boost=parameters["pos_boost"]' in source
    assert "dirac,\n        spectator_sink_prop," in source
    assert 'parameters["neg_boost"],' in source

    for method_name in ("contract_qTMD_CG", "contract_qTMD_GI", "contract_PDF"):
        calls = _named_calls(source, method_name)
        assert calls
        for call in calls:
            positional_names = [
                arg.id for arg in call.args if isinstance(arg, ast.Name)
            ]
            assert "active_prop" in positional_names
            assert "spectator_prop" not in positional_names

    assert '"operator_insertion_line": "neg_boost"' in source
    assert '"boost_line_convention": "pos_spectator_neg_active"' in source
    assert '"save_propagators"' not in source


def test_qtmd_contraction_interfaces_name_the_active_line():
    for method in (
        pion_TMD.contract_qTMD_CG,
        pion_TMD.contract_qTMD_GI,
        pion_TMD.contract_PDF,
    ):
        signature = inspect.signature(method)
        assert "active_prop" in signature.parameters
        assert "prop_f" not in signature.parameters


def test_qtmdwf_and_qda_keep_independent_source_lines():
    qda = (
        REPO_ROOT / "application/qDA/frontier_charm/pyquda_DA_k6.py"
    ).read_text()
    assert "srcDp = boosted_smearing" in qda
    assert "srcDm = boosted_smearing" in qda
    assert "propag_f = core.invertPropagator(dirac, srcDp" in qda
    assert "propag_b = core.invertPropagator(dirac, srcDm" in qda

    runner = (
        REPO_ROOT / "application/qTMDWF_CG/qTMDWF_runner.py"
    ).read_text()
    helper = inspect.getsource(pion_utils.build_pion_source_propagators)
    assert "build_pion_source_propagators(" in runner
    assert "pos_boost=measurement.pos_boost" in runner
    assert "neg_boost=measurement.neg_boost" in runner
    assert "src_positive" in helper
    assert "src_negative" in helper
    assert "prop_positive.copy()" in helper


def _random_su3(rng, count):
    matrices = []
    for _ in range(count):
        matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q, r = np.linalg.qr(matrix)
        q = q @ np.diag(np.diag(r) / np.abs(np.diag(r))).conj()
        q /= np.linalg.det(q) ** (1.0 / 3.0)
        matrices.append(q)
    return np.asarray(matrices)


def test_gi_active_line_transport_is_locally_gauge_invariant():
    """A link-transported active endpoint is invariant; a CG shift is not."""
    rng = np.random.default_rng(20260716)
    nsite = 7
    omega = _random_su3(rng, nsite)
    link = _random_su3(rng, nsite)
    active = rng.normal(size=(nsite, 4, 3)) + 1j * rng.normal(
        size=(nsite, 4, 3)
    )
    sequential = rng.normal(size=(nsite, 4, 3)) + 1j * rng.normal(
        size=(nsite, 4, 3)
    )
    gamma_spin = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    phase = np.exp(2j * np.pi * np.arange(nsite) / nsite)
    endpoint = np.roll(np.arange(nsite), -1)

    active_g = np.einsum("xab,xsb->xsa", omega, active)
    sequential_g = np.einsum(
        "xsa,xab->xsb", sequential, omega.conj().swapaxes(-1, -2)
    )
    link_g = np.einsum(
        "xab,xbc,xcd->xad",
        omega,
        link,
        omega[endpoint].conj().swapaxes(-1, -2),
    )

    def corr(left, right, transporter):
        shifted = right[endpoint]
        if transporter is not None:
            shifted = np.einsum("xab,xsb->xsa", transporter, shifted)
        site = np.einsum("xsa,sr,xra->x", left, gamma_spin, shifted)
        return np.einsum("x,x->", phase, site)

    gi_before = corr(sequential, active, link)
    gi_after = corr(sequential_g, active_g, link_g)
    np.testing.assert_allclose(gi_before, gi_after, rtol=1e-12, atol=1e-12)

    cg_before = corr(sequential, active, None)
    cg_after = corr(sequential_g, active_g, None)
    assert not np.allclose(cg_before, cg_after, rtol=1e-8, atol=1e-8)
