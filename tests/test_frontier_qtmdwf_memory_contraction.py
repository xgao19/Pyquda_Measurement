import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyquda_utils import gamma

import pyquda_measurement_utils.pion_utils_vibe_develop as pion_utils
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_gamma_scan,
    gamma_stack,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTIER_ENTRY = (
    REPO_ROOT / "application/qTMDWF_CG/Frontier/pyquda_qTMDWF.py"
)
AURORA_ENTRY = (
    REPO_ROOT / "application/qTMDWF_CG/Aurora/pyquda_qTMDWF.py"
)


class _Propagator:
    def __init__(self, data):
        self.data = data
        self.latt_info = SimpleNamespace(size=[2, 2, 2, 3])


def _gamma_matrix(gamma_like):
    return gamma_like.matrix if hasattr(gamma_like, "matrix") else gamma_like


def _old_stacked_reference(backward, forward, phases, source_gamma):
    gamma5 = np.asarray(_gamma_matrix(gamma.gamma(15)))
    source_gamma = np.asarray(_gamma_matrix(source_gamma))
    sink_gammas = gamma_stack(forward)

    forward_source = np.einsum(
        "wtzyxilab,lj->wtzyxijab",
        forward,
        source_gamma,
        optimize=True,
    )
    gamma_forward_source = np.einsum(
        "gim,wtzyxmjab->gwtzyxijab",
        sink_gammas,
        forward_source,
        optimize=True,
    )
    backward_line = np.einsum(
        "ki,wtzyxklab,jl->wtzyxjiba",
        gamma5,
        backward.conj(),
        gamma5,
        optimize=True,
    )
    gamma_site = np.einsum(
        "wtzyxjiba,gwtzyxijab->gwtzyx",
        backward_line,
        gamma_forward_source,
        optimize=True,
    )
    return np.einsum(
        "qwtzyx,gwtzyx->qgt", phases, gamma_site, optimize=True
    )


def _random_inputs(xp, seed=1701):
    rng = np.random.default_rng(seed)
    prop_shape = (1, 3, 2, 2, 2, 4, 4, 3, 3)
    phase_shape = (2, 1, 3, 2, 2, 2)
    forward = rng.normal(size=prop_shape) + 1j * rng.normal(size=prop_shape)
    backward = rng.normal(size=prop_shape) + 1j * rng.normal(size=prop_shape)
    phases = rng.normal(size=phase_shape) + 1j * rng.normal(size=phase_shape)
    return (
        xp.asarray(forward),
        xp.asarray(backward),
        xp.asarray(phases),
        forward,
        backward,
        phases,
    )


def _run_helper(monkeypatch, forward, backward, phases, source_gamma_label):
    monkeypatch.setattr(pion_utils.core, "gatherLattice", lambda values, axes: values)
    return contract_pion_gamma_scan(
        SimpleNamespace(size=[2, 2, 2, 3]),
        _Propagator(forward),
        _Propagator(backward),
        phases,
        [source_gamma_label],
    )[source_gamma_label]


@pytest.mark.parametrize("source_gamma_id", [15, 1, 8])
def test_two_stage_qtmdwf_contraction_matches_stacked_reference(
    monkeypatch, source_gamma_id
):
    forward, backward, phases, forward_np, backward_np, phases_np = (
        _random_inputs(np)
    )
    source_gamma = gamma.gamma(source_gamma_id)
    expected = _old_stacked_reference(
        backward_np, forward_np, phases_np, source_gamma
    )
    source_label = {15: "5", 1: "X", 8: "T"}[source_gamma_id]
    actual = _run_helper(monkeypatch, forward, backward, phases, source_label)
    np.testing.assert_allclose(actual, expected.transpose(1, 0, 2), rtol=1e-13, atol=1e-13)
    assert actual.shape == (16, 2, 3)


def test_two_stage_qtmdwf_contraction_cupy_if_available(monkeypatch):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device is available")

    forward, backward, phases, forward_np, backward_np, phases_np = (
        _random_inputs(cupy)
    )
    expected = _old_stacked_reference(
        backward_np, forward_np, phases_np, gamma.gamma(15)
    )
    actual = _run_helper(
        monkeypatch, forward, backward, phases, "5"
    )
    np.testing.assert_allclose(actual, expected.transpose(1, 0, 2), rtol=1e-13, atol=1e-13)


def test_two_stage_qtmdwf_contraction_dpnp_if_available(monkeypatch):
    dpnp = pytest.importorskip("dpnp")
    try:
        forward, backward, phases, forward_np, backward_np, phases_np = (
            _random_inputs(dpnp)
        )
    except Exception as error:
        pytest.skip(f"no usable dpnp device/queue: {error}")

    expected = _old_stacked_reference(
        backward_np, forward_np, phases_np, gamma.gamma(15)
    )
    actual = _run_helper(
        monkeypatch, forward, backward, phases, "5"
    )
    np.testing.assert_allclose(actual, expected.transpose(1, 0, 2), rtol=1e-13, atol=1e-13)


def test_platform_qtmdwf_entries_share_memory_light_kernel():
    helper_source = inspect.getsource(
        pion_utils.contract_pion_gamma_scan_from_backward_line
    )

    for entry in (FRONTIER_ENTRY, AURORA_ENTRY):
        entry_source = entry.read_text()
        assert "G16_fw_Gsrc" not in entry_source
        assert "fw_Gsrc" not in entry_source
        assert "gwtzyxijab" not in entry_source
        assert "run_qtmdwf_sources" in entry_source
    assert "gwtzyxijab" not in helper_source
