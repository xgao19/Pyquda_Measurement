import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop as qtmd
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    G5,
    _gamma_matrix,
    my_gammas,
    my_pyquda_gammas,
)


def test_complete_basis_estimator_targets_gamma_o_dinv_at_fixed_time_and_momentum():
    rng = np.random.default_rng(20260713)
    lt, lx, ns = 2, 2, 4
    n_sites = lt * lx
    n = n_sites * ns

    dinv = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    operator_sites = rng.normal(size=(n_sites, n_sites)) + 1j * rng.normal(
        size=(n_sites, n_sites)
    )
    operator = np.kron(operator_sites, np.eye(ns))
    gamma = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
    gamma_full = np.kron(np.eye(n_sites), gamma)

    tau = 1
    spatial_phase = np.asarray([1.0, 1.0j])
    weights = np.zeros((lt, lx), dtype=np.complex128)
    weights[tau] = spatial_phase
    projector_phase = np.diag(np.repeat(weights.reshape(-1), ns))

    estimate = 0.0j
    for source_index in range(n):
        xi = np.zeros(n, dtype=np.complex128)
        xi[source_index] = 1.0
        shifted_eta = operator @ (dinv @ xi)
        local = qtmd._contract_xi_dagger_gamma_shifted_eta(
            xi.reshape(1, lt, 1, 1, lx, ns, 1),
            gamma[None],
            shifted_eta.reshape(1, lt, 1, 1, lx, ns, 1),
        )
        estimate += np.sum(spatial_phase * local[0, 0, tau, 0, 0])

    target = np.trace(projector_phase @ gamma_full @ operator @ dinv)
    legacy_target = np.trace(
        dinv.conj().T @ projector_phase @ gamma_full @ operator
    )
    np.testing.assert_allclose(estimate, target, atol=1e-12, rtol=1e-12)
    assert not np.isclose(estimate, legacy_target)


class _FakeLatticeInfo:
    mpi_rank = 1


class _FakeFermion:
    def __init__(self, data, label, shift_log):
        self.data = np.asarray(data, dtype=np.complex128)
        self.label = label
        self.shift_log = shift_log

    def copy(self):
        return _FakeFermion(self.data.copy(), self.label, self.shift_log)

    def shift(self, steps, direction):
        if steps:
            self.shift_log.append((self.label, int(steps), int(direction)))
            scale = 2 if int(steps) > 0 else 3
            return _FakeFermion(
                self.data * scale, self.label, self.shift_log
            )
        return self


class _FakePureGauge:
    def __init__(self, shift_log):
        self.shift_log = shift_log

    def covDev(self, fermion, direction):
        return fermion.shift(1, direction)


class _FakeGauge:
    def __init__(self, shift_log):
        self.pure_gauge = _FakePureGauge(shift_log)


@pytest.mark.parametrize(
    "operator_kind,w_index,expected_shift,scale",
    [
        ("CG_qTMD", [1, 0, 0, 0], ("eta", 1, 0), 2),
        ("CG_PDF", [0, 1, 0, 0], ("eta", 1, 2), 2),
        ("CG_PDF", [0, -1, 0, 0], ("eta", -1, 2), 3),
        ("GI_PDF", [0, 1, 0, 0], ("eta", 1, 2), 2),
        ("GI_qTMD", [1, 0, 0, 0], ("eta", 1, 3), 2),
    ],
)
def test_operator_is_applied_to_eta_and_xi_remains_on_the_left(
    monkeypatch, operator_kind, w_index, expected_shift, scale
):
    shape = (1, 1, 1, 1, 1, 4, 1)
    shift_log = []
    xi_data = np.asarray([1 + 2j, 2 - 1j, -1 + 3j, 4 + 0.5j]).reshape(shape)
    eta_data = np.asarray([3 - 1j, -2 + 4j, 0.5 + 2j, 1 - 3j]).reshape(shape)
    xi = _FakeFermion(xi_data, "xi", shift_log)
    eta = _FakeFermion(eta_data, "eta", shift_log)

    monkeypatch.setattr(qtmd, "gamma_stack", lambda _data: np.eye(4)[None])
    monkeypatch.setattr(qtmd.core, "gatherLattice", lambda values, _axes: values)
    monkeypatch.setattr(
        qtmd,
        "create_fermion_TMD_GI_from_link",
        lambda _link, fermion, _w_index: fermion.shift(1, 3),
    )

    measurement = object.__new__(qtmd.DisconnectedQuarkqTMD1pt)
    loops = measurement._contract_one_operator_list(
        _FakeLatticeInfo(),
        _FakeGauge(shift_log),
        eta,
        xi,
        np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128),
        [w_index],
        operator_kind,
        staple_links={tuple(w_index): object()} if operator_kind == "GI_qTMD" else None,
    )

    expected = np.vdot(xi_data.reshape(-1), scale * eta_data.reshape(-1))
    legacy = np.vdot(eta_data.reshape(-1), scale * xi_data.reshape(-1))
    np.testing.assert_allclose(loops[0, 0, 0, 0], expected)
    assert not np.isclose(loops[0, 0, 0, 0], legacy)
    assert shift_log == [expected_shift]


def test_gamma5_conjugation_explains_legacy_channel_signs():
    gamma5 = np.asarray(_gamma_matrix(G5))
    even = {"5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"}
    for label, gamma_like in zip(my_gammas, my_pyquda_gammas):
        gamma = np.asarray(_gamma_matrix(gamma_like))
        sign = 1 if label in even else -1
        np.testing.assert_allclose(gamma5 @ gamma @ gamma5, sign * gamma)
