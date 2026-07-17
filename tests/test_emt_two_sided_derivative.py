"""Algebraic proof tests for the connected/disconnected EMT derivative."""

import numpy as np

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _MomentumProjectors,
    _build_momentum_projectors,
)
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA5_HERMITICITY_PARTNERS,
    GAMMA5_HERMITICITY_SIGNS,
    GAMMA_LABELS,
    VECTOR_GAMMA_POSITIONS,
    gamma_matrices_numpy,
)


def _algebra(seed=7301):
    rng = np.random.default_rng(seed)
    nsite = 5
    nspin = 4
    gammas = gamma_matrices_numpy()
    gamma5 = gammas[GAMMA_LABELS.index("5")]

    raw_d = rng.normal(size=(nsite, nsite)) + 1j * rng.normal(
        size=(nsite, nsite)
    )
    derivative = np.kron(raw_d - raw_d.conj().T, np.eye(nspin))
    gamma5_full = np.kron(np.eye(nsite), gamma5)

    raw_s = rng.normal(size=derivative.shape) + 1j * rng.normal(
        size=derivative.shape
    )
    propagator = 0.5 * (
        raw_s + gamma5_full @ raw_s.conj().T @ gamma5_full
    )

    phases = np.exp(2j * np.pi * np.arange(nsite) / nsite)
    projector_q = np.kron(np.diag(phases), np.eye(nspin))
    projector_minus_q = projector_q.conj()
    projector_full = np.eye(nsite * nspin, dtype=np.complex128)
    return gammas, gamma5_full, derivative, propagator, projector_q, projector_minus_q, projector_full


def _full_gamma(matrix, nsite):
    return np.kron(np.eye(nsite), matrix)


def test_gamma5_hermiticity_map_is_exact_and_involutive():
    gammas = gamma_matrices_numpy()
    gamma5 = gammas[GAMMA_LABELS.index("5")]
    for idx, matrix in enumerate(gammas):
        partner = GAMMA5_HERMITICITY_PARTNERS[idx]
        sign = GAMMA5_HERMITICITY_SIGNS[idx]
        np.testing.assert_allclose(
            gamma5 @ matrix.conj().T @ gamma5,
            sign * gammas[partner],
            rtol=0,
            atol=1e-13,
        )
        assert GAMMA5_HERMITICITY_PARTNERS[partner] == idx
        assert sign * GAMMA5_HERMITICITY_SIGNS[partner] == 1


def test_gamma5_reconstruction_equals_exact_two_sided_trace():
    gammas, _, derivative, propagator, projector_q, projector_minus_q, _ = _algebra()
    nsite = projector_q.shape[0] // 4
    for idx, matrix in enumerate(gammas):
        gamma = _full_gamma(matrix, nsite)
        partner = GAMMA5_HERMITICITY_PARTNERS[idx]
        sign = GAMMA5_HERMITICITY_SIGNS[idx]
        gamma_sharp = sign * _full_gamma(gammas[partner], nsite)

        right = np.trace(projector_q @ gamma @ derivative @ propagator)
        left = np.trace(derivative @ projector_q @ gamma @ propagator)
        target = -0.5 * (right + left)
        sharp_at_minus_q = np.trace(
            projector_minus_q @ gamma_sharp @ derivative @ propagator
        )
        reconstructed = -0.5 * (right - sharp_at_minus_q.conj())
        np.testing.assert_allclose(reconstructed, target, rtol=1e-13, atol=1e-13)


def test_complete_basis_noise_side_reference_equals_trace_target():
    gammas, _, derivative, propagator, projector_q, _, _ = _algebra(7302)
    nsite = projector_q.shape[0] // 4
    gamma = _full_gamma(gammas[GAMMA_LABELS.index("X")], nsite)
    target = -0.5 * (
        np.trace(projector_q @ gamma @ derivative @ propagator)
        + np.trace(derivative @ projector_q @ gamma @ propagator)
    )

    estimate = 0.0j
    for source in np.eye(derivative.shape[0], dtype=np.complex128):
        solution = propagator @ source
        right = source.conj() @ projector_q @ gamma @ derivative @ solution
        noise_side = (
            derivative @ source
        ).conj() @ projector_q @ gamma @ solution
        estimate += -0.5 * right + 0.5 * noise_side
    np.testing.assert_allclose(estimate, target, rtol=1e-13, atol=1e-13)


def test_old_one_sided_trace_only_works_when_projector_commutes_with_derivative():
    gammas, _, derivative, propagator, projector_q, _, projector_full = _algebra(7303)
    nsite = projector_q.shape[0] // 4
    gamma = _full_gamma(gammas[GAMMA_LABELS.index("T")], nsite)

    def old_and_target(projector):
        right = np.trace(projector @ gamma @ derivative @ propagator)
        left = np.trace(derivative @ projector @ gamma @ propagator)
        return -right, -0.5 * (right + left)

    old_full, target_full = old_and_target(projector_full)
    np.testing.assert_allclose(old_full, target_full, rtol=1e-13, atol=1e-13)
    old_q, target_q = old_and_target(projector_q)
    assert not np.isclose(old_q, target_q, rtol=1e-6, atol=1e-8)


def test_fixed_time_temporal_derivative_requires_the_left_term_at_zero_spatial_q():
    gammas, _, _, propagator, _, _, _ = _algebra(7310)
    ntime = 5
    nspin = 4
    forward = np.zeros((ntime, ntime), dtype=np.complex128)
    for t in range(ntime):
        forward[t, (t + 1) % ntime] = 1.0
    temporal_derivative = np.kron(forward - forward.conj().T, np.eye(nspin))
    fixed_time = np.kron(
        np.diag([1.0, 0.0, 0.0, 0.0, 0.0]), np.eye(nspin)
    )
    gamma = _full_gamma(gammas[GAMMA_LABELS.index("T")], ntime)
    right = np.trace(fixed_time @ gamma @ temporal_derivative @ propagator)
    left = np.trace(temporal_derivative @ fixed_time @ gamma @ propagator)
    old = -right
    target = -0.5 * (right + left)
    assert not np.isclose(old, target, rtol=1e-6, atol=1e-8)


def test_momentum_projectors_add_only_missing_negative_momenta(monkeypatch):
    calls = []

    class FakeMomentumPhase:
        def __init__(self, latt_info):
            self.latt_info = latt_info

        def getPhases(self, momenta, origin):
            calls.append((tuple(map(tuple, momenta)), tuple(origin)))
            return tuple(map(tuple, momenta))

    module = __import__(
        "pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop",
        fromlist=["phase"],
    )
    monkeypatch.setattr(module.phase, "MomentumPhase", FakeMomentumPhase)
    projectors = _build_momentum_projectors(
        object(),
        [(0, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0)],
        (2, 3, 4, 0),
    )
    assert projectors.requested_indices == (0, 1, 2)
    assert projectors.negative_indices == (0, 3, 4)
    assert calls == [
        (((0, 0, 0), (1, 0, 0), (0, -1, 0)), (2, 3, 4, 0)),
        (
            ((0, 0, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)),
            (2, 3, 4, 0),
        ),
    ]


def test_production_projection_helper_matches_exact_trace_and_connected_sign():
    gammas, _, derivative, propagator, projector_q, projector_minus_q, _ = _algebra(7304)
    nsite = projector_q.shape[0] // 4
    right_diff = np.empty((16, 2, 1), dtype=np.complex128)
    connected_targets = np.empty(16, dtype=np.complex128)
    for idx, matrix in enumerate(gammas):
        gamma = _full_gamma(matrix, nsite)
        right = np.trace(projector_q @ gamma @ derivative @ propagator)
        right_minus = np.trace(
            projector_minus_q @ gamma @ derivative @ propagator
        )
        left = np.trace(derivative @ projector_q @ gamma @ propagator)
        # Production covDev field is (Dplus-Dminus)=2*D.
        right_diff[idx, :, 0] = (2.0 * right, 2.0 * right_minus)
        connected_targets[idx] = 0.5 * (right + left)

    projectors = _MomentumProjectors(None, None, (0,), (1,))
    reconstructed = (
        EMTDisconnectedQuark1pt._closed_loop_derivative_from_right_projection(
            right_diff, projectors
        )[:, 0, 0]
    )
    np.testing.assert_allclose(
        reconstructed, -connected_targets, rtol=1e-13, atol=1e-13
    )
    # Connected uses the same two line-level derivative terms but has no
    # closed-fermion-loop Wick minus.
    np.testing.assert_allclose(
        -reconstructed, connected_targets, rtol=1e-13, atol=1e-13
    )


def test_two_sided_trace_is_invariant_under_a_local_similarity_transform():
    rng = np.random.default_rng(7311)
    nsite, nspin, ncolor = 3, 4, 3
    gammas = gamma_matrices_numpy()
    gamma5 = gammas[GAMMA_LABELS.index("5")]
    dimension = nsite * nspin * ncolor
    raw_d = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    derivative = raw_d - raw_d.conj().T
    gamma5_full = np.kron(np.eye(nsite), np.kron(gamma5, np.eye(ncolor)))
    raw_s = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    propagator = 0.5 * (
        raw_s + gamma5_full @ raw_s.conj().T @ gamma5_full
    )
    gamma = np.kron(
        np.eye(nsite),
        np.kron(gammas[GAMMA_LABELS.index("X5")], np.eye(ncolor)),
    )
    projector_q = np.kron(
        np.diag(np.exp(2j * np.pi * np.arange(nsite) / nsite)),
        np.eye(nspin * ncolor),
    )
    target = -0.5 * (
        np.trace(projector_q @ gamma @ derivative @ propagator)
        + np.trace(derivative @ projector_q @ gamma @ propagator)
    )
    omega = np.zeros((dimension, dimension), dtype=np.complex128)
    for site in range(nsite):
        raw = rng.normal(size=(ncolor, ncolor)) + 1j * rng.normal(
            size=(ncolor, ncolor)
        )
        q, r = np.linalg.qr(raw)
        q *= np.diag(r) / np.abs(np.diag(r))
        block = np.kron(np.eye(nspin), q)
        start = nspin * ncolor * site
        omega[start:start + nspin * ncolor, start:start + nspin * ncolor] = block
    derivative_g = omega @ derivative @ omega.conj().T
    propagator_g = omega @ propagator @ omega.conj().T
    # Local color transformations commute with spin Gamma matrices and the
    # diagonal site/momentum projector.
    np.testing.assert_allclose(omega @ gamma, gamma @ omega, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        omega @ projector_q, projector_q @ omega, rtol=0, atol=1e-13
    )
    transformed = -0.5 * (
        np.trace(projector_q @ gamma @ derivative_g @ propagator_g)
        + np.trace(derivative_g @ projector_q @ gamma @ propagator_g)
    )
    np.testing.assert_allclose(transformed, target, rtol=1e-13, atol=1e-13)


def test_vector_loop_obeys_q_minus_q_hermiticity_after_reconstruction():
    rng = np.random.default_rng(7305)
    right_diff = rng.normal(size=(16, 2, 3)) + 1j * rng.normal(
        size=(16, 2, 3)
    )
    projectors = _MomentumProjectors(None, None, (0, 1), (1, 0))
    loops = EMTDisconnectedQuark1pt._closed_loop_derivative_from_right_projection(
        right_diff, projectors
    )
    for gamma_position in VECTOR_GAMMA_POSITIONS:
        np.testing.assert_allclose(
            loops[gamma_position, 0],
            loops[gamma_position, 1].conj(),
            rtol=0,
            atol=1e-13,
        )
