import numpy as np

from pyquda_measurement_utils.fermion_bilinear_basis import (
    IDENTITY_GAMMA_POSITION,
    VECTOR_GAMMA_POSITIONS,
    gamma_matrices_numpy,
    symmetric_vector_emt,
)


def _c3_chi_formula(dst2, prop_fw, src_gamma):
    return np.einsum("tabij,tbcji,ca->t", dst2, prop_fw, src_gamma, optimize=True)


def _c3_tmunu_formula(dst2, prop_fw, left_d_dst2, d_fw, gamma_nu, src_gamma):
    gamma_d_fw = np.einsum("ab,tbdij->tadij", gamma_nu, d_fw, optimize=True)
    first = 0.5 * np.einsum("tabij,tbcji,ca->t", dst2, gamma_d_fw, src_gamma, optimize=True)

    gamma_fw = np.einsum("ab,tbdij->tadij", gamma_nu, prop_fw, optimize=True)
    second = -0.5 * np.einsum("tabij,tbcji,ca->t", left_d_dst2, gamma_fw, src_gamma, optimize=True)
    return first + second


def _manual_c3_chi(dst2, prop_fw, src_gamma):
    out = np.zeros(dst2.shape[0], dtype=np.result_type(dst2, prop_fw, src_gamma))
    for t in range(dst2.shape[0]):
        total = 0.0j
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for i in range(2):
                        for j in range(2):
                            total += dst2[t, a, b, i, j] * prop_fw[t, b, c, j, i] * src_gamma[c, a]
        out[t] = total
    return out


def test_pion_emt_c3_chi_index_order_matches_manual_trace():
    rng = np.random.default_rng(123)
    dst2 = rng.normal(size=(3, 2, 2, 2, 2)) + 1j * rng.normal(size=(3, 2, 2, 2, 2))
    prop_fw = rng.normal(size=(3, 2, 2, 2, 2)) + 1j * rng.normal(size=(3, 2, 2, 2, 2))
    src_gamma = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))

    np.testing.assert_allclose(
        _c3_chi_formula(dst2, prop_fw, src_gamma),
        _manual_c3_chi(dst2, prop_fw, src_gamma),
        atol=1e-13,
        rtol=1e-13,
    )


def test_pion_emt_tmunu_formula_has_forward_and_backward_derivative_terms():
    rng = np.random.default_rng(456)
    shape = (2, 2, 2, 2, 2)
    dst2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    prop_fw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    left_d_dst2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    d_fw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    gamma_nu = np.array([[0, 1], [2, 0]], dtype=np.complex128)
    src_gamma = np.array([[1, 1j], [-2j, 3]], dtype=np.complex128)

    full = _c3_tmunu_formula(dst2, prop_fw, left_d_dst2, d_fw, gamma_nu, src_gamma)
    no_backward = _c3_tmunu_formula(dst2, prop_fw, np.zeros_like(left_d_dst2), d_fw, gamma_nu, src_gamma)
    no_forward = _c3_tmunu_formula(np.zeros_like(dst2), prop_fw, left_d_dst2, d_fw, gamma_nu, src_gamma)

    np.testing.assert_allclose(full, no_backward + no_forward)
    assert not np.allclose(full, no_backward)
    assert not np.allclose(full, no_forward)


def test_emt_tmunu_symmetrization_keeps_upper_and_lower_triangle_equal():
    rng = np.random.default_rng(789)
    c3 = rng.normal(size=(2, 4, 4, 5)) + 1j * rng.normal(size=(2, 4, 4, 5))

    for mu in range(4):
        for nu in range(mu + 1, 4):
            c3[:, mu, nu] = 0.5 * (c3[:, mu, nu] + c3[:, nu, mu])
            c3[:, nu, mu] = c3[:, mu, nu]

    for mu in range(4):
        for nu in range(4):
            np.testing.assert_allclose(c3[:, mu, nu], c3[:, nu, mu])


def test_pion_batch_primitive_vector_channels_reproduce_old_formula():
    rng = np.random.default_rng(991)
    shape = (3, 4, 4, 2, 2)
    dst2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    prop_fw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    src_gamma = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gammas = gamma_matrices_numpy()
    derivative = np.zeros((16, 4, shape[0]), dtype=np.complex128)
    for mu in range(4):
        d_fw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        left_d = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        derivative[:, mu] = (
            0.5 * np.einsum(
                "tabij,gbn,tncji,ca->gt",
                dst2, gammas, d_fw, src_gamma, optimize=True,
            )
            - 0.5 * np.einsum(
                "tabij,gbn,tncji,ca->gt",
                left_d, gammas, prop_fw, src_gamma, optimize=True,
            )
        )
        for nu, gamma_position in enumerate(VECTOR_GAMMA_POSITIONS):
            expected = _c3_tmunu_formula(
                dst2, prop_fw, left_d, d_fw, gammas[gamma_position], src_gamma
            )
            np.testing.assert_allclose(
                derivative[gamma_position, mu], expected, rtol=1e-13, atol=1e-13
            )

    tensor = symmetric_vector_emt(derivative, gamma_axis=0, derivative_axis=1)
    np.testing.assert_allclose(tensor, tensor.swapaxes(0, 1), rtol=0, atol=0)


def test_proton_batch_primitive_identity_and_vector_channels_match_old_contractions():
    rng = np.random.default_rng(992)
    raw_seq = rng.normal(size=(3, 4, 4, 2, 2)) + 1j * rng.normal(
        size=(3, 4, 4, 2, 2)
    )
    prop = rng.normal(size=(3, 4, 4, 2, 2)) + 1j * rng.normal(
        size=(3, 4, 4, 2, 2)
    )
    gammas = gamma_matrices_numpy()
    gamma5 = gammas[0]
    g5_gammas = np.einsum("ai,gib->gab", gamma5, gammas)
    spin_trace = np.einsum("tajfc,tbjfc->tab", raw_seq.conj(), prop)
    batch = np.einsum("tab,gab->gt", spin_trace, g5_gammas)

    old_identity = np.einsum("tajfc,tijfc,ai->t", raw_seq.conj(), prop, gamma5)
    np.testing.assert_allclose(
        batch[IDENTITY_GAMMA_POSITION], old_identity, rtol=1e-13, atol=1e-13
    )
    for gamma_position in VECTOR_GAMMA_POSITIONS:
        old = np.einsum(
            "tajfc,tbjfc,ab->t",
            raw_seq.conj(), prop, gamma5 @ gammas[gamma_position],
        )
        np.testing.assert_allclose(batch[gamma_position], old, rtol=1e-13, atol=1e-13)


def test_disconnected_batch_primitive_vector_channels_match_old_gamma_loop():
    rng = np.random.default_rng(993)
    xi = rng.normal(size=(5, 4, 3)) + 1j * rng.normal(size=(5, 4, 3))
    shifted_eta = rng.normal(size=(5, 4, 3)) + 1j * rng.normal(size=(5, 4, 3))
    gammas = gamma_matrices_numpy()
    batch = np.einsum("tia,gij,tja->gt", xi.conj(), gammas, shifted_eta)
    for gamma_position in VECTOR_GAMMA_POSITIONS:
        inserted = np.einsum("ij,tja->tia", gammas[gamma_position], shifted_eta)
        old = np.einsum("tia,tia->t", xi.conj(), inserted)
        np.testing.assert_allclose(batch[gamma_position], old, rtol=1e-13, atol=1e-13)
