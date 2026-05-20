import numpy as np


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
