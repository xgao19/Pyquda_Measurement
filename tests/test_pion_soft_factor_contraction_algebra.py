import numpy as np


def _soft_factor_einsum(tmp_1, gamma2, tmp_2, gamma1):
    return np.einsum(
        "tzyxjiba,ik,tzyxklba,lj->tzyx",
        tmp_1,
        gamma2,
        tmp_2,
        gamma1,
        optimize=True,
    )


def _soft_factor_manual_trace(tmp_1, gamma2, tmp_2, gamma1):
    out = np.zeros(tmp_1.shape[:4], dtype=np.result_type(tmp_1, tmp_2, gamma1, gamma2))
    for index in np.ndindex(tmp_1.shape[:4]):
        total = 0.0j
        for b in range(tmp_1.shape[-2]):
            for a in range(tmp_1.shape[-1]):
                total += np.trace(tmp_1[index + (slice(None), slice(None), b, a)] @ gamma2 @ tmp_2[index + (slice(None), slice(None), b, a)] @ gamma1)
        out[index] = total
    return out


def test_pion_soft_factor_operator_order_matches_trace_formula():
    rng = np.random.default_rng(1234)
    shape = (2, 1, 1, 1, 2, 2, 2, 2)
    tmp_1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tmp_2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    gamma1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    gamma2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))

    actual = _soft_factor_einsum(tmp_1, gamma2, tmp_2, gamma1)
    expected = _soft_factor_manual_trace(tmp_1, gamma2, tmp_2, gamma1)

    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


def test_pion_soft_factor_gamma_order_is_not_accidentally_commuted():
    rng = np.random.default_rng(5678)
    shape = (1, 1, 1, 1, 2, 2, 1, 1)
    tmp_1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tmp_2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    gamma1 = np.array([[0, 1], [2, 0]], dtype=np.complex128)
    gamma2 = np.array([[1, 1j], [-2j, 3]], dtype=np.complex128)

    correct = _soft_factor_einsum(tmp_1, gamma2, tmp_2, gamma1)
    commuted = _soft_factor_einsum(tmp_1, gamma1, tmp_2, gamma2)

    assert not np.allclose(correct, commuted)
