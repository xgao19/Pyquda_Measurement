import time

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


def _soft_factor_reduced_gamma(tmp_1, tmp_2, gamma2_ls, gamma1_ls):
    color_spin = np.einsum("tzyxjiba,tzyxklba->tjikl", tmp_1, tmp_2, optimize=True)
    return np.einsum("tjikl,gik,glj->gt", color_spin, gamma2_ls, gamma1_ls, optimize=True)


def _soft_factor_gamma_loop(tmp_1, tmp_2, gamma2_ls, gamma1_ls):
    traces = []
    for igm in range(gamma2_ls.shape[0]):
        corr_local = _soft_factor_einsum(tmp_1, gamma2_ls[igm], tmp_2, gamma1_ls[igm])
        traces.append(np.einsum("tzyx->t", corr_local, optimize=True))
    return np.stack(traces, axis=0)


def test_pion_soft_factor_reduced_gamma_matches_per_gamma_trace():
    rng = np.random.default_rng(42)
    shape = (4, 4, 4, 4, 4, 4, 3, 3)
    tmp_1 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tmp_2 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    n_gamma = 6
    gamma2_ls = rng.normal(size=(n_gamma, 4, 4)) + 1j * rng.normal(size=(n_gamma, 4, 4))
    gamma1_ls = rng.normal(size=(n_gamma, 4, 4)) + 1j * rng.normal(size=(n_gamma, 4, 4))

    reduced = _soft_factor_reduced_gamma(tmp_1, tmp_2, gamma2_ls, gamma1_ls)
    looped = _soft_factor_gamma_loop(tmp_1, tmp_2, gamma2_ls, gamma1_ls)
    np.testing.assert_allclose(reduced, looped, atol=1e-10, rtol=1e-10)

    def _time(fn):
        fn()
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    reduced_time = _time(lambda: _soft_factor_reduced_gamma(tmp_1, tmp_2, gamma2_ls, gamma1_ls))
    loop_time = _time(lambda: _soft_factor_gamma_loop(tmp_1, tmp_2, gamma2_ls, gamma1_ls))
    assert reduced_time < loop_time, f"reduced {reduced_time:.3f}s was not faster than loop {loop_time:.3f}s"


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
