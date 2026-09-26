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


def _soft_factor_block(left, gamma_ls, gamma5, right):
    n_src = gamma_ls.shape[0]
    spatial = left.shape[:4]
    eye3 = np.eye(3, dtype=left.dtype)
    spin = np.matmul(gamma_ls, gamma5)
    spin12 = np.einsum("sil,ba->sialb", spin, eye3).reshape(n_src, 12, 12)
    gamma5_12 = np.einsum("mn,ba->mbna", gamma5, eye3).reshape(12, 12)
    left_m = np.transpose(left, (0, 1, 2, 3, 4, 6, 5, 7)).reshape(-1, 12, 12)
    right_m = np.transpose(right, (0, 1, 2, 3, 4, 6, 5, 7)).reshape(-1, 12, 12)
    right_t = np.swapaxes(np.matmul(gamma5_12, right_m), -1, -2)
    out_m = np.matmul(left_m[None], np.matmul(spin12[:, None], right_t[None]))
    out = out_m.reshape(n_src, *spatial, 4, 3, 4, 3)
    return np.transpose(out, (0, 1, 2, 3, 4, 5, 7, 6, 8))


def _time_median(fn, repeats=3):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def test_pion_soft_factor_block_matmul_matches_einsum_and_is_faster():
    rng = np.random.default_rng(7)
    shape = (8, 8, 8, 8, 4, 4, 3, 3)
    left = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    gamma_ls = rng.normal(size=(1, 4, 4)) + 1j * rng.normal(size=(1, 4, 4))
    gamma5 = np.diag([1, 1, -1, -1]).astype(np.complex128)

    got = _soft_factor_block(left, gamma_ls, gamma5, right)
    ref = np.einsum(
        "tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc",
        left,
        gamma_ls,
        gamma5,
        right,
        gamma5,
        optimize=True,
    )
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)

    matmul_time = _time_median(lambda: _soft_factor_block(left, gamma_ls, gamma5, right))
    einsum_time = _time_median(
        lambda: np.einsum(
            "tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc",
            left,
            gamma_ls,
            gamma5,
            right,
            gamma5,
            optimize=True,
        )
    )
    assert matmul_time < einsum_time, f"matmul {matmul_time:.3f}s was not faster than einsum {einsum_time:.3f}s"


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
