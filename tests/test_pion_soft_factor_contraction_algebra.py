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


def _roll_steps(field, steps, axis):
    out = field
    for _ in range(steps):
        out = np.roll(out, 1, axis=axis)
    return out


def test_incremental_periodic_shift_matches_direct_and_is_faster():
    rng = np.random.default_rng(11)
    field = rng.normal(size=(4, 6, 6, 8)) + 1j * rng.normal(size=(4, 6, 6, 8))
    separations = 16
    axis = -1

    direct = [_roll_steps(field, b, axis) for b in range(separations)]
    incremental = []
    current = field
    for b in range(separations):
        if b != 0:
            current = np.roll(current, 1, axis=axis)
        incremental.append(current.copy())
    for b in range(separations):
        np.testing.assert_allclose(incremental[b], direct[b], atol=0, rtol=0)

    direct_time = _time_median(lambda: [_roll_steps(field, b, axis) for b in range(separations)])
    def _incremental():
        current = field
        for b in range(separations):
            if b != 0:
                current = np.roll(current, 1, axis=axis)
    incremental_time = _time_median(_incremental)
    assert incremental_time < direct_time, (
        f"incremental {incremental_time:.3f}s was not faster than direct {direct_time:.3f}s"
    )


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


def _measurement():
    from pyquda_measurement_utils.pion_soft_factor_vibe_develop import pion_soft_factor

    return pion_soft_factor(
        {
            "quark_mom": [[0, 0, 4]],
            "bT_dir": [0],
            "bT_length": 1,
            "tsep_list": [6, 8],
        }
    )


def test_source_block_cache_reuses_tmp1_across_tsep():
    measurement = _measurement()
    fw, bw = object(), object()
    rng = np.random.default_rng(3)
    shape = (1, 2, 2, 2, 2, 2, 2, 2, 2)
    base = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        time.sleep(0.02)
        return base.copy()

    t0 = time.perf_counter()
    first = measurement._cached_source_block(fw, bw, 0, 0, build)
    first_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    second = measurement._cached_source_block(fw, bw, 0, 0, build)
    second_time = time.perf_counter() - t1

    assert calls["n"] == 1
    assert second is first
    assert second_time < first_time

    gamma2 = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    gamma1 = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    sink = rng.normal(size=first[0].shape) + 1j * rng.normal(size=first[0].shape)
    cached = _soft_factor_reduced_gamma(second[0], sink, gamma2, gamma1)
    fresh = _soft_factor_reduced_gamma(base[0], sink, gamma2, gamma1)
    np.testing.assert_allclose(cached, fresh, atol=1e-10, rtol=1e-10)

    measurement._cached_source_block(fw, bw, 0, 1, build)
    assert calls["n"] == 2
    assert measurement._source_block_hits == 1
    assert measurement._source_block_misses == 2


def test_source_block_cache_recomputes_when_it_does_not_fit(monkeypatch):
    import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft

    monkeypatch.setattr(soft, "_source_block_cache_limit_bytes", lambda: 0)
    measurement = _measurement()
    fw, bw = object(), object()
    base = np.ones((2, 2), dtype=np.complex128)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return base.copy()

    first = measurement._cached_source_block(fw, bw, 0, 0, build)
    second = measurement._cached_source_block(fw, bw, 0, 0, build)
    assert calls["n"] == 2
    assert second is not first
    np.testing.assert_allclose(first, second)


def _direct_tmdwf_plane(backward, forward, src_gamma, gamma5, b_axis, bT_length, bz_length):
    out = np.empty((bT_length + 1, bz_length + 1, backward.shape[0]), dtype=np.complex128)
    for bT in range(bT_length + 1):
        shifted_b = np.roll(backward, bT, axis=b_axis)
        for bz in range(bz_length + 1):
            shifted = np.roll(shifted_b, bz, axis=1)
            shifted_bar = np.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, shifted.conj(), gamma5)
            left = np.einsum("ij,tzyxjlca->tzyxilca", src_gamma, shifted_bar)
            corr_local = np.einsum("tzyxjiab,tzyxilba->tzyx", left, forward)
            out[bT, bz] = np.einsum("tzyx->t", corr_local)
    return out


def test_tmdwf_correlation_matches_shifts_and_is_faster():
    from pyquda_measurement_utils.pion_soft_factor_vibe_develop import tmdwf_separation_plane

    rng = np.random.default_rng(5)
    shape = (2, 8, 2, 8, 4, 4, 3, 3)
    backward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    forward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    src_gamma = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gamma5 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    bT_length, bz_length = 7, 7

    got = np.asarray(
        tmdwf_separation_plane(backward, forward, src_gamma, gamma5, 0, bT_length, bz_length)
    )
    ref = _direct_tmdwf_plane(backward, forward, src_gamma, gamma5, 3, bT_length, bz_length)
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=1e-8)

    corr_time = _time_median(
        lambda: tmdwf_separation_plane(backward, forward, src_gamma, gamma5, 0, bT_length, bz_length)
    )
    direct_time = _time_median(
        lambda: _direct_tmdwf_plane(backward, forward, src_gamma, gamma5, 3, bT_length, bz_length)
    )
    assert corr_time < direct_time, f"correlation {corr_time:.3f}s was not faster than shifts {direct_time:.3f}s"

    same_axis = np.asarray(
        tmdwf_separation_plane(backward, forward, src_gamma, gamma5, 2, 2, 2)
    )
    same_axis_ref = _direct_tmdwf_plane(backward, forward, src_gamma, gamma5, 1, 2, 2)
    np.testing.assert_allclose(same_axis, same_axis_ref, atol=1e-8, rtol=1e-8)


def test_stitch_lexico_rebuilds_a_split_volume():
    from pyquda_measurement_utils.pion_soft_factor_vibe_develop import _stitch_lexico

    rng = np.random.default_rng(9)
    full = rng.normal(size=(2, 4, 2, 4, 3)) + 1j * rng.normal(size=(2, 4, 2, 4, 3))
    blocks = []
    coords = []
    for gx in range(2):
        for gz in range(2):
            blocks.append(full[:, gz * 2 : (gz + 1) * 2, :, gx * 2 : (gx + 1) * 2])
            coords.append((gx, 0, gz, 0))
    stitched = _stitch_lexico(blocks, coords, grid=(2, 1, 2, 1), gather_dims={0, 2})
    np.testing.assert_allclose(stitched, full)
