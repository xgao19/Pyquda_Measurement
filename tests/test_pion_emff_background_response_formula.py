import numpy as np

from pyquda_measurement_utils.pion_EMFF_background_response_vibe_develop import (
    infer_source_momentum,
    response_at_sink_time,
    response_ratio,
    summed_explicit_emff,
    tau_window_list,
)


def test_response_propagator_formula_matches_explicit_summed_insertion_toy():
    rng = np.random.default_rng(20260520)
    sink_line = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    prop_forward = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    gamma_current = np.array([[0, 1], [2, 0]], dtype=np.complex128)
    gamma_src = np.array([[1, 1j], [-1j, 2]], dtype=np.complex128)
    phase_q = np.exp(2j * np.pi * np.arange(3) / 3)

    explicit = 0.0j
    for tau in range(3):
        current_inserted = gamma_current @ prop_forward[tau]
        explicit += phase_q[tau] * np.trace(sink_line[tau] @ current_inserted @ gamma_src)

    response_prop = np.einsum("t,ab,tbc->tac", phase_q, gamma_current, prop_forward, optimize=True)
    response = np.einsum("tab,tbc,ca->t", sink_line, response_prop, gamma_src, optimize=True).sum()

    np.testing.assert_allclose(response, explicit)


def test_summed_explicit_and_response_selectors_use_expected_gamma_indices():
    c3 = np.zeros((16, 2, 5), dtype=np.complex128)
    c3[1, 0, :] = np.arange(5)
    assert summed_explicit_emff(c3, current_gamma="T", q_index=0, tau_list=[1, 2, 3]) == 6

    corr = np.zeros((16, 3, 8), dtype=np.complex128)
    corr[0, 2, 4] = 7
    assert response_at_sink_time(corr, sink_gamma="5", p_index=2, tsep=4) == 7


def test_background_response_kinematics_and_tau_windows():
    assert infer_source_momentum([0, 0, 1], [0, 0, 2]) == [0, 0, -1]
    assert infer_source_momentum([0, 0, 1], [0, 0, -2]) == [0, 0, 3]
    assert tau_window_list(6, 32, "all") is None
    assert tau_window_list(6, 32, "source_sink") == [0, 1, 2, 3, 4, 5, 6]
    assert tau_window_list(6, 32, "open") == [1, 2, 3, 4, 5]
    assert tau_window_list(6, 32, "restricted", tau_min=2) == [2, 3, 4]
    assert tau_window_list(6, 32, "range:2-5") == [2, 3, 4, 5]


def test_response_ratio_uses_c2_denominator():
    assert response_ratio(6 + 3j, 3) == 2 + 1j
    assert np.isnan(response_ratio(1, 0).real)
