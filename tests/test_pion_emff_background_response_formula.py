import numpy as np

import pyquda_measurement_utils.pion_current_background_response_vibe_develop as response_module
from pyquda_measurement_utils.pion_current_background_response_vibe_develop import (
    build_local_current_inserted_source,
    current_current_response_toy,
    infer_source_momentum,
    relative_tau_to_absolute,
    response_at_sink_time,
    response_ratio,
    roll_to_source_relative,
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


def test_current_current_response_toy_matches_nested_current_insertions():
    rng = np.random.default_rng(20260521)
    prop_forward = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    gamma_1 = np.array([[0, 1], [2, 0]], dtype=np.complex128)
    gamma_2 = np.array([[1, 1j], [-1j, 3]], dtype=np.complex128)
    phase_1 = np.exp(2j * np.pi * np.arange(4) / 4)
    phase_2 = np.exp(-2j * np.pi * np.arange(4) / 4)

    expected = np.zeros_like(prop_forward)
    for tau in range(4):
        expected[tau] = phase_2[tau] * gamma_2 @ (phase_1[tau] * gamma_1 @ prop_forward[tau])

    response = current_current_response_toy(prop_forward, phase_1, phase_2, gamma_1, gamma_2)
    np.testing.assert_allclose(response, expected)


def test_summed_explicit_and_response_selectors_use_expected_gamma_indices():
    c3 = np.zeros((16, 2, 5), dtype=np.complex128)
    c3[1, 0, :] = np.arange(5)
    assert (
        summed_explicit_emff(
            c3,
            current_gamma="T",
            q_index=0,
            tau_relative_list=[1, 2, 3],
        )
        == 6
    )

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
    assert relative_tau_to_absolute([0, 1, 3], 7, 8) == [7, 0, 2]
    np.testing.assert_array_equal(
        roll_to_source_relative(np.arange(8), 3),
        [3, 4, 5, 6, 7, 0, 1, 2],
    )


def test_tau_window_rejects_nonperiodic_relative_indices():
    with np.testing.assert_raises(ValueError):
        tau_window_list(8, 8, "source_sink")
    with np.testing.assert_raises(ValueError):
        tau_window_list(3, 8, "range:2-8")


def test_inserted_source_converts_relative_tau_only_at_projector(monkeypatch):
    class LattInfo:
        global_size = [1, 1, 1, 4]

    class Propagator:
        def __init__(self, latt_info):
            self.latt_info = latt_info
            self.data = np.zeros((1, 4, 1, 1, 1, 1, 1, 1, 1), dtype=np.complex128)

    prop_forward = Propagator(LattInfo())
    prop_forward.data[:] = 1
    projected_times = []

    def sequential12(prop, tau_absolute):
        projected_times.append(int(tau_absolute))
        projected = Propagator(prop.latt_info)
        projected.data[:, int(tau_absolute)] = prop.data[:, int(tau_absolute)]
        return projected

    monkeypatch.setattr(response_module.core, "LatticePropagator", Propagator)
    monkeypatch.setattr(response_module.source, "sequential12", sequential12)

    result = build_local_current_inserted_source(
        prop_forward,
        np.ones((1, 4, 1, 1, 1), dtype=np.complex128),
        source_time=3,
        current_gamma=np.ones((1, 1), dtype=np.complex128),
        tau_relative_list=[0, 2],
    )

    assert projected_times == [3, 1]
    assert np.count_nonzero(result.data[:, 3]) > 0
    assert np.count_nonzero(result.data[:, 1]) > 0
    assert np.count_nonzero(result.data[:, 0]) == 0
    assert np.count_nonzero(result.data[:, 2]) == 0


def test_response_ratio_uses_c2_denominator():
    assert response_ratio(6 + 3j, 3) == 2 + 1j
    assert np.isnan(response_ratio(1, 0).real)
