import numpy as np

from pyquda_measurement_utils.pion_utils_vibe_develop import (
    G5,
    gamma_from_label,
    gamma_stack,
    source_gamma_stack,
)


def _matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def test_source_gamma_stack_fixed_g5_and_explicit_label_are_constant():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)

    fixed = source_gamma_stack("fixed_g5", sink_gamma_ls, reference)
    explicit = source_gamma_stack("T5", sink_gamma_ls, reference)

    for gamma_idx in range(len(sink_gamma_ls)):
        np.testing.assert_allclose(fixed[gamma_idx], _matrix(G5))
        np.testing.assert_allclose(explicit[gamma_idx], _matrix(gamma_from_label("T5")))


def test_source_gamma_stack_same_and_dagger_modes():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)
    gamma5 = _matrix(G5)

    same = source_gamma_stack("same_as_sink", sink_gamma_ls, reference)
    dagger = source_gamma_stack("dagger_of_sink", sink_gamma_ls, reference)

    np.testing.assert_allclose(same, sink_gamma_ls)
    for gamma_idx, sink_gamma in enumerate(sink_gamma_ls):
        expected = gamma5 @ sink_gamma.conj().T @ gamma5
        np.testing.assert_allclose(dagger[gamma_idx], expected)


def test_source_gamma_stack_rejects_invalid_label():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)

    try:
        source_gamma_stack("not_a_gamma", sink_gamma_ls, reference)
    except ValueError:
        return
    raise AssertionError("source_gamma_stack should reject invalid source gamma labels")
