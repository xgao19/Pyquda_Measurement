import numpy as np
import pytest

from pyquda_measurement_utils.pion_utils_vibe_develop import (
    G5,
    contract_pion_2pt_multi_src_gamma,
    gamma_from_label,
    gamma_stack,
    source_gamma_provenance,
    source_gamma_stack,
)
import pyquda_measurement_utils.pion_utils_vibe_develop as pion_utils


def _matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def test_source_gamma_stack_all_explicit_labels_are_constant():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)

    for label in pion_utils.my_gammas:
        explicit = source_gamma_stack(label, sink_gamma_ls, reference)
        expected = _matrix(gamma_from_label(label))
        for gamma_idx in range(len(sink_gamma_ls)):
            np.testing.assert_allclose(explicit[gamma_idx], expected)


def test_source_gamma_provenance_distinguishes_fixed_and_paired_modes():
    assert source_gamma_provenance("5") == {
        "source_gamma_mode": "fixed",
        "source_gamma_label": "5",
    }
    assert source_gamma_provenance("dagger_of_sink") == {
        "source_gamma_mode": "dagger_of_sink",
        "source_gamma_label": "dagger_of_sink",
    }


@pytest.mark.parametrize("removed_mode", ["fixed_g5", "same_as_sink"])
def test_source_gamma_stack_rejects_removed_modes(removed_mode):
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)

    with pytest.raises(ValueError, match="canonical Gamma label"):
        source_gamma_stack(removed_mode, sink_gamma_ls, reference)
    with pytest.raises(ValueError, match="canonical Gamma label"):
        source_gamma_provenance(removed_mode)


def test_source_gamma_stack_dagger_of_sink_for_all_channels():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)
    gamma5 = _matrix(G5)

    dagger = source_gamma_stack("dagger_of_sink", sink_gamma_ls, reference)
    for gamma_idx, sink_gamma in enumerate(sink_gamma_ls):
        expected = gamma5 @ sink_gamma.conj().T @ gamma5
        np.testing.assert_allclose(
            dagger[gamma_idx], expected, rtol=1e-14, atol=1e-14
        )


def test_source_gamma_stack_dagger_of_sink_cupy_backend_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        reference = cupy.zeros((1,), dtype=cupy.complex128)
    except Exception as exc:
        pytest.skip(f"CuPy device is unavailable: {exc}")

    sink_gamma_ls = gamma_stack(reference)
    dagger = source_gamma_stack("dagger_of_sink", sink_gamma_ls, reference)
    gamma5 = cupy.asarray(_matrix(G5))
    expected = cupy.einsum(
        "ab,gbc,cd->gad",
        gamma5,
        cupy.swapaxes(sink_gamma_ls.conj(), 1, 2),
        gamma5,
        optimize=True,
    )
    cupy.testing.assert_allclose(dagger, expected, rtol=1e-14, atol=1e-14)


def test_source_gamma_stack_dagger_of_sink_dpnp_backend_when_available():
    dpnp = pytest.importorskip("dpnp")
    try:
        reference = dpnp.zeros((1,), dtype=dpnp.complex128)
    except Exception as exc:
        pytest.skip(f"dpnp device is unavailable: {exc}")

    sink_gamma_ls = gamma_stack(reference)
    dagger = source_gamma_stack("dagger_of_sink", sink_gamma_ls, reference)
    gamma5 = dpnp.asarray(_matrix(G5), sycl_queue=reference.sycl_queue)
    expected = dpnp.einsum(
        "ab,gbc,cd->gad",
        gamma5,
        dpnp.swapaxes(sink_gamma_ls.conj(), 1, 2),
        gamma5,
        optimize=True,
    )
    np.testing.assert_allclose(
        dpnp.asnumpy(dagger), dpnp.asnumpy(expected), rtol=1e-14, atol=1e-14
    )


def test_source_gamma_stack_rejects_invalid_label():
    reference = np.zeros((1,), dtype=np.complex128)
    sink_gamma_ls = gamma_stack(reference)

    with pytest.raises(ValueError, match="canonical Gamma label"):
        source_gamma_stack("not_a_gamma", sink_gamma_ls, reference)


def test_multi_source_c2_uses_local_time_extent_and_matches_reference(monkeypatch):
    rng = np.random.default_rng(23)
    lattice_shape = (2, 2, 1, 1, 1)
    data_shape = lattice_shape + (4, 4, 1, 1)
    forward = rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape)
    backward = rng.normal(size=data_shape) + 1j * rng.normal(size=data_shape)
    phases = rng.normal(size=(2,) + lattice_shape) + 1j * rng.normal(
        size=(2,) + lattice_shape
    )

    class FakeLatticeInfo:
        size = [1, 1, 1, lattice_shape[1]]
        global_size = [1, 1, 1, 8]

    class FakePropagator:
        def __init__(self, data):
            self.data = data

    monkeypatch.setattr(pion_utils.core, "gatherLattice", lambda values, axes: values)
    result = contract_pion_2pt_multi_src_gamma(
        FakeLatticeInfo(),
        FakePropagator(forward),
        FakePropagator(backward),
        phases,
        ["5", "X"],
    )

    sink_gammas = gamma_stack(forward)
    backward_line = pion_utils.meson_backward_line(FakePropagator(backward))
    for source_label in ("5", "X"):
        source_gammas = source_gamma_stack(source_label, sink_gammas, forward)
        sink_inserted = np.einsum(
            "wtzyxjicf,gim->gwtzyxjmcf", backward_line, sink_gammas, optimize=True
        )
        corr_site = np.einsum(
            "gwtzyxjiab,wtzyxilba,glj->gwtzyx",
            sink_inserted,
            forward,
            source_gammas,
            optimize=True,
        )
        reference = np.einsum("qwtzyx,gwtzyx->gqt", phases, corr_site, optimize=True)
        assert result[source_label].shape[-1] == lattice_shape[1]
        np.testing.assert_allclose(
            result[source_label], reference, rtol=1e-13, atol=1e-13
        )
