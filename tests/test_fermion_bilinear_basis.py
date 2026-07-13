import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_utils import gamma
import pyquda_measurement_utils.fermion_bilinear_basis as basis_module

from pyquda_measurement_utils.fermion_bilinear_basis import (
    AXIAL_GAMMA_POSITIONS,
    GAMMA_LABELS,
    PHYSICAL_FROM_PYQUDA,
    PYQUDA_GAMMA_IDS,
    TENSOR_GAMMA_POSITIONS,
    VECTOR_GAMMA_POSITIONS,
    gamma_matrices_numpy,
    local_tensor_channels,
    symmetric_axial_twist2,
    symmetric_vector_emt,
    transform_to_physical,
)


def test_raw_gamma_basis_matches_pyquda_and_has_full_rank():
    matrices = gamma_matrices_numpy()
    expected = np.stack([np.asarray(gamma.gamma(idx)) for idx in PYQUDA_GAMMA_IDS])
    np.testing.assert_array_equal(matrices, expected)
    assert np.linalg.matrix_rank(matrices.reshape(16, 16)) == 16


def test_gamma_metadata_explicitly_copies_device_arrays_to_host(monkeypatch):
    expected = {
        gamma_id: np.full((4, 4), gamma_id, dtype=np.complex128)
        for gamma_id in PYQUDA_GAMMA_IDS
    }

    class FakeDeviceArray:
        def __init__(self, values):
            self.values = values

        def get(self):
            return self.values

        def __array__(self, *args, **kwargs):
            raise TypeError("implicit device-to-host conversion is forbidden")

    monkeypatch.setattr(
        basis_module.gamma,
        "gamma",
        lambda gamma_id: FakeDeviceArray(expected[gamma_id]),
    )
    matrices = basis_module.gamma_matrices_numpy()
    np.testing.assert_array_equal(
        matrices,
        np.stack([expected[gamma_id] for gamma_id in PYQUDA_GAMMA_IDS]),
    )


def test_physical_transform_produces_uniform_gamma_mu_gamma5():
    raw = gamma_matrices_numpy()
    physical = np.einsum("ab,bij->aij", PHYSICAL_FROM_PYQUDA, raw)
    g5 = np.asarray(gamma.gamma(15))
    directions = [np.asarray(gamma.gamma(idx)) for idx in (1, 2, 4, 8)]
    for position, gamma_mu in zip(AXIAL_GAMMA_POSITIONS, directions):
        np.testing.assert_array_equal(physical[position], gamma_mu @ g5)

    transformed = transform_to_physical(raw[:, None], gamma_axis=0)
    np.testing.assert_array_equal(transformed[:, 0], physical)


def test_tensor_channels_are_half_commutators_and_optional_hermitian_i_times():
    raw = gamma_matrices_numpy()
    directions = [np.asarray(gamma.gamma(idx)) for idx in (1, 2, 4, 8)]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for position, (mu, nu) in zip(TENSOR_GAMMA_POSITIONS, pairs):
        expected = 0.5 * (directions[mu] @ directions[nu] - directions[nu] @ directions[mu])
        np.testing.assert_array_equal(raw[position], expected)

    values = np.arange(2 * 16 * 3).reshape(2, 16, 3)
    plain = local_tensor_channels(values, gamma_axis=1)
    hermitian = local_tensor_channels(values, gamma_axis=1, hermitian=True)
    np.testing.assert_array_equal(hermitian, 1j * plain)


def test_vector_and_axial_symmetric_extraction_preserve_axes():
    rng = np.random.default_rng(1234)
    derivative = rng.normal(size=(2, 16, 4, 3, 2)) + 1j * rng.normal(
        size=(2, 16, 4, 3, 2)
    )

    vector = symmetric_vector_emt(derivative, gamma_axis=1, derivative_axis=2)
    selected = derivative[:, VECTOR_GAMMA_POSITIONS]
    expected_vector = 0.5 * (selected + selected.swapaxes(1, 2))
    np.testing.assert_allclose(vector, expected_vector, rtol=0, atol=0)

    axial = symmetric_axial_twist2(derivative, gamma_axis=1, derivative_axis=2)
    selected_axial = derivative[:, AXIAL_GAMMA_POSITIONS].copy()
    selected_axial[:, 1] *= -1
    selected_axial[:, 3] *= -1
    expected_axial = 0.5 * (selected_axial + selected_axial.swapaxes(1, 2))
    np.testing.assert_allclose(axial, expected_axial, rtol=0, atol=0)


def test_basis_label_positions_are_stable():
    assert tuple(GAMMA_LABELS[idx] for idx in VECTOR_GAMMA_POSITIONS) == ("X", "Y", "Z", "T")
    assert tuple(GAMMA_LABELS[idx] for idx in AXIAL_GAMMA_POSITIONS) == ("X5", "Y5", "Z5", "T5")
