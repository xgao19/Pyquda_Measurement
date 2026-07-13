"""Canonical 16-element Dirac basis for fermion bilinear measurements.

The raw basis deliberately follows the historical PyQUDA bit-mask ordering used
by the qTMD and two-point workflows.  ``PHYSICAL_FROM_PYQUDA`` records the two
axial sign changes needed to interpret the labels uniformly as
``gamma_mu gamma_5``.  Tensor bit masks are the Euclidean products
``gamma_mu gamma_nu = [gamma_mu, gamma_nu] / 2``; a Hermitian ``i sigma``
convention is obtained by multiplying those channels by ``1j``.
"""

from __future__ import annotations

import numpy as np

from pyquda_utils import gamma

from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


GAMMA_BASIS_SCHEMA = "pyquda_bitmask16_with_physics_transform_v1"
GAMMA_LABELS = (
    "5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I",
    "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT",
)
PYQUDA_GAMMA_IDS = (15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12)
DERIVATIVE_DIRECTIONS = ("X", "Y", "Z", "T")
VECTOR_GAMMA_LABELS = DERIVATIVE_DIRECTIONS
AXIAL_GAMMA_LABELS = ("X5", "Y5", "Z5", "T5")
TENSOR_GAMMA_LABELS = ("SXY", "SXZ", "SXT", "SYZ", "SYT", "SZT")

VECTOR_GAMMA_POSITIONS = tuple(GAMMA_LABELS.index(label) for label in VECTOR_GAMMA_LABELS)
AXIAL_GAMMA_POSITIONS = tuple(GAMMA_LABELS.index(label) for label in AXIAL_GAMMA_LABELS)
TENSOR_GAMMA_POSITIONS = tuple(GAMMA_LABELS.index(label) for label in TENSOR_GAMMA_LABELS)
IDENTITY_GAMMA_POSITION = GAMMA_LABELS.index("I")

# Same label ordering on both axes.  Only Y5 and T5 need signs to mean
# gamma_mu gamma_5 for mu=(X,Y,Z,T) with gamma_5=gamma_1 gamma_2 gamma_3 gamma_4.
PHYSICAL_FROM_PYQUDA = np.eye(16, dtype=np.complex128)
PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("Y5"), GAMMA_LABELS.index("Y5")] = -1
PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("T5"), GAMMA_LABELS.index("T5")] = -1


def _gamma_matrix(gamma_like):
    return gamma_like.matrix if hasattr(gamma_like, "matrix") else gamma_like


def _matrix_to_numpy(matrix):
    """Copy a backend Gamma matrix to host without implicit device conversion."""
    matrix = _gamma_matrix(matrix)
    if hasattr(matrix, "get"):
        matrix = matrix.get()
    elif type(matrix).__module__.split(".")[0] == "dpnp":
        import dpnp

        matrix = dpnp.asnumpy(matrix)
    return np.asarray(matrix, dtype=np.complex128)


def gamma_matrices_numpy():
    """Return the raw PyQUDA matrices as ``[gamma, spin, spin]``."""
    return np.stack([
        _matrix_to_numpy(gamma.gamma(gamma_id))
        for gamma_id in PYQUDA_GAMMA_IDS
    ])


def gamma_stack(reference_array):
    """Return the raw basis on the backend and queue of ``reference_array``."""
    xp = _get_xp_from_array(reference_array)
    matrices = []
    for gamma_id in PYQUDA_GAMMA_IDS:
        matrix = _gamma_matrix(gamma.gamma(gamma_id))
        if hasattr(matrix, "get"):
            matrix = matrix.get()
        matrices.append(_asarray_on_queue(matrix, xp, reference_array))
    return xp.stack(matrices)


def basis_metadata():
    """Small HDF5-safe datasets describing the raw and physical bases."""
    return {
        "gamma_list": np.asarray(GAMMA_LABELS, dtype="S"),
        "gamma_pyquda_ids": np.asarray(PYQUDA_GAMMA_IDS, dtype=np.int32),
        "gamma_matrices": gamma_matrices_numpy(),
        "physical_gamma_list": np.asarray(GAMMA_LABELS, dtype="S"),
        "physical_from_pyquda": PHYSICAL_FROM_PYQUDA.copy(),
        "derivative_directions": np.asarray(DERIVATIVE_DIRECTIONS, dtype="S"),
    }


def basis_attrs():
    return {
        "gamma_basis_schema": GAMMA_BASIS_SCHEMA,
        "gamma_basis_order": ",".join(GAMMA_LABELS),
        "physical_transform_definition": (
            "Gamma_physical[A]=sum_B physical_from_pyquda[A,B]*Gamma_raw[B]"
        ),
        "gamma5_definition": "gamma1*gamma2*gamma3*gamma4",
        "axial_definition": "gamma_mu*gamma5",
        "raw_tensor_definition": "0.5*[gamma_mu,gamma_nu]",
        "hermitian_tensor_from_raw_factor": "1j",
        "derivative_direction_order": ",".join(DERIVATIVE_DIRECTIONS),
    }


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += int(ndim)
    if not 0 <= axis < int(ndim):
        raise np.AxisError(axis, ndim=ndim)
    return axis


def transform_to_physical(values, gamma_axis):
    """Apply ``PHYSICAL_FROM_PYQUDA`` while preserving the gamma-axis position."""
    values = np.asarray(values)
    gamma_axis = _normalize_axis(gamma_axis, values.ndim)
    if values.shape[gamma_axis] != len(GAMMA_LABELS):
        raise ValueError(
            f"gamma axis should have length {len(GAMMA_LABELS)}, got {values.shape[gamma_axis]}"
        )
    moved = np.moveaxis(values, gamma_axis, 0)
    transformed = np.tensordot(PHYSICAL_FROM_PYQUDA, moved, axes=(1, 0))
    return np.moveaxis(transformed, 0, gamma_axis)


def _symmetrized_direction_tensor(values, positions, factors, gamma_axis, derivative_axis):
    values = np.asarray(values)
    gamma_axis = _normalize_axis(gamma_axis, values.ndim)
    derivative_axis = _normalize_axis(derivative_axis, values.ndim)
    if gamma_axis == derivative_axis:
        raise ValueError("gamma_axis and derivative_axis should be different")
    if values.shape[gamma_axis] != len(GAMMA_LABELS):
        raise ValueError("unexpected gamma-axis length")
    if values.shape[derivative_axis] != 4:
        raise ValueError("derivative axis should have length 4")
    selected = np.take(values, positions, axis=gamma_axis)
    factor_shape = [1] * selected.ndim
    factor_shape[gamma_axis] = 4
    selected = selected * np.asarray(factors, dtype=np.complex128).reshape(factor_shape)
    return 0.5 * (selected + np.swapaxes(selected, gamma_axis, derivative_axis))


def symmetric_vector_emt(values, gamma_axis, derivative_axis):
    """Construct the old symmetric EMT tensor from raw derivative bilinears."""
    return _symmetrized_direction_tensor(
        values, VECTOR_GAMMA_POSITIONS, (1, 1, 1, 1), gamma_axis, derivative_axis
    )


def symmetric_axial_twist2(values, gamma_axis, derivative_axis):
    """Construct gamma_{mu} gamma_5 symmetric one-derivative bilinears."""
    return _symmetrized_direction_tensor(
        values, AXIAL_GAMMA_POSITIONS, (1, -1, 1, -1), gamma_axis, derivative_axis
    )


def local_tensor_channels(values, gamma_axis, hermitian=False):
    """Extract (XY,XZ,XT,YZ,YT,ZT) local tensor-current channels."""
    values = np.asarray(values)
    gamma_axis = _normalize_axis(gamma_axis, values.ndim)
    selected = np.take(values, TENSOR_GAMMA_POSITIONS, axis=gamma_axis)
    return (1j * selected) if hermitian else selected


__all__ = [
    "AXIAL_GAMMA_LABELS",
    "DERIVATIVE_DIRECTIONS",
    "GAMMA_BASIS_SCHEMA",
    "GAMMA_LABELS",
    "IDENTITY_GAMMA_POSITION",
    "PHYSICAL_FROM_PYQUDA",
    "PYQUDA_GAMMA_IDS",
    "TENSOR_GAMMA_LABELS",
    "VECTOR_GAMMA_LABELS",
    "VECTOR_GAMMA_POSITIONS",
    "basis_attrs",
    "basis_metadata",
    "gamma_matrices_numpy",
    "gamma_stack",
    "local_tensor_channels",
    "symmetric_axial_twist2",
    "symmetric_vector_emt",
    "transform_to_physical",
]
