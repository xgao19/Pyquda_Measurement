"""Shared production utilities for proton measurements."""

from pyquda_utils import core, gamma

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.fermion_bilinear_basis import gamma_stack
from pyquda_measurement_utils.tools import (
    _asarray_on_queue,
    _get_xp_from_array,
    mpi_print,
)


_PROTON_INTERPOLATORS = {
    "5": (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15),
    "T5": (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7),
    "Z5": (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11),
}


def proton_interpolator_matrix(interpolator):
    """Return the host/backend-independent ``C Gamma`` proton interpolator."""
    try:
        return _PROTON_INTERPOLATORS[interpolator]
    except KeyError as error:
        raise ValueError(f"Invalid interpolator: {interpolator}") from error


def contract_proton_c2(
    latt_info,
    prop_f,
    phases,
    *,
    interpolator="5",
    sink_smearing=True,
    smearing_width=None,
    smearing_boost=None,
    gamma_matrices=None,
    interpolator_matrix=None,
):
    """Return the gathered 16-Gamma proton C2 array without writing files.

    ``sink_smearing=False`` leaves ``prop_f`` untouched.  Callers that already
    own an EMT Gamma cache may pass backend-resident ``gamma_matrices`` and
    ``interpolator_matrix``; qTMD callers can omit both.
    """
    if sink_smearing:
        if smearing_width is None or smearing_boost is None:
            raise ValueError("sink smearing requires width and boost")
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(
            prop_f, w=smearing_width, boost=smearing_boost
        )
        mpi_print(latt_info, "Sink smearing completed")

    xp = _get_xp_from_array(prop_f.data)
    if gamma_matrices is None:
        gamma_matrices = gamma_stack(prop_f.data)
    else:
        gamma_matrices = _asarray_on_queue(gamma_matrices, xp, prop_f.data)
    if interpolator_matrix is None:
        interpolator_matrix = proton_interpolator_matrix(interpolator)
    interpolator_matrix = _asarray_on_queue(
        interpolator_matrix, xp, prop_f.data
    )

    # Preserve the established local-time broadcast used by the qTMD kernel.
    p_2pt_gamma = _asarray_on_queue(
        xp.zeros((16, latt_info.Lt, 4, 4), dtype=prop_f.data.dtype),
        xp,
        prop_f.data,
    )
    for gamma_idx in range(16):
        p_2pt_gamma[gamma_idx] = gamma_matrices[gamma_idx]

    epsilon = _asarray_on_queue(
        xp.zeros((3, 3, 3), dtype=prop_f.data.real.dtype), xp, prop_f.data
    )
    for a in range(3):
        b = (a + 1) % 3
        c = (a + 2) % 3
        epsilon[a, b, c] = 1
        epsilon[a, c, b] = -1
    phases = _asarray_on_queue(phases, xp, prop_f.data)

    # First Wick-contraction term, split to bound peak device memory.
    t1_s1 = xp.einsum(
        "abc, wtzyxikad -> wtzyxikbcd", epsilon, prop_f.data, optimize=True
    )
    t1_s2 = xp.einsum(
        "ij, wtzyxjlbe -> wtzyxilbe",
        interpolator_matrix,
        prop_f.data,
        optimize=True,
    )
    term1_sink = xp.einsum(
        "wtzyxikbcd, wtzyxilbe -> wtzyxklcde",
        t1_s1,
        t1_s2,
        optimize=True,
    )
    del t1_s1, t1_s2
    term1_p3 = xp.einsum(
        "gtmn, wtzyxmncf -> gwtzyxcf",
        p_2pt_gamma,
        prop_f.data,
        optimize=True,
    )
    t1_f1 = xp.einsum(
        "def, wtzyxklcde -> wtzyxklcf", epsilon, term1_sink, optimize=True
    )
    del term1_sink
    t1_f2 = xp.einsum(
        "kl, wtzyxklcf -> wtzyxcf",
        interpolator_matrix,
        t1_f1,
        optimize=True,
    )
    del t1_f1
    t1_f3 = xp.einsum(
        "wtzyxcf, gwtzyxcf -> gwtzyx", t1_f2, term1_p3, optimize=True
    )
    del t1_f2, term1_p3
    term1 = xp.einsum(
        "pwtzyx, gwtzyx -> gpt", phases, t1_f3, optimize=True
    )
    del t1_f3

    # Exchange term for the two identical up-quark lines.
    t2_s1 = xp.einsum(
        "abc, wtzyxikad -> wtzyxikbcd", epsilon, prop_f.data, optimize=True
    )
    t2_s2 = xp.einsum(
        "ij, wtzyxjnbe -> wtzyxinbe",
        interpolator_matrix,
        prop_f.data,
        optimize=True,
    )
    term2_sink = xp.einsum(
        "wtzyxikbcd, wtzyxinbe -> wtzyxkncde",
        t2_s1,
        t2_s2,
        optimize=True,
    )
    del t2_s1, t2_s2
    term2_p3 = xp.einsum(
        "gtmn, wtzyxmlcf -> gwtzyxnlcf",
        p_2pt_gamma,
        prop_f.data,
        optimize=True,
    )
    t2_f1 = xp.einsum(
        "def, wtzyxkncde -> wtzyxkncf", epsilon, term2_sink, optimize=True
    )
    del term2_sink
    t2_f2 = xp.einsum(
        "wtzyxkncf, gwtzyxnlcf -> gwtzyxkl",
        t2_f1,
        term2_p3,
        optimize=True,
    )
    del t2_f1, term2_p3
    t2_f3 = xp.einsum(
        "kl, gwtzyxkl -> gwtzyx",
        interpolator_matrix,
        t2_f2,
        optimize=True,
    )
    del t2_f2
    term2 = xp.einsum(
        "pwtzyx, gwtzyx -> gpt", phases, t2_f3, optimize=True
    )
    del t2_f3

    corr = -term1 - term2
    corr_collect = core.gatherLattice(array_to_numpy(corr), [2, -1, -1, -1])
    del corr
    return corr_collect


__all__ = ["contract_proton_c2", "proton_interpolator_matrix"]
