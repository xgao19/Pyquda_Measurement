import numpy as np

from pyquda_utils import core, gamma, source
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
    gamma_stack as canonical_gamma_stack,
)
from pyquda_measurement_utils.tools import (
    _asarray_on_queue,
    _get_xp_from_array,
    array_to_numpy,
)


my_gammas = list(GAMMA_LABELS)
pyquda_gammas_order = list(PYQUDA_GAMMA_IDS)
my_pyquda_gammas = [gamma.gamma(idx) for idx in pyquda_gammas_order]
G5 = gamma.gamma(15)


def build_pion_source_propagators(
    dirac,
    latt_info,
    src_pos,
    *,
    gaussian_smearing,
    width,
    pos_boost,
    neg_boost,
):
    """Return positive spectator and negative active point-sink propagators.

    The two lines share one inversion only when their source smearing is
    identical.  Gauge restore/load ownership remains with the caller.
    """
    src_positive = source.propagator(latt_info, "point", src_pos)
    if not gaussian_smearing:
        prop_positive = core.invertPropagator(dirac, src_positive, 1, 0)
        return prop_positive, prop_positive.copy()

    src_positive = boosted_smearing(
        src_positive, w=width, boost=pos_boost
    )
    prop_positive = core.invertPropagator(dirac, src_positive, 1, 0)
    if list(pos_boost) == list(neg_boost):
        return prop_positive, prop_positive.copy()

    src_negative = source.propagator(latt_info, "point", src_pos)
    src_negative = boosted_smearing(
        src_negative, w=width, boost=neg_boost
    )
    prop_negative = core.invertPropagator(dirac, src_negative, 1, 0)
    return prop_positive, prop_negative


def _gamma_matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def matrix_on_backend(value, reference_array):
    """Place a small matrix on the backend and queue of ``reference_array``."""
    xp = _get_xp_from_array(reference_array)
    value = _gamma_matrix(value)
    value_backend = type(value).__module__.split(".")[0]
    if value_backend == xp.__name__:
        if xp.__name__ != "dpnp":
            return value
        reference_queue = getattr(reference_array, "sycl_queue", None)
        if getattr(value, "sycl_queue", None) is reference_queue:
            return value
    if hasattr(value, "get"):
        value = value.get()
    return _asarray_on_queue(value, xp, reference_array)


def matrix_stack_on_backend(values, reference_array):
    """Stack small matrices on the backend and queue of ``reference_array``."""
    xp = _get_xp_from_array(reference_array)
    return xp.stack(
        [matrix_on_backend(value, reference_array) for value in values]
    )


def gamma_stack(reference_array):
    return canonical_gamma_stack(reference_array)


def zeros_on_backend(shape, dtype, xp, reference_array):
    if xp.__name__ == "dpnp" and hasattr(reference_array, "sycl_queue"):
        return xp.zeros(
            shape,
            dtype=dtype,
            sycl_queue=reference_array.sycl_queue,
        )
    return xp.zeros(shape, dtype=dtype)


def gamma_from_label(label):
    if label not in my_gammas:
        raise ValueError(f"Invalid gamma label: {label}. Expected one of {my_gammas}.")
    return my_pyquda_gammas[my_gammas.index(label)]


def source_gamma_provenance(src_gamma):
    """Return the canonical provenance fields for a pion source Gamma."""
    if src_gamma in my_gammas:
        return {
            "source_gamma_mode": "fixed",
            "source_gamma_label": src_gamma,
        }
    if src_gamma == "dagger_of_sink":
        return {
            "source_gamma_mode": "dagger_of_sink",
            "source_gamma_label": "dagger_of_sink",
        }
    raise ValueError(
        f"Invalid src_gamma: {src_gamma}. "
        f"Use a canonical Gamma label from {my_gammas} or 'dagger_of_sink'."
    )


def source_gamma_stack(src_gamma, sink_gamma_ls, reference_array):
    """Return the source matrices paired with the sink Gamma axis.

    A canonical Gamma label produces a constant source matrix.
    ``dagger_of_sink`` is a relational mode whose output Gamma axis is paired
    one-to-one with the sink Gamma axis.
    """
    source_gamma_provenance(src_gamma)
    xp = _get_xp_from_array(reference_array)
    gamma5 = matrix_on_backend(G5, reference_array)

    if src_gamma == "dagger_of_sink":
        return xp.einsum(
            "ab,gbc,cd->gad",
            gamma5,
            xp.swapaxes(sink_gamma_ls.conj(), 1, 2),
            gamma5,
            optimize=True,
        )
    if src_gamma in my_gammas:
        source_gamma = matrix_on_backend(gamma_from_label(src_gamma), reference_array)
        source_gamma_ls = sink_gamma_ls.copy()
        source_gamma_ls[:] = source_gamma
        return source_gamma_ls

    raise AssertionError("unreachable source-Gamma mode")


def meson_backward_line(prop):
    xp = _get_xp_from_array(prop.data)
    gamma5 = matrix_on_backend(G5, prop.data)
    return xp.einsum("ij,wtzyxilab,kl->wtzyxkjba", gamma5, prop.data.conj(), gamma5, optimize=True)


def contract_pion_2pt(latt_info, prop_forward, prop_backward, phases, src_gamma="5"):
    return contract_pion_gamma_scan(
        latt_info,
        prop_forward,
        prop_backward,
        phases,
        [src_gamma],
    )[src_gamma]


def contract_pion_2pt_multi_src_gamma(latt_info, prop_forward, prop_backward, phases, src_gammas):
    return contract_pion_gamma_scan(
        latt_info,
        prop_forward,
        prop_backward,
        phases,
        src_gammas,
    )


def contract_pion_gamma_scan(
    latt_info,
    forward_prop,
    backward_prop,
    phases,
    src_gammas,
):
    """Contract a pion 16-Gamma scan from two propagators.

    This is the propagator-level entry point shared by pion C2, EMFF,
    qTMDWF, and qDA.  The backward propagator is converted to the standard
    gamma5-hermitian backward line exactly once.
    """
    return contract_pion_gamma_scan_from_backward_line(
        latt_info,
        forward_prop,
        meson_backward_line(backward_prop),
        phases,
        src_gammas,
    )


def contract_pion_gamma_scan_from_backward_line(
    latt_info,
    forward_prop,
    backward_line,
    phases,
    src_gammas,
):
    """Contract all sink Gamma channels from a prebuilt backward line.

    The result is a root-only mapping ``src_gamma -> [16, Nq, Nt]``.  At most
    one propagator-sized sink-Gamma temporary is live at a time.
    """
    src_gammas = list(src_gammas)
    if not src_gammas:
        raise ValueError("src_gammas must contain at least one source Gamma")

    xp = _get_xp_from_array(forward_prop.data)
    sink_gamma_ls = gamma_stack(forward_prop.data)
    source_gamma_ls_by_src = {
        src_gamma: source_gamma_stack(src_gamma, sink_gamma_ls, forward_prop.data)
        for src_gamma in src_gammas
    }
    phases = _asarray_on_queue(phases, xp, forward_prop.data)

    corr_local_by_src = {
        src_gamma: zeros_on_backend(
            (len(sink_gamma_ls), phases.shape[0], latt_info.size[3]),
            forward_prop.data.dtype,
            xp,
            forward_prop.data,
        )
        for src_gamma in src_gammas
    }

    for gamma_idx, sink_gamma in enumerate(sink_gamma_ls):
        sink_inserted = xp.einsum("wtzyxjicf,im->wtzyxjmcf", backward_line, sink_gamma, optimize=True)
        for src_gamma in src_gammas:
            corr_site = xp.einsum(
                "wtzyxjiab,wtzyxilba,lj->wtzyx",
                sink_inserted,
                forward_prop.data,
                source_gamma_ls_by_src[src_gamma][gamma_idx],
                optimize=True,
            )
            corr_local_by_src[src_gamma][gamma_idx] = xp.einsum("qwtzyx,wtzyx->qt", phases, corr_site, optimize=True)
            del corr_site
        del sink_inserted

    corr_by_src = {
        src_gamma: core.gatherLattice(array_to_numpy(corr_local), [2, -1, -1, -1])
        for src_gamma, corr_local in corr_local_by_src.items()
    }
    del corr_local_by_src, source_gamma_ls_by_src
    return corr_by_src
