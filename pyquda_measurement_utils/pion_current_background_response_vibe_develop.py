"""Pion local-current background-response helpers.

This module implements first- and second-order response-propagator diagnostics
for local pion current insertions.  It does not modify the QUDA Dirac operator.
Instead it uses the identities

    S O S = D^{-1} O S
    S O_2 S O_1 S = D^{-1} O_2 D^{-1} O_1 S

for a local vector-current insertion

    O_q(x) = Gamma_current * Phi_q(x - x0),

where ``Phi_q`` is the same ``MomentumPhase`` convention used by the ordinary
pion EMFF three-point code.  The second identity is a minimal
current-current-response diagnostic; it is not a full finite-lambda background
field calculation.  For a true finite-difference derivative of
``(D + lambda O)^{-1}`` there is an additional overall minus sign:

    d (D + lambda O)^{-1} / d lambda |_{0} = - S O S.

The helpers here keep the explicit ``response_sign`` parameter so diagnostics
can compare directly against the summed explicit three-point function before
choosing the finite-difference sign convention.
"""

from pyquda_utils import core, gamma, source

from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt_multi_src_gamma,
    gamma_from_label,
    matrix_on_backend,
)
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


def relative_tau_to_absolute(tau_relative_list, source_time, Nt):
    """Map source-relative insertion times to absolute lattice time."""
    if tau_relative_list is None:
        return None
    Nt = int(Nt)
    if Nt <= 0:
        raise ValueError("Nt must be positive")
    return [
        int((int(source_time) + int(tau_relative)) % Nt)
        for tau_relative in tau_relative_list
    ]

def build_local_current_inserted_source(
    prop_forward,
    phase_q,
    *,
    source_time,
    current_gamma="T",
    tau_relative_list=None,
):
    """Build ``Gamma_current * phase_q * S`` as a propagator source.

    Parameters
    ----------
    prop_forward
        Ordinary source-to-current propagator ``S(x, x0)``.
    phase_q
        One current momentum phase with shape matching the local lattice site
        layout ``wtzyx``.
    current_gamma
        Gamma label or explicit gamma matrix.  The EMFF temporal vector current
        is label ``"T"``.
    source_time
        Absolute time coordinate of the pion source.
    tau_relative_list
        Optional insertion times relative to ``source_time``.  They are mapped
        to absolute lattice time only when the projector is constructed.
    """
    xp = _get_xp_from_array(prop_forward.data)
    phase_q = _asarray_on_queue(phase_q, xp, prop_forward.data)
    gamma_current = gamma_from_label(current_gamma) if isinstance(current_gamma, str) else current_gamma
    gamma_current = matrix_on_backend(gamma_current, prop_forward.data)

    src = core.LatticePropagator(prop_forward.latt_info)
    src.data = xp.einsum(
        "ki,wtzyx,wtzyxilba->wtzyxklba",
        gamma_current,
        phase_q,
        prop_forward.data,
        optimize=True,
    )
    tau_absolute_list = relative_tau_to_absolute(
        tau_relative_list,
        source_time,
        prop_forward.latt_info.global_size[3],
    )
    if tau_absolute_list is not None:
        src_window = core.LatticePropagator(prop_forward.latt_info)
        src_window.data[:] = 0
        for tau_i in tau_absolute_list:
            src_window.data[:] += source.sequential12(src, int(tau_i)).data
        src = src_window
    return src


def invert_local_current_response_propagator(
    dirac,
    prop_forward,
    phase_q,
    *,
    source_time,
    current_gamma="T",
    tau_relative_list=None,
    response_sign=1,
    mrhs=1,
    restart=0,
):
    """Invert the local-current inserted source and return a response propagator.

    If ``tau_relative_list`` is ``None``, the inserted source covers all time
    slices.  Otherwise the relative-time window is summed before one inversion.
    """
    inserted_source = build_local_current_inserted_source(
        prop_forward,
        phase_q,
        source_time=source_time,
        current_gamma=current_gamma,
        tau_relative_list=tau_relative_list,
    )
    response = core.invertPropagator(dirac, inserted_source, mrhs, restart)
    response.data[:] *= response_sign
    return response


def invert_current_current_response_propagator(
    dirac,
    prop_forward,
    first_phase_q,
    second_phase_q,
    *,
    source_time,
    first_current_gamma="T",
    second_current_gamma="T",
    first_tau_relative_list=None,
    second_tau_relative_list=None,
    response_sign=1,
    mrhs=1,
    restart=0,
):
    """Build the nested local-current response ``D^-1 O2 D^-1 O1 S``.

    The implementation intentionally does not store per-tau response
    propagators.  Each tau window is summed at the inserted-source level before
    its inversion.  The output is useful as a second-order response diagnostic
    or as the starting point for current-current pion two-point contractions.
    """
    first_response = invert_local_current_response_propagator(
        dirac,
        prop_forward,
        first_phase_q,
        source_time=source_time,
        current_gamma=first_current_gamma,
        tau_relative_list=first_tau_relative_list,
        response_sign=1,
        mrhs=mrhs,
        restart=restart,
    )
    second_response = invert_local_current_response_propagator(
        dirac,
        first_response,
        second_phase_q,
        source_time=source_time,
        current_gamma=second_current_gamma,
        tau_relative_list=second_tau_relative_list,
        response_sign=response_sign,
        mrhs=mrhs,
        restart=restart,
    )
    return second_response


def contract_response_pion_2pt(
    latt_info,
    prop_response,
    prop_antiquark,
    sink_phases,
    src_gamma="5",
):
    """Contract the response propagator with the ordinary antiquark line.

    The output has the same gamma/momentum/time layout as
    ``contract_pion_2pt_multi_src_gamma(...)[src_gamma]``.
    """
    return contract_pion_2pt_multi_src_gamma(
        latt_info,
        prop_response,
        prop_antiquark,
        sink_phases,
        [src_gamma],
    )[src_gamma]


__all__ = [
    "build_local_current_inserted_source",
    "contract_response_pion_2pt",
    "invert_current_current_response_propagator",
    "invert_local_current_response_propagator",
    "relative_tau_to_absolute",
]
