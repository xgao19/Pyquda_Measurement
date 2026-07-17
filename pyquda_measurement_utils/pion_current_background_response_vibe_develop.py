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

import numpy as np

from pyquda_utils import core, gamma, source

from pyquda_measurement_utils.io_corr import _prepare_h5_file
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt_multi_src_gamma,
    gamma_from_label,
    my_gammas,
)
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


def _gamma_matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def _gamma_on_backend(gamma_like, xp, ref_arr):
    return _asarray_on_queue(_gamma_matrix(gamma_like), xp, ref_arr)


def infer_source_momentum(pf, qext):
    """Return the source pion momentum convention pi = pf - qext."""
    return [int(pf_i) - int(q_i) for pf_i, q_i in zip(pf[:3], qext[:3])]


def tau_window_list(tsep, Nt, window="all", tau_min=1):
    """Build a source-relative insertion-time window."""
    tsep = int(tsep)
    Nt = int(Nt)
    tau_min = int(tau_min)
    if Nt <= 0:
        raise ValueError("Nt must be positive")
    if not 0 <= tsep < Nt:
        raise ValueError(f"tsep={tsep} is outside the periodic time extent Nt={Nt}")
    if window == "all":
        return None
    if window == "source_sink":
        tau_relative_list = list(range(0, tsep + 1))
    elif window == "open":
        tau_relative_list = list(range(1, tsep))
    elif window == "restricted":
        if tau_min < 0:
            raise ValueError("tau_min must be non-negative")
        tau_relative_list = list(range(tau_min, tsep - tau_min + 1))
    elif window.startswith("range:"):
        start, stop = [int(item) for item in window.split(":", 1)[1].split("-")]
        tau_relative_list = list(range(start, stop + 1))
    else:
        raise ValueError("window must be one of all, source_sink, open, restricted, or range:start-stop")
    if any(tau < 0 or tau >= Nt for tau in tau_relative_list):
        raise ValueError(
            f"source-relative tau window {tau_relative_list} is outside the periodic time extent Nt={Nt}"
        )
    return tau_relative_list


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


def roll_to_source_relative(values, source_time, axis=-1):
    """Roll a global-time array so index zero is the source time."""
    return np.roll(np.asarray(values), -int(source_time), axis=axis)


def response_ratio(response_value, c2_value):
    """Return C_response / C2 with a stable zero-denominator guard."""
    if abs(c2_value) == 0:
        return np.nan + 0j
    return response_value / c2_value


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
    gamma_current = _gamma_on_backend(gamma_current, xp, prop_forward.data)

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


def current_current_response_toy(prop_forward, first_phase, second_phase, first_gamma, second_gamma):
    """Small NumPy diagnostic for the nested current-current response algebra."""
    return np.einsum(
        "t,ab,t,bc,tcd->tad",
        second_phase,
        second_gamma,
        first_phase,
        first_gamma,
        prop_forward,
        optimize=True,
    )


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


def contract_current_current_response_pion_2pt(
    latt_info,
    prop_current_current_response,
    prop_antiquark,
    sink_phases,
    src_gamma="5",
):
    """Contract a nested current-current response with the pion antiquark line."""
    return contract_response_pion_2pt(
        latt_info,
        prop_current_current_response,
        prop_antiquark,
        sink_phases,
        src_gamma=src_gamma,
    )


def summed_explicit_emff(c3, current_gamma="T", q_index=0, tau_relative_list=None):
    """Select one channel from source-relative EMFF C3 and sum relative tau."""
    gamma_idx = my_gammas.index(current_gamma)
    values = np.asarray(c3)[gamma_idx, q_index]
    if tau_relative_list is not None:
        values = values[np.asarray(tau_relative_list, dtype=np.int64)]
    return np.sum(values)


def response_at_sink_time(corr_response, sink_gamma="5", p_index=0, tsep=0):
    """Select one sink gamma/momentum/time point from a response 2pt array."""
    gamma_idx = my_gammas.index(sink_gamma)
    return np.asarray(corr_response)[gamma_idx, p_index, int(tsep)]


def save_pion_EMFF_background_response_hdf5(tag, records, attrs=None):
    """Save pion EMFF background-response diagnostics with explicit metadata.

    ``records`` is a list of dictionaries.  Each record stores one
    (current gamma, qext, tsep, tau window) comparison between the explicit
    summed C3 and the response-propagator C2-like contraction.
    """
    with _prepare_h5_file(f"{tag}.h5", attrs) as h5:
        h5.attrs["measurement"] = "pion_EMFF_background_response"
        h5.attrs["schema_version"] = "3"
        h5.attrs["time_axis"] = "source_relative"

        summary = h5.require_group("summary")
        summary.create_dataset("record_index", data=np.arange(len(records), dtype=np.int32))
        summary.create_dataset("current_gamma", data=np.asarray([record["current_gamma"] for record in records], dtype="S"))
        summary.create_dataset("sink_gamma", data=np.asarray([record["sink_gamma"] for record in records], dtype="S"))
        summary.create_dataset("src_gamma", data=np.asarray([record["src_gamma"] for record in records], dtype="S"))
        summary.create_dataset("tau_window", data=np.asarray([record["tau_window"] for record in records], dtype="S"))
        summary.create_dataset("pf", data=np.asarray([record["pf"] for record in records], dtype=np.int32))
        summary.create_dataset("qext", data=np.asarray([record["qext"] for record in records], dtype=np.int32))
        summary.create_dataset("pi", data=np.asarray([record["pi"] for record in records], dtype=np.int32))
        summary.create_dataset("tsep", data=np.asarray([record["tsep"] for record in records], dtype=np.int32))
        summary.create_dataset("tau_min", data=np.asarray([record["tau_min"] for record in records], dtype=np.int32))
        summary.create_dataset("q_index", data=np.asarray([record["q_index"] for record in records], dtype=np.int32))
        summary.create_dataset("relative_difference", data=np.asarray([record["relative_difference"] for record in records]))
        summary.create_dataset("response_R_sum", data=np.asarray([record["response_R_sum"] for record in records]))
        summary.create_dataset("explicit_R_sum", data=np.asarray([record["explicit_R_sum"] for record in records]))
        summary.create_dataset("c2_tsep", data=np.asarray([record["c2_tsep"] for record in records]))
        summary.create_dataset("response_c2_like", data=np.asarray([record["response_c2_like"] for record in records]))
        summary.create_dataset("explicit_summed_c3", data=np.asarray([record["explicit_summed_c3"] for record in records]))

        results = h5.require_group("results")
        for irec, record in enumerate(records):
            group = results.require_group(f"record_{irec:04d}")
            for key in (
                "current_gamma",
                "sink_gamma",
                "src_gamma",
                "tau_window",
                "response_sign",
                "finite_difference_derivative_sign",
            ):
                if key in record:
                    group.attrs[key] = record[key]
            for key in ("pf", "qext", "pi", "tsep", "tau_min", "q_index"):
                if key in record:
                    group.create_dataset(key, data=np.asarray(record[key]))
            tau_relative_list = record.get("tau_relative_list")
            tau_absolute_list = record.get("tau_absolute_list")
            group.create_dataset(
                "tau_relative_list",
                data=np.asarray(
                    [] if tau_relative_list is None else tau_relative_list,
                    dtype=np.int32,
                ),
            )
            group.create_dataset(
                "tau_absolute_list",
                data=np.asarray(
                    [] if tau_absolute_list is None else tau_absolute_list,
                    dtype=np.int32,
                ),
            )
            group.attrs["tau_list_is_all_time_slices"] = tau_relative_list is None
            for key in (
                "c2_tsep",
                "explicit_summed_c3",
                "response_c2_like",
                "response_R_sum",
                "explicit_R_sum",
                "difference",
                "relative_difference",
                "explicit_c3_all_tau",
                "response_corr_all_t",
                "c2_all_t",
            ):
                if key in record:
                    group.create_dataset(key, data=record[key])


def save_pion_current_current_response_hdf5(tag, records, attrs=None):
    """Save nested pion current-current response diagnostics.

    Each record stores one ``D^-1 O2 D^-1 O1 S`` contraction with the pion
    two-point sink line.  The schema is intentionally compact because this is a
    diagnostic building block rather than a production hadronic-tensor writer.
    """
    with _prepare_h5_file(f"{tag}.h5", attrs) as h5:
        h5.attrs["measurement"] = "pion_current_current_response"
        h5.attrs["schema_version"] = "2"
        h5.attrs["current_order"] = "Dinv_O2_Dinv_O1_S"
        h5.attrs["time_axis"] = "source_relative"

        summary = h5.require_group("summary")
        summary.create_dataset("record_index", data=np.arange(len(records), dtype=np.int32))
        summary.create_dataset("first_current_gamma", data=np.asarray([record["first_current_gamma"] for record in records], dtype="S"))
        summary.create_dataset("second_current_gamma", data=np.asarray([record["second_current_gamma"] for record in records], dtype="S"))
        summary.create_dataset("sink_gamma", data=np.asarray([record["sink_gamma"] for record in records], dtype="S"))
        summary.create_dataset("src_gamma", data=np.asarray([record["src_gamma"] for record in records], dtype="S"))
        summary.create_dataset("first_tau_window", data=np.asarray([record["first_tau_window"] for record in records], dtype="S"))
        summary.create_dataset("second_tau_window", data=np.asarray([record["second_tau_window"] for record in records], dtype="S"))
        summary.create_dataset("pf", data=np.asarray([record["pf"] for record in records], dtype=np.int32))
        summary.create_dataset("first_qext", data=np.asarray([record["first_qext"] for record in records], dtype=np.int32))
        summary.create_dataset("second_qext", data=np.asarray([record["second_qext"] for record in records], dtype=np.int32))
        summary.create_dataset("total_qext", data=np.asarray([record["total_qext"] for record in records], dtype=np.int32))
        summary.create_dataset("pi", data=np.asarray([record["pi"] for record in records], dtype=np.int32))
        summary.create_dataset("tsep", data=np.asarray([record["tsep"] for record in records], dtype=np.int32))
        summary.create_dataset("response_R_sum", data=np.asarray([record["response_R_sum"] for record in records]))
        summary.create_dataset("response_c2_like", data=np.asarray([record["response_c2_like"] for record in records]))
        summary.create_dataset("c2_tsep", data=np.asarray([record["c2_tsep"] for record in records]))

        results = h5.require_group("results")
        for irec, record in enumerate(records):
            group = results.require_group(f"record_{irec:04d}")
            for key in (
                "first_current_gamma",
                "second_current_gamma",
                "sink_gamma",
                "src_gamma",
                "first_tau_window",
                "second_tau_window",
                "response_sign",
            ):
                if key in record:
                    group.attrs[key] = record[key]
            for key in (
                "pf",
                "first_qext",
                "second_qext",
                "total_qext",
                "pi",
                "tsep",
                "first_tau_min",
                "second_tau_min",
                "c2_tsep",
                "response_c2_like",
                "response_R_sum",
                "response_corr_all_t",
                "c2_all_t",
            ):
                if key in record:
                    group.create_dataset(key, data=record[key])
            for prefix in ("first", "second"):
                tau_relative_list = record.get(f"{prefix}_tau_relative_list")
                tau_absolute_list = record.get(f"{prefix}_tau_absolute_list")
                group.create_dataset(
                    f"{prefix}_tau_relative_list",
                    data=np.asarray(
                        [] if tau_relative_list is None else tau_relative_list,
                        dtype=np.int32,
                    ),
                )
                group.create_dataset(
                    f"{prefix}_tau_absolute_list",
                    data=np.asarray(
                        [] if tau_absolute_list is None else tau_absolute_list,
                        dtype=np.int32,
                    ),
                )
                group.attrs[f"{prefix}_tau_list_is_all_time_slices"] = (
                    tau_relative_list is None
                )
