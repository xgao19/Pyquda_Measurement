"""Analysis, time-window, and HDF5 helpers for pion current responses."""

import numpy as np

from pyquda_measurement_utils.io_corr import _prepare_h5_file
from pyquda_measurement_utils.pion_utils_vibe_develop import my_gammas


def infer_source_momentum(pf, qext):
    return [int(pf_i) - int(q_i) for pf_i, q_i in zip(pf[:3], qext[:3])]


def tau_window_list(tsep, Nt, window="all", tau_min=1):
    """Build a source-relative insertion-time window."""
    tsep, Nt, tau_min = int(tsep), int(Nt), int(tau_min)
    if Nt <= 0:
        raise ValueError("Nt must be positive")
    if not 0 <= tsep < Nt:
        raise ValueError(f"tsep={tsep} is outside the periodic time extent Nt={Nt}")
    if window == "all":
        return None
    if window == "source_sink":
        values = list(range(tsep + 1))
    elif window == "open":
        values = list(range(1, tsep))
    elif window == "restricted":
        if tau_min < 0:
            raise ValueError("tau_min must be non-negative")
        values = list(range(tau_min, tsep - tau_min + 1))
    elif window.startswith("range:"):
        start, stop = [int(item) for item in window.split(":", 1)[1].split("-")]
        values = list(range(start, stop + 1))
    else:
        raise ValueError(
            "window must be one of all, source_sink, open, restricted, "
            "or range:start-stop"
        )
    if any(tau < 0 or tau >= Nt for tau in values):
        raise ValueError(
            f"source-relative tau window {values} is outside the periodic "
            f"time extent Nt={Nt}"
        )
    return values


def roll_to_source_relative(values, source_time, axis=-1):
    return np.roll(np.asarray(values), -int(source_time), axis=axis)


def response_ratio(response_value, c2_value):
    return np.nan + 0j if abs(c2_value) == 0 else response_value / c2_value


def summed_explicit_emff(
    c3, current_gamma="T", q_index=0, tau_relative_list=None
):
    gamma_idx = my_gammas.index(current_gamma)
    values = np.asarray(c3)[gamma_idx, q_index]
    if tau_relative_list is not None:
        values = values[np.asarray(tau_relative_list, dtype=np.int64)]
    return np.sum(values)


def response_at_sink_time(corr_response, sink_gamma="5", p_index=0, tsep=0):
    gamma_idx = my_gammas.index(sink_gamma)
    return np.asarray(corr_response)[gamma_idx, p_index, int(tsep)]


def save_pion_EMFF_background_response_hdf5(tag, records, attrs=None):
    with _prepare_h5_file(f"{tag}.h5", attrs) as h5:
        h5.attrs["measurement"] = "pion_EMFF_background_response"
        h5.attrs["schema_version"] = "3"
        h5.attrs["time_axis"] = "source_relative"
        summary = h5.require_group("summary")
        summary.create_dataset("record_index", data=np.arange(len(records), dtype=np.int32))
        for key, dtype in (
            ("current_gamma", "S"),
            ("sink_gamma", "S"),
            ("src_gamma", "S"),
            ("tau_window", "S"),
            ("pf", np.int32),
            ("qext", np.int32),
            ("pi", np.int32),
            ("tsep", np.int32),
            ("tau_min", np.int32),
            ("q_index", np.int32),
            ("relative_difference", None),
            ("response_R_sum", None),
            ("explicit_R_sum", None),
            ("c2_tsep", None),
            ("response_c2_like", None),
            ("explicit_summed_c3", None),
        ):
            values = np.asarray([record[key] for record in records], dtype=dtype)
            summary.create_dataset(key, data=values)
        results = h5.require_group("results")
        for index, record in enumerate(records):
            group = results.require_group(f"record_{index:04d}")
            for key in (
                "current_gamma", "sink_gamma", "src_gamma", "tau_window",
                "response_sign", "finite_difference_derivative_sign",
            ):
                if key in record:
                    group.attrs[key] = record[key]
            for key in ("pf", "qext", "pi", "tsep", "tau_min", "q_index"):
                if key in record:
                    group.create_dataset(key, data=np.asarray(record[key]))
            for name in ("tau_relative_list", "tau_absolute_list"):
                values = record.get(name)
                group.create_dataset(
                    name,
                    data=np.asarray([] if values is None else values, dtype=np.int32),
                )
            group.attrs["tau_list_is_all_time_slices"] = (
                record.get("tau_relative_list") is None
            )
            for key in (
                "c2_tsep", "explicit_summed_c3", "response_c2_like",
                "response_R_sum", "explicit_R_sum", "difference",
                "relative_difference", "explicit_c3_all_tau",
                "response_corr_all_t", "c2_all_t",
            ):
                if key in record:
                    group.create_dataset(key, data=record[key])


def save_pion_current_current_response_hdf5(tag, records, attrs=None):
    with _prepare_h5_file(f"{tag}.h5", attrs) as h5:
        h5.attrs["measurement"] = "pion_current_current_response"
        h5.attrs["schema_version"] = "2"
        h5.attrs["current_order"] = "Dinv_O2_Dinv_O1_S"
        h5.attrs["time_axis"] = "source_relative"
        summary = h5.require_group("summary")
        summary.create_dataset("record_index", data=np.arange(len(records), dtype=np.int32))
        for key, dtype in (
            ("first_current_gamma", "S"),
            ("second_current_gamma", "S"),
            ("sink_gamma", "S"),
            ("src_gamma", "S"),
            ("first_tau_window", "S"),
            ("second_tau_window", "S"),
            ("pf", np.int32),
            ("first_qext", np.int32),
            ("second_qext", np.int32),
            ("total_qext", np.int32),
            ("pi", np.int32),
            ("tsep", np.int32),
            ("response_R_sum", None),
            ("response_c2_like", None),
            ("c2_tsep", None),
        ):
            summary.create_dataset(
                key,
                data=np.asarray([record[key] for record in records], dtype=dtype),
            )
        results = h5.require_group("results")
        for index, record in enumerate(records):
            group = results.require_group(f"record_{index:04d}")
            for key in (
                "first_current_gamma", "second_current_gamma", "sink_gamma",
                "src_gamma", "first_tau_window", "second_tau_window",
                "response_sign",
            ):
                if key in record:
                    group.attrs[key] = record[key]
            for key in (
                "pf", "first_qext", "second_qext", "total_qext", "pi",
                "tsep", "first_tau_min", "second_tau_min", "c2_tsep",
                "response_c2_like", "response_R_sum", "response_corr_all_t",
                "c2_all_t",
            ):
                if key in record:
                    group.create_dataset(key, data=record[key])
            for prefix in ("first", "second"):
                relative = record.get(f"{prefix}_tau_relative_list")
                absolute = record.get(f"{prefix}_tau_absolute_list")
                group.create_dataset(
                    f"{prefix}_tau_relative_list",
                    data=np.asarray(
                        [] if relative is None else relative, dtype=np.int32
                    ),
                )
                group.create_dataset(
                    f"{prefix}_tau_absolute_list",
                    data=np.asarray(
                        [] if absolute is None else absolute, dtype=np.int32
                    ),
                )
                group.attrs[f"{prefix}_tau_list_is_all_time_slices"] = (
                    relative is None
                )


__all__ = [
    "infer_source_momentum",
    "response_at_sink_time",
    "response_ratio",
    "roll_to_source_relative",
    "save_pion_EMFF_background_response_hdf5",
    "save_pion_current_current_response_hdf5",
    "summed_explicit_emff",
    "tau_window_list",
]
