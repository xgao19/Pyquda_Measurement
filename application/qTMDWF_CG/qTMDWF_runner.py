"""Shared application runner for pion CG qTMDWF production."""

from pathlib import Path
from time import perf_counter

import numpy as np

from pyquda import getMPIComm
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.io_corr import (
    get_c2pt_file_tag,
    get_qTMDWF_file_tag,
    get_sample_log_tag,
    save_qTMDWF_hdf5_noRoll,
)
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    build_pion_source_propagators,
)
from pyquda_measurement_utils.tools import (
    append_sample_log_entry,
    mpi_print,
    read_sample_log_entries,
)


def run_qtmdwf_sources(
    *,
    latt_info,
    dirac,
    gauge,
    measurement,
    source_positions,
    data_dir,
    lat_tag,
    config_num,
    sm_tag,
    source_gamma_label="5",
):
    """Run C2 and CG qTMDWF, with one root-written 16-Gamma file per source."""
    data_dir = str(data_dir)
    sample_log = Path(data_dir) / "sample_log" / (
        f"TMDWF_{sm_tag}_{int(config_num)}"
    )
    if latt_info.mpi_rank == 0:
        completed = read_sample_log_entries(sample_log)
    else:
        completed = None
    completed = set(getMPIComm().bcast(completed, root=0))

    pz_values = range(measurement.pzmin, measurement.pzmax)
    momentum_list = [[0, 0, pz, 0] for pz in pz_values]
    phase_momenta = [[0, 0, -pz] for pz in pz_values]
    w_dir0, w_dir1 = measurement.create_TMD_Wilsonline_index_list_CG()
    wilson_indices = w_dir0 + w_dir1

    for source_position in source_positions:
        entry = get_sample_log_tag(
            "ex",
            source_position,
            f"{sm_tag}.c2src{source_gamma_label}",
        )
        if entry in completed:
            mpi_print(latt_info, f"Contraction SKIP: {entry}")
            continue

        started = perf_counter()
        dirac.loadGauge(gauge)
        prop_forward, prop_backward = build_pion_source_propagators(
            dirac,
            latt_info,
            source_position,
            gaussian_smearing=True,
            width=measurement.width,
            pos_boost=measurement.pos_boost,
            neg_boost=measurement.neg_boost,
        )
        phases = MomentumPhase(latt_info).getPhases(
            phase_momenta, x0=source_position
        )
        c2_tag = get_c2pt_file_tag(
            data_dir,
            lat_tag,
            config_num,
            "ex",
            source_position,
            sm_tag,
        )
        measurement.contract_2pt_pion(
            latt_info,
            prop_forward,
            prop_backward,
            phases,
            f"{c2_tag}.src{source_gamma_label}",
            source_gamma_label=source_gamma_label,
        )
        corr_by_source = measurement.contract_qTMDWF_CG(
            latt_info,
            prop_forward,
            prop_backward,
            phases,
            w_dir0,
            w_dir1,
            [source_gamma_label],
        )
        if latt_info.mpi_rank == 0:
            corr = np.roll(
                corr_by_source[source_gamma_label],
                -int(source_position[3]),
                axis=-1,
            )
            tag = get_qTMDWF_file_tag(
                data_dir,
                lat_tag,
                config_num,
                "ex",
                source_position,
                f"{sm_tag}.src{source_gamma_label}",
            )
            Path(tag).parent.mkdir(parents=True, exist_ok=True)
            save_qTMDWF_hdf5_noRoll(
                corr,
                tag,
                measurement_gamma_labels(),
                momentum_list,
                wilson_indices,
            )
            append_sample_log_entry(sample_log, entry)
        getMPIComm().Barrier()
        mpi_print(latt_info, f"DONE: {entry} {perf_counter() - started:.3f}s")


def measurement_gamma_labels():
    from pyquda_measurement_utils.pion_utils_vibe_develop import my_gammas

    return list(my_gammas)


__all__ = ["run_qtmdwf_sources"]
