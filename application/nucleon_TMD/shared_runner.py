"""Shared connected-proton qTMD/PDF production runner."""

from dataclasses import dataclass, field
import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class PlatformDefaults:
    name: str
    mpi_geometry: str
    gauge_path: str
    data_dir: str
    lat_tag: str
    mass: float
    csw: float
    tol: float
    maxiter: int
    width: float
    num_src: int
    qmax: int
    b_z: int
    b_T: int
    eta: int
    t_separations: tuple
    stream: str = ""
    source_shift: tuple = (0, 0, 0, 0)
    init_kwargs: dict = field(default_factory=dict)


def _parse_t_separations(value):
    try:
        return tuple(
            int(field)
            for field in str(value).replace(".", ",").split(",")
            if field
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid separations {value!r}; expected comma-separated integers"
        ) from exc


def _parse_single_t_separations(value):
    values = _parse_t_separations(value)
    if len(values) != 1:
        raise argparse.ArgumentTypeError(
            "proton qTMD requires exactly one sink separation"
        )
    return values


def build_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_num", type=int, required=True)
    parser.add_argument(
        "--mpi_geometry",
        default=os.environ.get(
            "NUCLEON_TMD_MPI_GEOMETRY", defaults.mpi_geometry
        ),
    )
    parser.add_argument(
        "--gauge_path",
        default=os.environ.get("NUCLEON_TMD_GAUGE_PATH", defaults.gauge_path),
    )
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("NUCLEON_TMD_DATA_DIR", defaults.data_dir),
    )
    parser.add_argument(
        "--lat-tag",
        default=os.environ.get("NUCLEON_TMD_LAT_TAG", defaults.lat_tag),
    )
    parser.add_argument("--stream", default=defaults.stream)
    parser.add_argument(
        "--num_src",
        type=int,
        default=int(os.environ.get("NUCLEON_TMD_NUM_SRC", defaults.num_src)),
    )
    parser.add_argument(
        "--qmax",
        type=int,
        default=int(os.environ.get("NUCLEON_TMD_QMAX", defaults.qmax)),
    )
    parser.add_argument(
        "--b_z",
        type=int,
        default=int(os.environ.get("NUCLEON_TMD_BZ", defaults.b_z)),
    )
    parser.add_argument(
        "--b_T",
        type=int,
        default=int(os.environ.get("NUCLEON_TMD_BT", defaults.b_T)),
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=int(os.environ.get("NUCLEON_TMD_ETA", defaults.eta)),
    )
    parser.add_argument(
        "--t_separations",
        type=_parse_single_t_separations,
        default=_parse_single_t_separations(
            ",".join(str(value) for value in defaults.t_separations)
        ),
    )
    parser.add_argument(
        "--width",
        type=float,
        default=float(os.environ.get("NUCLEON_TMD_WIDTH", defaults.width)),
    )
    parser.add_argument(
        "--interpolator",
        default=os.environ.get("NUCLEON_TMD_INTERPOLATOR", "5"),
    )
    parser.add_argument(
        "--pol", default=os.environ.get("NUCLEON_TMD_POL", "PpUnpol")
    )
    parser.add_argument(
        "--run_cg_qtmd",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("NUCLEON_TMD_RUN_CG_QTMD", 1)),
    )
    parser.add_argument(
        "--run_gi_qtmd",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("NUCLEON_TMD_RUN_GI_QTMD", 1)),
    )
    parser.add_argument(
        "--run_pdf",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("NUCLEON_TMD_RUN_PDF", 1)),
    )
    parser.add_argument(
        "--mg-block", default="8.8.4.4", help="X.Y.Z.T[;...] or none"
    )
    return parser


def _format_path(value, *, config_num, stream):
    return str(value).format(conf=config_num, config_num=config_num, stream=stream)


def _sync_backend_array(arr):
    stream = getattr(arr, "stream", None)
    if stream is not None:
        stream.synchronize()
    queue = getattr(arr, "sycl_queue", None)
    if queue is not None:
        queue.wait()


def _contract_operator_list(
    *,
    latt_info,
    gauge,
    prop_f,
    seq_down,
    seq_up,
    gamma_ls,
    phases,
    wilson_indices,
    operator_kind,
):
    from pyquda import getMPIComm
    from pyquda_utils import core
    from pyquda_measurement_utils.qtmd_operator_utils import (
        apply_gi_qtmd_staple_to_propagator,
        build_gi_qtmd_staple_links,
        shift_propagator_pdf_gi,
        shift_qtmd_cg,
    )
    from pyquda_measurement_utils.tools import (
        _asarray_on_queue,
        _get_xp_from_array,
        array_to_numpy,
        mpi_print,
    )

    xp = _get_xp_from_array(prop_f.data)
    phases = _asarray_on_queue(phases, xp, prop_f.data)
    corr_down = []
    corr_up = []
    shifted_prop = prop_f.copy()
    staple_links = None

    if operator_kind == "GI_qTMD":
        mpi_print(
            latt_info,
            f"Build {len(wilson_indices)} connected nucleon GI staple transporters.",
        )
        staple_links = build_gi_qtmd_staple_links(gauge, wilson_indices)

    for index, wilson_index in enumerate(wilson_indices):
        mpi_print(
            latt_info,
            f"Contract {operator_kind} {index + 1}/{len(wilson_indices)} "
            f"{wilson_index}",
        )
        if operator_kind == "CG_qTMD":
            previous = (
                [0, 0, 0, wilson_index[3]]
                if index == 0
                else wilson_indices[index - 1]
            )
            if wilson_index[3] != previous[3]:
                shifted_prop = prop_f.copy()
                previous = [0, 0, 0, wilson_index[3]]
            shifted_prop = shift_qtmd_cg(
                shifted_prop, wilson_index, previous
            )
            current_prop = shifted_prop
        elif operator_kind in {"CG_PDF", "GI_PDF"}:
            if wilson_index[1] in {0, -1}:
                shifted_prop = prop_f.copy()
                previous = [0, 0, 0, 0]
            else:
                previous = wilson_indices[index - 1]
            if operator_kind == "CG_PDF":
                shifted_prop = shift_qtmd_cg(
                    shifted_prop, wilson_index, previous
                )
            else:
                shifted_prop = shift_propagator_pdf_gi(
                    gauge, shifted_prop, wilson_index, previous
                )
            current_prop = shifted_prop
        elif operator_kind == "GI_qTMD":
            current_prop = apply_gi_qtmd_staple_to_propagator(
                prop_f, wilson_index, staple_links
            )
        else:
            raise ValueError(f"unsupported operator kind {operator_kind!r}")

        for sequential, output in (
            (seq_down, corr_down),
            (seq_up, corr_up),
        ):
            scalar = xp.einsum(
                "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
                sequential,
                gamma_ls,
                current_prop.data,
                optimize=True,
            )
            projected = xp.einsum(
                "qwtzyx,pgwtzyx->pqgt", phases, scalar, optimize=True
            )
            output.append(
                core.gatherLattice(
                    array_to_numpy(projected), [3, -1, -1, -1]
                )
            )
            _sync_backend_array(projected)
            del scalar, projected

        if operator_kind == "GI_qTMD":
            del current_prop

    if getMPIComm().Get_rank() != 0:
        return None, None
    return np.asarray(corr_down), np.asarray(corr_up)


def _roll_trim_root(corr, source_time, t_sep):
    if corr is None:
        return None
    return np.roll(corr, -int(source_time), axis=-1)[..., : int(t_sep) + 2]


def _save_qtmd_by_gamma(
    *,
    latt_info,
    data_dir,
    lat_tag,
    config_num,
    operator_tag,
    source_position,
    smearing_tag,
    corr_down,
    corr_up,
    momentum_list,
    wilson_indices,
    t_sep,
):
    if latt_info.mpi_rank != 0:
        return
    from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS
    from pyquda_measurement_utils.io_corr import (
        get_qTMD_file_tag,
        save_qTMD_proton_hdf5_noRoll,
    )

    for gamma_index, gamma_label in enumerate(GAMMA_LABELS):
        for flavor, corr in (("D", corr_down), ("U", corr_up)):
            tag = get_qTMD_file_tag(
                str(data_dir),
                lat_tag,
                config_num,
                f"{operator_tag}.{flavor}.ex",
                source_position,
                f"{smearing_tag}.{gamma_label}",
            )
            save_qTMD_proton_hdf5_noRoll(
                corr[:, 0, :, gamma_index : gamma_index + 1, :],
                tag,
                [gamma_label],
                momentum_list,
                wilson_indices,
                t_sep,
                latt_info,
            )


def _save_pdf(
    *,
    latt_info,
    data_dir,
    lat_tag,
    config_num,
    operator_tag,
    source_position,
    smearing_tag,
    corr_down,
    corr_up,
    momentum_list,
    wilson_indices,
    t_sep,
):
    if latt_info.mpi_rank != 0:
        return
    from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS
    from pyquda_measurement_utils.io_corr import (
        get_qTMD_file_tag,
        save_qTMD_proton_hdf5_noRoll,
    )

    for flavor, corr in (("D", corr_down), ("U", corr_up)):
        tag = get_qTMD_file_tag(
            str(data_dir),
            lat_tag,
            config_num,
            f"{operator_tag}.{flavor}.ex",
            source_position,
            smearing_tag,
        )
        save_qTMD_proton_hdf5_noRoll(
            corr[:, 0, :, :, :],
            tag,
            list(GAMMA_LABELS),
            momentum_list,
            wilson_indices,
            t_sep,
            latt_info,
        )


def run(defaults, argv=None):
    args = build_parser(defaults).parse_args(argv)
    t_sep = args.t_separations[0]
    mpi_geometry = [int(entry) for entry in args.mpi_geometry.split(".")]
    if len(mpi_geometry) != 4 or any(entry <= 0 for entry in mpi_geometry):
        raise ValueError("--mpi_geometry must contain four positive integers")

    from pyquda import getMPIComm, init

    init(mpi_geometry, enable_mps=True, **dict(defaults.init_kwargs))

    from pyquda_utils import core, io, phase, source
    from pyquda_utils.phase import MomentumPhase
    from pyquda_measurement_utils.boosted_smearing_pyquda import (
        boosted_smearing,
    )
    from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_pyquda
    from pyquda_measurement_utils.fermion_bilinear_basis import gamma_stack
    from pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop import (
        parse_optional_multigrid_blocks,
    )
    from pyquda_measurement_utils.io_corr import (
        get_c2pt_file_tag,
        get_sample_log_tag,
    )
    from pyquda_measurement_utils.proton_qTMD_pyquda import proton_TMD
    from pyquda_measurement_utils.qtmd_operator_utils import (
        create_cg_qtmd_wilsonline_index_lists,
        create_gi_qtmd_wilsonline_index_lists,
        create_pdf_wilsonline_index_list,
    )
    from pyquda_measurement_utils.tools import (
        append_sample_log_entry,
        mpi_print,
        read_sample_log_entries,
        srcLoc_distri_eq,
    )

    config_num = int(args.config_num)
    data_dir = Path(
        _format_path(
            args.data_dir, config_num=config_num, stream=args.stream
        )
    )
    gauge_path = _format_path(
        args.gauge_path, config_num=config_num, stream=args.stream
    )
    run_cg_qtmd = bool(args.run_cg_qtmd)
    run_gi_qtmd = bool(args.run_gi_qtmd)
    run_pdf = bool(args.run_pdf)
    q_range = range(-args.qmax, args.qmax + 1)
    qext = [[x, y, 0, 0] for x in q_range for y in q_range]
    qext_pdf = [
        [x, y, z, 0] for x in q_range for y in q_range for z in q_range
    ]
    p_2pt = list(qext_pdf)
    pol_list = [args.pol]
    parameters = {
        "eta": [args.eta],
        "b_z": args.b_z,
        "b_T": args.b_T,
        "qext": qext,
        "qext_PDF": qext_pdf,
        "pf": [0, 0, 0, 0],
        "p_2pt": p_2pt,
        "boost_in": [0, 0, 0],
        "boost_out": [0, 0, 0],
        "width": args.width,
        "pol": pol_list,
    }
    measurement = proton_TMD(parameters)
    pf = parameters["pf"]
    pf_tag = (
        f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{t_sep}"
    )
    sm_tag = os.environ.get(
        "NUCLEON_TMD_SM_TAG",
        f"1HYP_GSRC_W{args.width:g}_k0_{args.interpolator}",
    )
    output_tag = f"{sm_tag}.{pf_tag}.{args.pol}"

    if getMPIComm().Get_rank() == 0:
        for subdirectory in ("sample_log_qtmd", "c2pt", "qTMD"):
            (data_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    getMPIComm().Barrier()

    gauge = io.readNERSCGauge(gauge_path)
    gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
    gauge.latt_info.t_boundary = -1
    latt_info = gauge.latt_info
    mpi_print(latt_info, f"--platform {defaults.name}")
    mpi_print(latt_info, f"--gauge_path {gauge_path}")
    mpi_print(latt_info, f"--data_dir {data_dir}")
    mpi_print(latt_info, f"--config_num {config_num}")
    mpi_print(latt_info, f"--mg-block {args.mg_block}")
    mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

    mass = float(os.environ.get("NUCLEON_TMD_MASS", defaults.mass))
    csw = float(os.environ.get("NUCLEON_TMD_CSW", defaults.csw))
    tol = float(os.environ.get("NUCLEON_TMD_TOL", defaults.tol))
    maxiter = int(os.environ.get("NUCLEON_TMD_MAXITER", defaults.maxiter))
    multigrid = parse_optional_multigrid_blocks(args.mg_block)
    dirac = core.getDirac(
        latt_info, mass, tol, maxiter, 1.0, csw, csw, multigrid
    )
    dirac.loadGauge(gauge)
    gamma_ls = gamma_stack(gauge.data).astype(
        gauge.data.dtype, copy=False
    )

    lattice_size = latt_info.global_size
    source_origin = np.array(
        [
            int(config_num) % lattice_size[direction]
            for direction in range(4)
        ]
    ) + np.asarray(defaults.source_shift)
    source_positions = srcLoc_distri_eq(
        lattice_size, source_origin
    )[: args.num_src]

    sample_log_file = (
        data_dir / "sample_log_qtmd" / f"{config_num}_{sm_tag}_{pf_tag}"
    )
    if latt_info.mpi_rank == 0:
        completed = read_sample_log_entries(sample_log_file)
    else:
        completed = None
    completed = set(getMPIComm().bcast(completed, root=0))

    for source_position in source_positions:
        entry = get_sample_log_tag(
            str(config_num), source_position, f"{sm_tag}_{pf_tag}"
        )
        if entry in completed:
            mpi_print(latt_info, f"SKIP: {entry}")
            continue

        source_started = perf_counter()
        mpi_print(latt_info, f"START: {entry}")
        point_source = source.propagator(
            latt_info, "point", source_position
        )
        smeared_source = boosted_smearing(
            point_source,
            w=parameters["width"],
            boost=parameters["boost_in"],
        )
        prop_fw = core.invertPropagator(dirac, smeared_source, 1, 0)

        c2_tag = get_c2pt_file_tag(
            str(data_dir),
            args.lat_tag,
            config_num,
            "ex",
            source_position,
            sm_tag,
        )
        phases_2pt = MomentumPhase(latt_info).getPhases(
            [[-v[0], -v[1], -v[2]] for v in p_2pt],
            x0=source_position,
        )
        measurement.contract_2pt_TMD(
            latt_info,
            prop_fw,
            phases_2pt,
            c2_tag,
            args.interpolator,
        )

        seq_down = create_bw_seq_pyquda(
            dirac,
            prop_fw,
            source_position,
            parameters["width"],
            parameters["boost_out"],
            parameters["pf"],
            t_sep,
            parameters["pol"],
            2,
            args.interpolator,
        )
        seq_up = create_bw_seq_pyquda(
            dirac,
            prop_fw,
            source_position,
            parameters["width"],
            parameters["boost_out"],
            parameters["pf"],
            t_sep,
            parameters["pol"],
            1,
            args.interpolator,
        )

        phases_tmd = phase.MomentumPhase(latt_info).getPhases(
            [[v[0], v[1], v[2]] for v in qext], source_position
        )
        phases_pdf = MomentumPhase(latt_info).getPhases(
            [[v[0], v[1], v[2]] for v in qext_pdf],
            x0=source_position,
        )
        cg_dir0, cg_dir1 = create_cg_qtmd_wilsonline_index_lists(
            parameters["b_z"], parameters["b_T"]
        )
        cg_indices = cg_dir0 + cg_dir1
        gi_dir0, gi_dir1 = create_gi_qtmd_wilsonline_index_lists(
            parameters["eta"], parameters["b_z"], parameters["b_T"]
        )
        gi_indices = gi_dir0 + gi_dir1
        pdf_indices = create_pdf_wilsonline_index_list(
            parameters["b_z"]
        )

        if run_cg_qtmd:
            down, up = _contract_operator_list(
                latt_info=latt_info,
                gauge=gauge,
                prop_f=prop_fw,
                seq_down=seq_down,
                seq_up=seq_up,
                gamma_ls=gamma_ls,
                phases=phases_tmd,
                wilson_indices=cg_indices,
                operator_kind="CG_qTMD",
            )
            _save_qtmd_by_gamma(
                latt_info=latt_info,
                data_dir=data_dir,
                lat_tag=args.lat_tag,
                config_num=config_num,
                operator_tag="CG",
                source_position=source_position,
                smearing_tag=output_tag,
                corr_down=_roll_trim_root(
                    down, source_position[3], t_sep
                ),
                corr_up=_roll_trim_root(
                    up, source_position[3], t_sep
                ),
                momentum_list=qext,
                wilson_indices=cg_indices,
                t_sep=t_sep,
            )

        if run_gi_qtmd:
            down, up = _contract_operator_list(
                latt_info=latt_info,
                gauge=gauge,
                prop_f=prop_fw,
                seq_down=seq_down,
                seq_up=seq_up,
                gamma_ls=gamma_ls,
                phases=phases_tmd,
                wilson_indices=gi_indices,
                operator_kind="GI_qTMD",
            )
            _save_qtmd_by_gamma(
                latt_info=latt_info,
                data_dir=data_dir,
                lat_tag=args.lat_tag,
                config_num=config_num,
                operator_tag="GI_qTMD",
                source_position=source_position,
                smearing_tag=output_tag,
                corr_down=_roll_trim_root(
                    down, source_position[3], t_sep
                ),
                corr_up=_roll_trim_root(
                    up, source_position[3], t_sep
                ),
                momentum_list=qext,
                wilson_indices=gi_indices,
                t_sep=t_sep,
            )

        if run_pdf:
            for operator_kind in ("GI_PDF", "CG_PDF"):
                down, up = _contract_operator_list(
                    latt_info=latt_info,
                    gauge=gauge,
                    prop_f=prop_fw,
                    seq_down=seq_down,
                    seq_up=seq_up,
                    gamma_ls=gamma_ls,
                    phases=phases_pdf,
                    wilson_indices=pdf_indices,
                    operator_kind=operator_kind,
                )
                _save_pdf(
                    latt_info=latt_info,
                    data_dir=data_dir,
                    lat_tag=args.lat_tag,
                    config_num=config_num,
                    operator_tag=operator_kind,
                    source_position=source_position,
                    smearing_tag=output_tag,
                    corr_down=_roll_trim_root(
                        down, source_position[3], t_sep
                    ),
                    corr_up=_roll_trim_root(
                        up, source_position[3], t_sep
                    ),
                    momentum_list=qext_pdf,
                    wilson_indices=pdf_indices,
                    t_sep=t_sep,
                )

        _sync_backend_array(prop_fw.data)
        if latt_info.mpi_rank == 0:
            append_sample_log_entry(sample_log_file, entry)
        completed.add(entry)
        getMPIComm().Barrier()
        mpi_print(
            latt_info,
            f"DONE: {entry} total {perf_counter() - source_started:.6f}s",
        )


__all__ = ["PlatformDefaults", "build_parser", "run"]
