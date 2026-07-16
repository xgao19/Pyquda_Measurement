import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("NUCLEON_TMD_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("NUCLEON_TMD_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("NUCLEON_TMD_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("NUCLEON_TMD_DATA_DIR", ""))
parser.add_argument("--num_src", type=int, default=int(os.environ.get("NUCLEON_TMD_NUM_SRC", 1)))
parser.add_argument("--qmax", type=int, default=int(os.environ.get("NUCLEON_TMD_QMAX", 0)))
parser.add_argument("--b_z", type=int, default=int(os.environ.get("NUCLEON_TMD_BZ", 2)))
parser.add_argument("--b_T", type=int, default=int(os.environ.get("NUCLEON_TMD_BT", 1)))
parser.add_argument("--eta", type=int, default=int(os.environ.get("NUCLEON_TMD_ETA", 1)))
parser.add_argument("--t_insert", type=int, default=int(os.environ.get("NUCLEON_TMD_T_INSERT", 2)))
parser.add_argument("--width", type=float, default=float(os.environ.get("NUCLEON_TMD_WIDTH", 1.0)))
parser.add_argument("--interpolator", type=str, default=os.environ.get("NUCLEON_TMD_INTERPOLATOR", "5"))
parser.add_argument("--pol", type=str, default=os.environ.get("NUCLEON_TMD_POL", "PpUnpol"))
parser.add_argument("--run_cg_qtmd", type=int, default=int(os.environ.get("NUCLEON_TMD_RUN_CG_QTMD", 1)))
parser.add_argument("--run_gi_qtmd", type=int, default=int(os.environ.get("NUCLEON_TMD_RUN_GI_QTMD", 1)))
parser.add_argument("--run_pdf", type=int, default=int(os.environ.get("NUCLEON_TMD_RUN_PDF", 1)))
args, unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, io, phase, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import build_gi_qtmd_staple_links
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_pyquda
from pyquda_measurement_utils.fermion_bilinear_basis import gamma_stack
from pyquda_measurement_utils.io_corr import (
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    get_sample_log_tag,
    save_qTMD_proton_hdf5_noRoll,
)
from pyquda_measurement_utils.proton_qTMD_pyquda import my_gammas, proton_TMD
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print, srcLoc_distri_eq


def sync_backend_array(arr):
    stream = getattr(arr, "stream", None)
    if stream is not None:
        stream.synchronize()
    queue = getattr(arr, "sycl_queue", None)
    if queue is not None:
        queue.wait()


def gamma_stack_like(reference):
    return gamma_stack(reference).astype(reference.dtype, copy=False)


def contract_nucleon_operator_list(
    latt_info,
    measurement,
    gauge,
    prop_f,
    seq_down,
    seq_up,
    gamma_ls,
    phases,
    W_index_list,
    operator_kind,
):
    xp = _get_xp_from_array(prop_f.data)
    phases = _asarray_on_queue(phases, xp, prop_f.data)
    corr_down = []
    corr_up = []
    shifted_prop = prop_f.copy()
    staple_links = None

    if operator_kind == "GI_qTMD":
        mpi_print(latt_info, f"Build {len(W_index_list)} connected nucleon GI_qTMD staple transporters.")
        staple_links = build_gi_qtmd_staple_links(gauge, W_index_list)

    for iW, W_index in enumerate(W_index_list):
        mpi_print(latt_info, f"Contract nucleon {operator_kind} {iW + 1}/{len(W_index_list)} {W_index}")

        if operator_kind == "CG_qTMD":
            W_index_previous = [0, 0, 0, W_index[3]] if iW == 0 else W_index_list[iW - 1]
            if W_index[3] != W_index_previous[3]:
                shifted_prop = prop_f.copy()
                W_index_previous = [0, 0, 0, W_index[3]]
            shifted_prop = measurement.create_fw_prop_TMD_CG(shifted_prop, W_index, W_index_previous)
            current_prop = shifted_prop
        elif operator_kind == "CG_PDF":
            if W_index[1] in {0, -1}:
                shifted_prop = prop_f.copy()
                W_index_previous = [0, 0, 0, 0]
            else:
                W_index_previous = W_index_list[iW - 1]
            shifted_prop = measurement.create_fw_prop_TMD_CG(shifted_prop, W_index, W_index_previous)
            current_prop = shifted_prop
        elif operator_kind == "GI_PDF":
            if W_index[1] in {0, -1}:
                shifted_prop = prop_f.copy()
                W_index_previous = [0, 0, 0, 0]
            else:
                W_index_previous = W_index_list[iW - 1]
            shifted_prop = measurement.create_fw_prop_PDF_GI(gauge, shifted_prop, W_index, W_index_previous)
            current_prop = shifted_prop
        elif operator_kind == "GI_qTMD":
            current_prop = measurement.create_fw_prop_TMD_GI(gauge, prop_f, W_index, staple_links=staple_links)
        else:
            raise ValueError(f"Unsupported operator_kind {operator_kind!r}")

        temp_down = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            seq_down,
            gamma_ls,
            current_prop.data,
            optimize=True,
        )
        temp_down = xp.einsum("qwtzyx,pgwtzyx->pqgt", phases, temp_down, optimize=True)
        corr_down.append(core.gatherLattice(xp.asnumpy(temp_down), [3, -1, -1, -1]))
        sync_backend_array(temp_down)

        temp_up = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            seq_up,
            gamma_ls,
            current_prop.data,
            optimize=True,
        )
        temp_up = xp.einsum("qwtzyx,pgwtzyx->pqgt", phases, temp_up, optimize=True)
        corr_up.append(core.gatherLattice(xp.asnumpy(temp_up), [3, -1, -1, -1]))
        sync_backend_array(temp_up)

        if operator_kind == "GI_qTMD":
            del current_prop

    return np.asarray(corr_down), np.asarray(corr_up)


def roll_trim_bcast(corr, pos_t, tsep):
    if getMPIComm().Get_rank() == 0:
        corr = np.roll(corr, -pos_t, axis=-1)
        corr = corr[:, :, :, :, : tsep + 2]
    return getMPIComm().bcast(corr, root=0)


def save_nucleon_qtmd_by_gamma(
    latt_info,
    data_dir,
    lat_tag,
    conf,
    operator_tag,
    pos,
    sm_pf_pol_tag,
    corr_down,
    corr_up,
    qlist,
    W_index_list,
    tsep,
    pol_idx=0,
):
    if latt_info.mpi_rank != 0:
        return
    tasks = [(gidx, flavor) for gidx in range(len(my_gammas)) for flavor in ("D", "U")]

    for gidx, flavor in tasks:
        gm = my_gammas[gidx]
        tag = get_qTMD_file_tag(
            str(data_dir),
            lat_tag,
            conf,
            f"{operator_tag}.{flavor}.ex",
            pos,
            f"{sm_pf_pol_tag}.{gm}",
        )
        corr = corr_down if flavor == "D" else corr_up
        save_qTMD_proton_hdf5_noRoll(
            corr[:, pol_idx, :, gidx : gidx + 1, :],
            tag,
            [gm],
            qlist,
            W_index_list,
            tsep,
            latt_info,
        )


def save_nucleon_pdf(
    latt_info,
    data_dir,
    lat_tag,
    conf,
    operator_tag,
    pos,
    sm_pf_pol_tag,
    corr_down,
    corr_up,
    qlist,
    W_index_list,
    tsep,
    pol_idx=0,
):
    if latt_info.mpi_rank != 0:
        return
    for flavor in ("D", "U"):
        tag = get_qTMD_file_tag(
            str(data_dir),
            lat_tag,
            conf,
            f"{operator_tag}.{flavor}.ex",
            pos,
            sm_pf_pol_tag,
        )
        corr = corr_down if flavor == "D" else corr_up
        save_qTMD_proton_hdf5_noRoll(
            corr[:, pol_idx, :, :, :],
            tag,
            my_gammas,
            qlist,
            W_index_list,
            tsep,
            latt_info,
        )


# ============================================================
# Production parameters
# ============================================================
software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("NUCLEON_TMD_LAT_TAG", "S8T32")
conf = args.config_num
interpolator = args.interpolator
pol_list = [args.pol]
sm_tag = os.environ.get("NUCLEON_TMD_SM_TAG", f"1HYP_GSRC_W{args.width:g}_k0_{interpolator}")
run_cg_qtmd = bool(args.run_cg_qtmd)
run_gi_qtmd = bool(args.run_gi_qtmd)
run_pdf = bool(args.run_pdf)

q_range = range(-args.qmax, args.qmax + 1)
qext = [[x, y, 0, 0] for x in q_range for y in q_range]
qext_PDF = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
p_2pt = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
parameters = {
    "eta": [args.eta],
    "b_z": args.b_z,
    "b_T": args.b_T,
    "qext": qext,
    "qext_PDF": qext_PDF,
    "pf": [0, 0, 0, 0],
    "p_2pt": p_2pt,
    "boost_in": [0, 0, 0],
    "boost_out": [0, 0, 0],
    "width": args.width,
    "pol": pol_list,
    "t_insert": args.t_insert,
}
pf = parameters["pf"]
pf_tag = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{parameters['t_insert']}"
sm_pf_pol_tag = f"{sm_tag}.{pf_tag}.{pol_list[0]}"
measurement = proton_TMD(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "sample_log_qtmd").mkdir(parents=True, exist_ok=True)
    (data_dir / "c2pt").mkdir(parents=True, exist_ok=True)
    (data_dir / "qTMD").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--config_num {conf}")
    print(f"--mpi_geometry {args.mpi_geometry}")
    print(f"--run_cg_qtmd {int(run_cg_qtmd)}")
    print(f"--run_gi_qtmd {int(run_gi_qtmd)}")
    print(f"--run_pdf {int(run_pdf)}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
mass = float(os.environ.get("NUCLEON_TMD_MASS", 0.236))
csw_r = float(os.environ.get("NUCLEON_TMD_CSW", 1.0372))
csw_t = csw_r
tol = float(os.environ.get("NUCLEON_TMD_TOL", 1e-15))
maxiter = int(os.environ.get("NUCLEON_TMD_MAXITER", 300))
multigrid = [[max(1, latt_info.global_size[0] // 1), max(1, latt_info.global_size[1] // 1), max(1, latt_info.global_size[2] // 2), max(1, latt_info.global_size[3] // 8)]]

dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)
gamma_ls = gamma_stack_like(gauge.data)

L = latt_info.global_size
src_shift = np.array([0, 0, 0, 0])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)[: args.num_src]

sample_log_file = data_dir / "sample_log_qtmd" / f"{conf}_{sm_tag}_{pf_tag}"
if latt_info.mpi_rank == 0:
    sample_log_file.touch(exist_ok=True)

for pos in src_positions:
    t0_pos = time.time()
    sample_log_tag = get_sample_log_tag(str(conf), pos, f"{sm_tag}_{pf_tag}")
    mpi_print(latt_info, f"START: {sample_log_tag}")

    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    mpi_print(latt_info, f"TIME PyQUDA: Generating boosted source {time.time() - t0}s")

    t0 = time.time()
    prop_fw = core.invertPropagator(dirac, srcDp, 1, 0)
    mpi_print(latt_info, f"TIME PyQUDA: Forward propagator inversion {time.time() - t0}s")

    t0 = time.time()
    c2_tag = get_c2pt_file_tag(str(data_dir), lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    measurement.contract_2pt_TMD(latt_info, prop_fw, phases_2pt, c2_tag, interpolator)
    mpi_print(latt_info, f"TIME PyQUDA: Nucleon 2pt contraction {time.time() - t0}s")

    t0 = time.time()
    seq_down = create_bw_seq_pyquda(
        dirac,
        prop_fw,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        2,
        interpolator,
    )
    seq_up = create_bw_seq_pyquda(
        dirac,
        prop_fw,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        1,
        interpolator,
    )
    mpi_print(latt_info, f"TIME PyQUDA: Nucleon sequential propagators {time.time() - t0}s")

    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_TMD = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)
    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phases_PDF = MomentumPhase(latt_info).getPhases(qext_pdf_xyz, x0=pos)

    W_index_list_CG_dir0, W_index_list_CG_dir1 = measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    W_index_list_GI_dir0, W_index_list_GI_dir1 = measurement.create_TMD_Wilsonline_index_list_GI()
    W_index_list_GI = W_index_list_GI_dir0 + W_index_list_GI_dir1
    W_index_list_PDF = measurement.create_PDF_Wilsonline_index_list()

    if run_cg_qtmd:
        t0 = time.time()
        corr_down, corr_up = contract_nucleon_operator_list(
            latt_info,
            measurement,
            gauge,
            prop_fw,
            seq_down,
            seq_up,
            gamma_ls,
            phases_TMD,
            W_index_list_CG,
            "CG_qTMD",
        )
        mpi_print(latt_info, f"contract_CG_qTMD over: {np.shape(corr_down)} {time.time() - t0}s")
        corr_down = roll_trim_bcast(corr_down, pos[3], parameters["t_insert"])
        corr_up = roll_trim_bcast(corr_up, pos[3], parameters["t_insert"])
        save_nucleon_qtmd_by_gamma(latt_info, data_dir, lat_tag, conf, "CG", pos, sm_pf_pol_tag, corr_down, corr_up, parameters["qext"], W_index_list_CG, parameters["t_insert"])

    if run_gi_qtmd:
        t0 = time.time()
        corr_down, corr_up = contract_nucleon_operator_list(
            latt_info,
            measurement,
            gauge,
            prop_fw,
            seq_down,
            seq_up,
            gamma_ls,
            phases_TMD,
            W_index_list_GI,
            "GI_qTMD",
        )
        mpi_print(latt_info, f"contract_GI_qTMD over: {np.shape(corr_down)} {time.time() - t0}s")
        corr_down = roll_trim_bcast(corr_down, pos[3], parameters["t_insert"])
        corr_up = roll_trim_bcast(corr_up, pos[3], parameters["t_insert"])
        save_nucleon_qtmd_by_gamma(latt_info, data_dir, lat_tag, conf, "GI_qTMD", pos, sm_pf_pol_tag, corr_down, corr_up, parameters["qext"], W_index_list_GI, parameters["t_insert"])

    if run_pdf:
        for operator_kind, operator_tag in [("GI_PDF", "GI_PDF"), ("CG_PDF", "CG_PDF")]:
            t0 = time.time()
            corr_down, corr_up = contract_nucleon_operator_list(
                latt_info,
                measurement,
                gauge,
                prop_fw,
                seq_down,
                seq_up,
                gamma_ls,
                phases_PDF,
                W_index_list_PDF,
                operator_kind,
            )
            mpi_print(latt_info, f"contract_{operator_tag} over: {np.shape(corr_down)} {time.time() - t0}s")
            corr_down = roll_trim_bcast(corr_down, pos[3], parameters["t_insert"])
            corr_up = roll_trim_bcast(corr_up, pos[3], parameters["t_insert"])
            save_nucleon_pdf(latt_info, data_dir, lat_tag, conf, operator_tag, pos, sm_pf_pol_tag, corr_down, corr_up, parameters["qext_PDF"], W_index_list_PDF, parameters["t_insert"])

    sync_backend_array(prop_fw.data)

    if latt_info.mpi_rank == 0:
        with sample_log_file.open("a+") as f:
            f.write(sample_log_tag + "\n")
    mpi_print(latt_info, f"DONE: {sample_log_tag} total {time.time() - t0_pos}s")
