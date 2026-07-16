'''
v2 means no mesonAllSinkTwoPoint, for non-CUDA environments
'''

# ============================================================
# Imports
# ============================================================
import time
import os
import argparse

import numpy as np
import dpnp as dnp

from pyquda import init, getMPIComm


# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--stream", type=str, default="c", help="Ensemble stream")
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()

stream = args.stream
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]


# ============================================================
# Initialize QUDA backend
# ============================================================
init(
    mpi_geometry,
    enable_mps=True,
    backend="dpnp",
    backend_target="sycl",
    resource_path=".cache",
)

from pyquda_utils import core, phase, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda.field import LatticeGauge

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_pyquda
from pyquda_measurement_utils.fermion_bilinear_basis import gamma_stack
from pyquda_measurement_utils.proton_qTMD_pyquda import proton_TMD
from pyquda_measurement_utils.io_corr import (
    get_sample_log_tag,
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    save_qTMD_proton_hdf5_noRoll,
)
from pyquda_measurement_utils.tools import srcLoc_distri_eq, mpi_print


# ============================================================
# Global parameters
# ============================================================
data_dir = f"/lus/flare/projects/StructNGB/xgao/run/l80c80a050/nucleon_TMD_pyquda/data_{stream}"  # NOTE
lat_tag = "l80c80a050"  # NOTE
interpolation = "5"  # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W130_k0_" + interpolation  # NOTE

parameters = {
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 24,
    "b_T": 24,
    "qext": [[x, y, z, 0] for x in [-2, -1, 0, 1, 2] for y in [-2, -1, 0, 1, 2] for z in [0]],
    "qext_PDF": [[x, y, z, 0] for x in [-2, -1, 0, 1, 2] for y in [-2, -1, 0, 1, 2] for z in [-2, -1, 0, 1, 2]],
    "pf": [0, 0, 0, 0],
    "p_2pt": [[x, y, z, 0] for x in [-2, -1, 0, 1, 2] for y in [-2, -1, 0, 1, 2] for z in [-2, -1, 0, 1, 2]],
    "boost_in": [0, 0, 0],
    "boost_out": [0, 0, 0],
    "width": 13.0,
    "pol": ["PpUnpol"],
    "t_insert": 9,  # time separation for TMD
}

pf = parameters["pf"]
pf_tag = "PX" + str(pf[0]) + "PY" + str(pf[1]) + "PZ" + str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

Measurement = proton_TMD(parameters)
xp = dnp


# ============================================================
# Utilities
# ============================================================
def sync_usm(x):
    q = getattr(x, "sycl_queue", None)
    if q is not None:
        q.wait()


# ============================================================
# Lattice / inverter setup
# ============================================================
Ls = 80
Lt = 80
L = [Ls, Ls, Ls, Lt]

xi_0, nu = 1.0, 1.0
mass = -0.0386
csw_r = 1.03094
csw_t = 1.03094
multigrid = [[5, 4, 5, 4]]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)


# ============================================================
# Phase sanity checks
# ============================================================
qext_xyz_check = [[v[0], v[1], v[2]] for v in parameters["qext"]]
phases_check = phase.MomentumPhase(latt_info).getPhases(qext_xyz_check, [0, 0, 0, 0])
p_sum_local = xp.sum(phases_check[0])
p_sum_local_np = xp.asnumpy(p_sum_local)
p_sum_global = getMPIComm().allreduce(p_sum_local_np)
mpi_print(latt_info, f"DEBUG: Global Phase Sum (p={qext_xyz_check[0]}): {p_sum_global}")

phases_check_0 = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0, 0]], [0, 0, 0, 0])
p_sum_local_0 = xp.sum(phases_check_0[0])
p_sum_local_np_0 = xp.asnumpy(p_sum_local_0)
p_sum_global_0 = getMPIComm().allreduce(p_sum_local_np_0)
mpi_print(
    latt_info,
    f"DEBUG: Global Phase Sum (p=[0,0,0,0]): {p_sum_global_0} (Expected {latt_info.global_volume})",
)

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")


# ============================================================
# Dirac operator and gauge field
# ============================================================
dirac = core.getDirac(latt_info, mass, 1e-10, 5000, xi_0, csw_r, csw_t, multigrid)

gauge = io.readNERSCGauge(
    f"/lus/flare/projects/StructNGB/xgao/ensembles/s8080b7596/gauge_fixed/{stream}/l8080f21b7596m00101m0202{stream}.coulomb.1e-14.220",
    checksum=False,
    link_trace=False,
    plaquette=False,
)
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")


# ============================================================
# Prepare gamma matrices on device
# ============================================================
pyquda_gamma_ls = gamma_stack(gauge.data).astype(gauge.data.dtype, copy=False)


# ============================================================
# Source positions
# ============================================================
src_shift = np.array([0, 0, 0, 0]) + np.array([7, 11, 13, 23])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)
src_production = src_positions[0:2]  # NOTE: take the number of sources needed for this project


# ============================================================
# Sample log file
# ============================================================
sample_log_file = data_dir + "/sample_log_qtmd/" + str(conf) + '_' + sm_tag + "_" + pf_tag
if latt_info.mpi_rank == 0:
    f = open(sample_log_file, "a+")
    f.close()


# ============================================================
# Start measurements over source positions
# ============================================================
for ipos, pos in enumerate(src_production):
    t0_pos = time.time()
    sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
    mpi_print(latt_info, f"START: {sample_log_tag}")

    with open(sample_log_file, "a+") as f:
        f.seek(0)
        if sample_log_tag in f.read():
            mpi_print(latt_info, f"SKIP: {sample_log_tag}")
            # continue  # NOTE: comment this out for test otherwise it will skip all the sources that are already done

    # ========================================================
    # Forward source and forward propagator
    # ========================================================
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    mpi_print(latt_info, f"TIME Pyquda: Generatring boosted src {time.time() - t0}s")

    t0 = time.time()
    dirac.loadGauge(gauge)
    propag = core.invertPropagator(dirac, srcDp, 1, 0)
    mpi_print(latt_info, f"TIME Pyquda: Forward propagator inversion {time.time() - t0}s")

    # ========================================================
    # 2pt contraction
    # ========================================================
    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)

    Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, tag, interpolation)
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}s")

    # ========================================================
    # Sequential backward propagators
    # ========================================================
    t0 = time.time()
    sequential_bw_prop_down_pyq = create_bw_seq_pyquda(
        dirac,
        propag,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        2,
        interpolation,
    )
    sequential_bw_prop_up_pyq = create_bw_seq_pyquda(
        dirac,
        propag,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        1,
        interpolation,
    )
    mpi_print(latt_info, f"TIME Pyquda: Backward propagator through sequential source for U and D {time.time() - t0}s")

    # ========================================================
    # Momentum phases for TMD / PDF
    # ========================================================
    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_TMD = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)

    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phase_PDF = MomentumPhase(latt_info).getPhases(qext_pdf_xyz, pos)

    #
    # ######################################################################
    # ## CG TMD PART
    # ######################################################################
    W_index_list_CG_dir0, W_index_list_CG_dir1 = Measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1

    mpi_print(latt_info, f"DEBUG contract_TMD loop: CG no links")

    t0_contract = time.time()
    proton_TMDs_down = []  # [WL_indices][pol][qext][gammalist][tau]
    proton_TMDs_up = []

    # --------------------------------------------------------
    # +X direction
    # --------------------------------------------------------
    tmd_forward_prop_dir0 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir0):
        t0 = time.time()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")

        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir0[iW - 1]

        tmd_forward_prop_dir0 = Measurement.create_fw_prop_TMD_CG(
            tmd_forward_prop_dir0,
            WL_indices,
            WL_indices_previous,
        )
        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}s")

        temp_down = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_down_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_dir0.data,
            optimize=True,
        )
        temp_down = xp.einsum("qwtzyx, pgwtzyx -> pqgt", phases_TMD, temp_down)
        proton_TMDs_down += [core.gatherLattice(xp.asnumpy(temp_down), [3, -1, -1, -1])]
        sync_usm(temp_down)

        temp_up = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_up_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_dir0.data,
            optimize=True,
        )
        temp_up = xp.einsum("qwtzyx, pgwtzyx -> pqgt", phases_TMD, temp_up)
        proton_TMDs_up += [core.gatherLattice(xp.asnumpy(temp_up), [3, -1, -1, -1])]
        sync_usm(temp_up)


        mpi_print(latt_info, f"TIME PyQUDA: contract TMD for U and D {time.time() - t0}s")
    del tmd_forward_prop_dir0

    # --------------------------------------------------------
    # +Y direction
    # --------------------------------------------------------
    tmd_forward_prop_dir1 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir1):
        t0 = time.time()
        mpi_print(
            latt_info,
            f"TIME PyQUDA: contract TMD {iW+1+len(W_index_list_CG_dir0)}/{len(W_index_list_CG)} {WL_indices}",
        )

        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir1[iW - 1]

        tmd_forward_prop_dir1 = Measurement.create_fw_prop_TMD_CG(
            tmd_forward_prop_dir1,
            WL_indices,
            WL_indices_previous,
        )
        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}s")

        temp_down = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_down_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_dir1.data,
            optimize=True,
        )
        proton_TMDs_down += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phases_TMD, temp_down)),
                [3, -1, -1, -1],
            )
        ]
        sync_usm(temp_down)

        temp_up = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_up_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_dir1.data,
            optimize=True,
        )
        proton_TMDs_up += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phases_TMD, temp_up)),
                [3, -1, -1, -1],
            )
        ]
        sync_usm(temp_up)

        mpi_print(latt_info, f"TIME PyQUDA: contract TMD for U and D {time.time() - t0}s")
    del tmd_forward_prop_dir1

    proton_TMDs_down = np.array(proton_TMDs_down)
    proton_TMDs_up = np.array(proton_TMDs_up)
    mpi_print(latt_info, f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)} {time.time() - t0_contract}s")

    # ========================================================
    # Save CG TMD correlators
    # ========================================================
    for i, pol in enumerate(parameters["pol"]):
        t0 = time.time()

        if latt_info.mpi_rank == 0 and i == 0:
            proton_TMDs_down = np.roll(proton_TMDs_down, -pos[3], axis=-1)
            proton_TMDs_up = np.roll(proton_TMDs_up, -pos[3], axis=-1)
            proton_TMDs_down = proton_TMDs_down[:, :, :, :, : parameters["t_insert"] + 2]
            proton_TMDs_up = proton_TMDs_up[:, :, :, :, : parameters["t_insert"] + 2]

        proton_TMDs_down = getMPIComm().bcast(proton_TMDs_down, root=0)
        proton_TMDs_up = getMPIComm().bcast(proton_TMDs_up, root=0)

        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append((gidx, 'D'))
            tasks.append((gidx, 'U'))

        rank = latt_info.mpi_rank
        if rank < len(tasks):
            gidx, flavor = tasks[rank]
            gm = gammalist[gidx]
            tag = get_qTMD_file_tag(
                data_dir,
                lat_tag,
                conf,
                f"CG.{flavor}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{pol}.{gm}",
            )
            print(f"DEBUG: rank {rank}, {tag}")
            data = proton_TMDs_down[:, i, :, gidx:gidx + 1, :] if flavor == 'D' else proton_TMDs_up[:, i, :, gidx:gidx + 1, :]
            save_qTMD_proton_hdf5_noRoll(
                data,
                tag,
                [gm],
                parameters["qext"],
                W_index_list_CG,
                parameters["t_insert"],
                latt_info,
            )

        mpi_print(latt_info, f"TIME: save TMDs for {pol} {time.time() - t0}s")
    mpi_print(latt_info, f"contract_TMD DONE: CG no links")

    #
    # ######################################################################
    # ## CG PDF PART
    # ######################################################################
    W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list()

    mpi_print(latt_info, f"contract_PDF loop: GI with links")
    t0_contract = time.time()
    proton_PDFs_down = []  # [WL_indices][pol][qext][gammalist][tau]
    proton_PDFs_up = []

    for iW, WL_indices in enumerate(W_index_list_PDF):
        t0 = time.time()

        if WL_indices[1] == 0:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] > 0:
            WL_indices_previous = W_index_list_PDF[iW - 1]
        elif WL_indices[1] == -1:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] < -1:
            WL_indices_previous = W_index_list_PDF[iW - 1]

        tmd_forward_prop_pyq = Measurement.create_fw_prop_PDF_GI(
            gauge,
            tmd_forward_prop_pyq,
            WL_indices,
            WL_indices_previous,
        )

        temp_down = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_down_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_pyq.data,
            optimize=True,
        )
        proton_PDFs_down += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phase_PDF, temp_down)),
                [3, -1, -1, -1],
            )
        ]

        temp_up = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_up_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_pyq.data,
            optimize=True,
        )
        proton_PDFs_up += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phase_PDF, temp_up)),
                [3, -1, -1, -1],
            )
        ]

        mpi_print(latt_info, f"TIME PyQUDA: contract GI PDF for U and D {time.time() - t0}s")
    del tmd_forward_prop_pyq

    proton_PDFs_down = np.array(proton_PDFs_down)
    proton_PDFs_up = np.array(proton_PDFs_up)

    mpi_print(latt_info, f"contract_GI_PDF over: proton_PDFs.shape {xp.shape(proton_PDFs_down)} {time.time() - t0}s")
    mpi_print(latt_info, f"TIME PyQUDA: contract GI PDF for U and D {time.time() - t0_contract}s")

    # ========================================================
    # Save GI PDF correlators
    # ========================================================
    for i, pol in enumerate(parameters["pol"]):
        t0 = time.time()

        if latt_info.mpi_rank == 0 and i == 0:
            proton_PDFs_down = np.roll(proton_PDFs_down, -pos[3], axis=-1)
            proton_PDFs_up = np.roll(proton_PDFs_up, -pos[3], axis=-1)
            proton_PDFs_down = proton_PDFs_down[:, :, :, :, : parameters["t_insert"] + 2]
            proton_PDFs_up = proton_PDFs_up[:, :, :, :, : parameters["t_insert"] + 2]

        proton_PDFs_down = getMPIComm().bcast(proton_PDFs_down, root=0)
        proton_PDFs_up = getMPIComm().bcast(proton_PDFs_up, root=0)

        tasks = ['D', 'U']
        if latt_info.mpi_rank < len(tasks):
            flavor = tasks[latt_info.mpi_rank]
            tag = get_qTMD_file_tag(
                data_dir,
                lat_tag,
                conf,
                f"GI_PDF.{flavor}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{pol}",
            )
            data = proton_PDFs_down[:, i, :, :, :] if flavor == 'D' else proton_PDFs_up[:, i, :, :, :]
            save_qTMD_proton_hdf5_noRoll(
                data,
                tag,
                gammalist,
                parameters["qext_PDF"],
                W_index_list_PDF,
                parameters["t_insert"],
                latt_info,
            )

        mpi_print(latt_info, f"TIME: save PDFs for {pol} {time.time() - t0}s")
    mpi_print(latt_info, f"contract_PDF DONE: GI with links")

    #
    # ######################################################################
    # ## CG PDF PART
    # ######################################################################
    W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list()

    mpi_print(latt_info, f"contract_PDF loop: CG with links")
    t0_contract = time.time()
    proton_PDFs_down = []  # [WL_indices][pol][qext][gammalist][tau]
    proton_PDFs_up = []

    for iW, WL_indices in enumerate(W_index_list_PDF):
        t0 = time.time()

        if WL_indices[1] == 0:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] > 0:
            WL_indices_previous = W_index_list_PDF[iW - 1]
        elif WL_indices[1] == -1:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] < -1:
            WL_indices_previous = W_index_list_PDF[iW - 1]

        tmd_forward_prop_pyq = Measurement.create_fw_prop_TMD_CG(
            tmd_forward_prop_pyq,
            WL_indices,
            WL_indices_previous,
        )

        temp_down = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_down_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_pyq.data,
            optimize=True,
        )
        proton_PDFs_down += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phase_PDF, temp_down)),
                [3, -1, -1, -1],
            )
        ]

        temp_up = xp.einsum(
            "pwtzyxjicf,gim,wtzyxmjfc->pgwtzyx",
            sequential_bw_prop_up_pyq,
            pyquda_gamma_ls,
            tmd_forward_prop_pyq.data,
            optimize=True,
        )
        proton_PDFs_up += [
            core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phase_PDF, temp_up)),
                [3, -1, -1, -1],
            )
        ]

        mpi_print(latt_info, f"TIME PyQUDA: contract CG PDF for U and D {time.time() - t0}s")
    del tmd_forward_prop_pyq

    proton_PDFs_down = np.array(proton_PDFs_down)
    proton_PDFs_up = np.array(proton_PDFs_up)

    mpi_print(latt_info, f"contract_CG_PDF over: proton_PDFs.shape {xp.shape(proton_PDFs_down)} {time.time() - t0}s")
    mpi_print(latt_info, f"TIME PyQUDA: contract CG PDF for U and D {time.time() - t0_contract}s")

    # ========================================================
    # Save CG PDF correlators
    # ========================================================
    for i, pol in enumerate(parameters["pol"]):
        t0 = time.time()

        if latt_info.mpi_rank == 0 and i == 0:
            proton_PDFs_down = np.roll(proton_PDFs_down, -pos[3], axis=-1)
            proton_PDFs_up = np.roll(proton_PDFs_up, -pos[3], axis=-1)
            proton_PDFs_down = proton_PDFs_down[:, :, :, :, : parameters["t_insert"] + 2]
            proton_PDFs_up = proton_PDFs_up[:, :, :, :, : parameters["t_insert"] + 2]

        proton_PDFs_down = getMPIComm().bcast(proton_PDFs_down, root=0)
        proton_PDFs_up = getMPIComm().bcast(proton_PDFs_up, root=0)

        tasks = ['D', 'U']
        if latt_info.mpi_rank < len(tasks):
            flavor = tasks[latt_info.mpi_rank]
            tag = get_qTMD_file_tag(
                data_dir,
                lat_tag,
                conf,
                f"CG_PDF.{flavor}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{pol}",
            )
            data = proton_PDFs_down[:, i, :, :, :] if flavor == 'D' else proton_PDFs_up[:, i, :, :, :]
            save_qTMD_proton_hdf5_noRoll(
                data,
                tag,
                gammalist,
                parameters["qext_PDF"],
                W_index_list_PDF,
                parameters["t_insert"],
                latt_info,
            )

        mpi_print(latt_info, f"TIME: save PDFs for {pol} {time.time() - t0}s")
    mpi_print(latt_info, f"contract_PDF DONE: CG with links")

    # ========================================================
    # Finish this source position
    # ========================================================
    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag + "\n")

    mpi_print(latt_info, f"DONE {sample_log_tag}: {time.time() - t0_pos}s")
