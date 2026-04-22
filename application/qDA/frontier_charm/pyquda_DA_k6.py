"""
Charmonium DA measurement with PyQUDA.

Physics summary
---------------
This script measures the bT = 0 special case of the pion/charmonium TMDWF
correlator, i.e. the DA correlator with a straight Wilson line in the z
direction. For each sink gamma Gamma_g and source gamma Gamma_src, the basic
contraction is written in the common form

    C^(g, src)(q, t; z) =
        sum_x exp(i q . x)
        Tr_{c,s}[ gamma5 S_b(x)^\dagger gamma5 Gamma_g F(x; z) Gamma_src ] .

Here:
- In the CG block, F(x; z) means a shifted forward propagator S_f(x + z zhat)
  without explicit gauge links, i.e. a fixed-gauge correlator.
- In the GI block, F(x; z) means a gauge-covariantly shifted propagator
  W(x, x + z zhat) S_f(x + z zhat), where the straight Wilson line is built
  into the covariant shift.
- The sink gamma runs over the 16-element gamma basis in `gammalist`.
- The source gamma is chosen from `da_src_gammalist`.
- The 2pt correlator is also measured with an independent source convention
  controlled by `src_2pt_mode`.

Implementation goals
--------------------
1. Keep the DA workflow explicit: 2pt -> CG DA -> GI DA.
2. Share the common contraction/save logic between CG and GI.
3. Save one file per block/source choice, with all sink gamma channels inside.
"""

import time
import numpy as np
import cupy as cp

from pyquda import init, getMPIComm
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement, my_pyquda_gammas
from pyquda_measurement_utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMDWF_file_tag, save_qTMDWF_hdf5_noRoll
from pyquda_measurement_utils.tools import srcLoc_distri_eq, mpi_print


import argparse

# ============================================================================
# CLI arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# ============================================================================
# User-facing run configuration
# ============================================================================
data_dir="/lustre/orion/nph158/proj-shared/xgao/l64c64a040/charmonium_DA/data" # NOTE
lat_tag = "l64c64a040" # NOTE

# Main smearing knobs. Keep these here so the tag and the actual parameters
# always stay synchronized.
smearing_width = 1.0   # corresponds to W10 in the tag
smearing_boost_k = 6   # corresponds to k6 in the tag

sm_tag = f"1HYP_GSRC_W{int(round(smearing_width * 10))}_k{smearing_boost_k}" # NOTE
# Source-gamma construction rule used only in `contract_2pt_pion(...)`.
# It sets how the 2pt source Dirac structure is built from the sink gamma:
# `fixed_g5`, `same_as_sink`, or `dagger_of_sink`.
src_2pt_mode = "dagger_of_sink" # NOTE

# Source-gamma channels measured in the DA blocks (CG / GI).
# This is a list of source choices, e.g. `["5"]`, `["5", "X", "T"]`, or `gammalist`;
# for each chosen source, the sink still runs over the full `gammalist`.
da_src_gammalist = ["5", "X", "T"] # NOTE


init(mpi_geometry, enable_mps=True)
G5 = gamma.gamma(15)

# ============================================================================
# Physics / measurement setup
# ============================================================================
parameters = {
    # Longitudinal staple extent used in the more general TMDWF setup.
    # It is kept here for compatibility with the shared measurement class;
    # in the present bT = 0 DA workflow it does not play an explicit role.
    "eta" : [0],
    # Maximum transverse separation. Setting b_T = 0 selects the DA limit.
    "b_T": 0,
    # Maximum longitudinal separation z used for the DA Wilson line.
    "b_z" : 32,
    # Minimum longitudinal momentum p_z included in the momentum projection.
    "pzmin" : 4,
    # Upper bound of the longitudinal momentum range (Python range end, excluded).
    "pzmax" : 11,
    # Width parameter of the boosted Gaussian smearing.
    "width" : smearing_width,
    # Momentum used in boosted smearing for the forward/source-side propagator.
    "pos_boost" : [0,0,smearing_boost_k],
    # Momentum used in boosted smearing for the backward/sink-side propagator.
    "neg_boost" : [0,0,-smearing_boost_k],
    # Retained for interface compatibility; this script does not save propagators.
    "save_propagators" : False
}
Measurement = pion_TMDWF_measurement(parameters)
xp = cp
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
momentum_list = [[0, 0, p, 0] for p in range(parameters["pzmin"], parameters["pzmax"])]


# ============================================================================
# Small helpers
# ============================================================================
# Synchronize CUDA work only where timing boundaries matter.
def sync_cuda():
    cp.cuda.runtime.deviceSynchronize()


# Build the full 16-gamma sink basis in the PyQUDA ordering used by this script.
def build_pyquda_gamma_ls():
    first_gamma = my_pyquda_gammas[0]
    gamma_ls = xp.empty((len(my_pyquda_gammas),) + first_gamma.shape, dtype=first_gamma.dtype)
    for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
        gamma_ls[gamma_idx] = gamma_pyq
    return gamma_ls


# Validate the requested DA source gamma names and return their matrix list.
def build_da_src_gamma_ls(pyquda_gamma_ls):
    gamma_index = {name: idx for idx, name in enumerate(gammalist)}
    invalid_src = [name for name in da_src_gammalist if name not in gamma_index]
    if invalid_src:
        raise ValueError(f"Invalid da_src_gammalist entries: {invalid_src}")
    return [(name, pyquda_gamma_ls[gamma_index[name]]) for name in da_src_gammalist]


# Save all sink-gamma channels for each chosen DA source gamma into one file.
def save_da_correlators(collect_by_src, pos, W_index_list, block_tag):
    t0 = time.time()
    for src_name, data_collect in collect_by_src:
        if latt_info.mpi_rank == 0:
            # Shift the source time to t=0 before writing the correlator.
            data_collect = np.roll(data_collect, -pos[3], axis=-1)
            output_tag = get_qTMDWF_file_tag(
                data_dir,
                lat_tag,
                conf,
                "ex",
                pos,
                f"{sm_tag}.{block_tag}.src{src_name}",
            )
            save_qTMDWF_hdf5_noRoll(
                data_collect,
                output_tag,
                gammalist,
                momentum_list,
                W_index_list,
            )
    mpi_print(latt_info, f"TIME: save {block_tag} DAs {time.time() - t0}")


# Contract one DA block (CG or GI) and save the resulting correlators.
def contract_da_block(propag_f, propag_b, phases_2pt, W_index_list, block_tag, shift_prop):
    mpi_print(latt_info, f"Contraction: Start DA: {block_tag.upper()} no links (bT=0 TMDWF)")
    # Common sink-side gamma structure for all source gamma choices.
    G5_gamma = xp.einsum("ki,gim->gkm", G5, pyquda_gamma_ls)
    collect_by_src = []

    for src_name, Gsrc in da_src_gamma_ls:
        collect_src = []
        tmd_backward_prop = propag_b.copy()
        # Source gamma is combined with gamma5 once and reused for all Wilson-line lengths.
        Gsrc_G5 = xp.einsum("ij,jk->ik", Gsrc, G5)

        for iW, WL_indices in enumerate(W_index_list):
            t0 = time.time()
            mpi_print(
                latt_info,
                f"TIME PyQUDA: contract {block_tag.upper()} src{src_name} {iW+1}/{len(W_index_list)} {WL_indices}",
            )

            if iW == 0:
                WL_indices_previous = [0, 0, 0, 0]
            else:
                WL_indices_previous = W_index_list[iW - 1]

            # Update the backward line to the next DA Wilson-line geometry.
            tmd_backward_prop = shift_prop(tmd_backward_prop, WL_indices, WL_indices_previous)
            mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}")

            t0 = time.time()
            # Spin/color contraction at fixed Wilson-line length.
            temp = xp.einsum(
                "ij,wtzyxkjba,wtzyxliba->wtzyxkl",
                Gsrc_G5,
                tmd_backward_prop.data.conj(),
                propag_f.data,
            )
            # Insert the sink gamma basis and project to all sink channels.
            temp = xp.einsum("wtzyxkl,gkl->gwtzyx", temp, G5_gamma)
            # Fourier transform to the requested longitudinal momenta.
            temp = xp.einsum("qwtzyx,gwtzyx->qgt", phases_2pt, temp)
            # Gather the correlator from the distributed lattice onto the root rank.
            temp2 = core.gatherLattice(xp.asnumpy(temp), [2, -1, -1, -1])
            collect_src.append(temp2)
            mpi_print(latt_info, f"TIME PyQUDA: contract {block_tag.upper()} src{src_name} {time.time() - t0}")

            del temp, temp2

        del tmd_backward_prop
        collect_by_src.append((src_name, collect_src))

    save_da_correlators(collect_by_src, pos, W_index_list, block_tag)
    mpi_print(latt_info, f"Contraction: Done DA: {block_tag.upper()} no links (bT=0 TMDWF)")


# ============================================================================
# Lattice / inverter / gamma preparation
# ============================================================================
Ls = 64
Lt = 64
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = 0.167
csw_r = 1.02868 
csw_t = 1.02868 
multigrid = [[8, 8, 4, 4]]
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

gauge = io.readNERSCGauge(f"/lustre/orion/nph174/proj-shared/ensembles/l64c64a040/l6464f21b7825m00082m0164a_fixed_GLU/l6464f21b7825m00082m0164a.{conf}.coulomb.1e-14")
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

dirac = core.getClover(latt_info, mass, 1e-15, 10000, xi_0, csw_r, csw_t, multigrid)

pyquda_gamma_ls = build_pyquda_gamma_ls()
da_src_gamma_ls = build_da_src_gamma_ls(pyquda_gamma_ls)

src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)
src_production = src_positions[0:10]

# ============================================================================
# Per-run bookkeeping
# ============================================================================
sample_log_file = data_dir + f"/sample_log/TMDWF_{sm_tag}_{conf}"
if latt_info.mpi_rank == 0:
    open(sample_log_file, "a+").close()
time.sleep(2)


# ============================================================================
# Main source loop
# ============================================================================
for ipos, pos in enumerate(src_production):
    sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
    mpi_print(latt_info, f"Contraction START: {sample_log_tag}")
    with open(sample_log_file, "a+") as f:
        f.seek(0)
        if sample_log_tag in f.read():
            mpi_print(latt_info, f"Contraction SKIP: {sample_log_tag}")
            continue

    # ------------------------------------------------------------------------
    # Source construction and boosted smearing
    # ------------------------------------------------------------------------
    sync_cuda()
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["pos_boost"])
    srcDm = boosted_smearing(srcD, w=parameters["width"], boost=parameters["neg_boost"])
    sync_cuda()
    mpi_print(latt_info, f"TIME Pyquda: Generatring boosted src {time.time() - t0}")

    # ------------------------------------------------------------------------
    # Forward / backward propagator inversions
    # ------------------------------------------------------------------------
    t0 = time.time()
    dirac.loadGauge(gauge) #TODO: debug
    propag_f = core.invertPropagator(dirac, srcDp, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    propag_b = core.invertPropagator(dirac, srcDm, 1, 0)
    sync_cuda()
    mpi_print(latt_info, f"TIME: Pyquda inversion * 2 {time.time() - t0}")

    # ------------------------------------------------------------------------
    # Two-point correlator
    # ------------------------------------------------------------------------
    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    Measurement.contract_2pt_pion(latt_info, propag_f, propag_b, phases_2pt, tag, src_mode=src_2pt_mode)

    sync_cuda()
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}")

    # ------------------------------------------------------------------------
    # DA Wilson-line geometry
    # ------------------------------------------------------------------------
    W_index_list = Measurement.create_PDF_Wilsonline_index_list()

    # ------------------------------------------------------------------------
    # CG DA block
    # ------------------------------------------------------------------------
    contract_da_block(
        propag_f,
        propag_b,
        phases_2pt,
        W_index_list,
        "CG",
        Measurement.create_fw_prop_TMD_CG,
    )

    # ------------------------------------------------------------------------
    # GI DA block
    # ------------------------------------------------------------------------
    contract_da_block(
        propag_f,
        propag_b,
        phases_2pt,
        W_index_list,
        "GI",
        Measurement.create_fw_prop_TMD_GI,
    )

    # ------------------------------------------------------------------------
    # Bookkeeping for completed source positions
    # ------------------------------------------------------------------------
    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag+"\n")

    mpi_print(latt_info, f"DONE: {sample_log_tag}")
