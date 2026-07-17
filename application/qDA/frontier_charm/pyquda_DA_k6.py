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
        Tr_{c,s}[ gamma5 B_-(x;z)^\dagger gamma5 Gamma_g
                  S_+(x) Gamma_src ] .

Here:
- In the CG block, B_-(x;z) is the shifted negative-boost backward active line
  without explicit gauge links.
- In the GI block, B_-(x;z) is gauge-covariantly transported with a straight
  Wilson line. The positive-boost forward spectator S_+ remains fixed.
- The sink gamma runs over the 16-element gamma basis in `gammalist`.
- The source gamma is chosen from `da_src_gammalist`.
- The local 2pt correlator uses the paired source convention
  `gamma5 * Gamma_sink^dagger * gamma5`.

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
from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMDWF_file_tag, save_qTMDWF_hdf5_noRoll
from pyquda_measurement_utils.tools import (
    append_sample_log_entry,
    mpi_print,
    read_sample_log_entries,
    srcLoc_distri_eq,
)


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
# Source-gamma channels measured in the DA blocks (CG / GI).
# This is a list of source choices, e.g. `["5"]`, `["5", "X", "T"]`, or `gammalist`;
# for each chosen source, the sink still runs over the full `gammalist`.
da_src_gammalist = ["5", "X", "T"] # NOTE


init(mpi_geometry, enable_mps=True)

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
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
momentum_list = [[0, 0, p, 0] for p in range(parameters["pzmin"], parameters["pzmax"])]


# ============================================================================
# Small helpers
# ============================================================================
# Synchronize CUDA work only where timing boundaries matter.
def sync_cuda():
    cp.cuda.runtime.deviceSynchronize()


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

src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)
src_production = src_positions[0:10]

# ============================================================================
# Per-run bookkeeping
# ============================================================================
sample_log_file = data_dir + f"/sample_log/TMDWF_{sm_tag}_{conf}"
if latt_info.mpi_rank == 0:
    completed_samples = read_sample_log_entries(sample_log_file)
else:
    completed_samples = None
completed_samples = set(getMPIComm().bcast(completed_samples, root=0))


# ============================================================================
# Main source loop
# ============================================================================
for ipos, pos in enumerate(src_production):
    sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
    mpi_print(latt_info, f"Contraction START: {sample_log_tag}")
    if sample_log_tag in completed_samples:
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
    p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    Measurement.contract_2pt_pion(
        latt_info,
        propag_f,
        propag_b,
        phases_2pt,
        tag,
        source_gamma_label="dagger_of_sink",
    )

    sync_cuda()
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}")

    # ------------------------------------------------------------------------
    # DA Wilson-line geometry
    # ------------------------------------------------------------------------
    W_index_list = Measurement.create_PDF_Wilsonline_index_list()

    # ------------------------------------------------------------------------
    # CG DA block
    # ------------------------------------------------------------------------
    mpi_print(latt_info, "Contraction: Start DA: CG (bT=0 TMDWF)")
    collect_by_src = Measurement.contract_DA(
        latt_info,
        None,
        propag_f,
        propag_b,
        phases_2pt,
        W_index_list,
        da_src_gammalist,
        gauge_invariant=False,
    )
    save_da_correlators(collect_by_src, pos, W_index_list, "CG")
    mpi_print(latt_info, "Contraction: Done DA: CG (bT=0 TMDWF)")

    # ------------------------------------------------------------------------
    # GI DA block
    # ------------------------------------------------------------------------
    mpi_print(latt_info, "Contraction: Start DA: GI straight link (bT=0 TMDWF)")
    collect_by_src = Measurement.contract_DA(
        latt_info,
        gauge,
        propag_f,
        propag_b,
        phases_2pt,
        W_index_list,
        da_src_gammalist,
        gauge_invariant=True,
    )
    save_da_correlators(collect_by_src, pos, W_index_list, "GI")
    mpi_print(latt_info, "Contraction: Done DA: GI straight link (bT=0 TMDWF)")

    # ------------------------------------------------------------------------
    # Bookkeeping for completed source positions
    # ------------------------------------------------------------------------
    if latt_info.mpi_rank == 0:
        append_sample_log_entry(sample_log_file, sample_log_tag)
    getMPIComm().Barrier()

    mpi_print(latt_info, f"DONE: {sample_log_tag}")
