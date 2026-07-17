"""
Charmonium mass-tuning script with PyQUDA.

This script keeps only the workflow needed for the smeared-source inversions
and the 2pt correlator:

    source -> boosted smearing -> invert propagators -> contract 2pt

The measured 2pt correlator is

    C^(g)_2pt(q, t) =
        sum_x exp(i q . x)
        Tr_{c,s}[ gamma5 S_b(x)^\dagger gamma5 Gamma_g S_f(x) Gamma_src ] ,

with the fixed pseudoscalar source Gamma labelled ``5``.
"""

import time
import numpy as np
import cupy as cp

from pyquda import init, getMPIComm
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.io_corr import get_sample_log_tag, get_c2pt_file_tag
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

source_gamma_label = "5"


# ============================================================================
# Physics / measurement setup
# ============================================================================
init(mpi_geometry, enable_mps=True)

parameters = {
    # Longitudinal staple extent used in the more general TMDWF setup.
    # It is kept here for compatibility with the shared measurement class;
    # in this mass-tuning script it does not play an explicit role.
    "eta": [0],
    # Maximum transverse separation. Setting b_T = 0 keeps only the local / DA-limit setup.
    "b_T": 0,
    # Maximum longitudinal separation. Set to 0 here because this script stops at the 2pt level.
    "b_z": 0,
    # Minimum longitudinal momentum p_z included in the 2pt momentum projection.
    "pzmin": 4,
    # Upper bound of the longitudinal momentum range (Python range end, excluded).
    "pzmax": 11,
    # Width parameter of the boosted Gaussian smearing.
    "width": smearing_width,
    # Momentum used in boosted smearing for the forward/source-side propagator.
    "pos_boost": [0, 0, smearing_boost_k],
    # Momentum used in boosted smearing for the backward/sink-side propagator.
    "neg_boost": [0, 0, -smearing_boost_k],
    # Retained for interface compatibility; this script does not save propagators.
    "save_propagators": False,
}
Measurement = pion_TMDWF_measurement(parameters)


# ============================================================================
# Small helpers
# ============================================================================
# Synchronize CUDA work only where timing boundaries matter.
def sync_cuda():
    cp.cuda.runtime.deviceSynchronize()


# ============================================================================
# Lattice / inverter preparation
# ============================================================================
Ls = 64
Lt = 64
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0

# Main tuning parameter: adjust this to scan the charm quark mass.
mass = 0.167

csw_r = 1.02868
csw_t = 1.02868
multigrid = [[8, 8, 4, 4]]
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

gauge = io.readNERSCGauge(
    f"/lustre/orion/nph174/proj-shared/ensembles/l64c64a040/"
    f"l6464f21b7825m00082m0164a_fixed_GLU/"
    f"l6464f21b7825m00082m0164a.{conf}.coulomb.1e-14"
)
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--mass {mass}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

dirac = core.getClover(latt_info, mass, 1e-15, 10000, xi_0, csw_r, csw_t, multigrid)


# ============================================================================
# Source positions and bookkeeping
# ============================================================================
src_shift = np.array([0, 0, 0, 0]) + np.array([7, 11, 13, 23])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)
src_production = src_positions[0:10]

sample_log_file = data_dir + f"/sample_log/charm_mass_{sm_tag}_{conf}"
if latt_info.mpi_rank == 0:
    completed_samples = read_sample_log_entries(sample_log_file)
else:
    completed_samples = None
completed_samples = set(getMPIComm().bcast(completed_samples, root=0))


# ============================================================================
# Main source loop
# ============================================================================
for ipos, pos in enumerate(src_production):
    sample_log_tag = get_sample_log_tag(
        "ex", pos, sm_tag + f".c2src{source_gamma_label}"
    )
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
    dirac.loadGauge(gauge)
    propag_f = core.invertPropagator(dirac, srcDp, 1, 0)
    propag_b = core.invertPropagator(dirac, srcDm, 1, 0)
    sync_cuda()
    mpi_print(latt_info, f"TIME: Pyquda inversion * 2 {time.time() - t0}")

    # ------------------------------------------------------------------------
    # Two-point correlator
    # ------------------------------------------------------------------------
    t0 = time.time()
    tag = get_c2pt_file_tag(
        data_dir, lat_tag, conf, "ex", pos, sm_tag + f".src{source_gamma_label}"
    )
    p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    Measurement.contract_2pt_pion(
        latt_info,
        propag_f,
        propag_b,
        phases_2pt,
        tag,
        source_gamma_label=source_gamma_label,
    )
    sync_cuda()
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}")

    # ------------------------------------------------------------------------
    # Bookkeeping for completed source positions
    # ------------------------------------------------------------------------
    if latt_info.mpi_rank == 0:
        append_sample_log_entry(sample_log_file, sample_log_tag)
    getMPIComm().Barrier()

    mpi_print(latt_info, f"DONE: {sample_log_tag}")
