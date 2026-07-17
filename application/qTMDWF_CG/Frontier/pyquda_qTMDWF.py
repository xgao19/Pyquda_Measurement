"""Frontier entry point for the shared pion CG qTMDWF runner."""

import argparse
import numpy as np
import sys
from pathlib import Path

from pyquda import init


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", default="1.1.1.1")
parser.add_argument("--source-count", type=int, default=1)
args = parser.parse_args()
geometry = [int(item) for item in args.mpi_geometry.split(".")]
init(geometry, enable_mps=True)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyquda_utils import core, io
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.tools import srcLoc_distri_eq
from application.qTMDWF_CG.qTMDWF_runner import run_qtmdwf_sources


lattice = [64, 64, 64, 64]
lat_tag = "l64c64a076"
sm_tag = "1HYP_M140_GSRC_W45_k6"
data_dir = "/ccs/home/xiangg/latwork/l64c64a076/qTMDWF_pyquda/data"
gauge_path = (
    "/ccs/home/xiangg/latwork/l64c64a076/nucleon_TMD_noGPT/gauge/"
    f"l6464f21b7130m00119m0322a.{args.config_num}.coulomb.1e-14.HYP"
)
latt_info = core.LatticeInfo(lattice, -1, 1.0)
gauge = io.readNERSCGauge(gauge_path)
dirac = core.getClover(
    latt_info, -0.049, 1e-10, 10000, 1.0, 1.0372, 1.0372,
    [[8, 8, 4, 4]],
)
measurement = pion_TMDWF_measurement({
    "eta": [0],
    "b_T": 20,
    "b_z": 20,
    "pzmin": 4,
    "pzmax": 11,
    "width": 4.5,
    "pos_boost": [0, 0, 6],
    "neg_boost": [0, 0, -6],
})
origin = (
    np.asarray([args.config_num % extent for extent in lattice])
    + np.asarray([7, 11, 13, 23])
)
positions = srcLoc_distri_eq(lattice, origin)[: args.source_count]
run_qtmdwf_sources(
    latt_info=latt_info,
    dirac=dirac,
    gauge=gauge,
    measurement=measurement,
    source_positions=positions,
    data_dir=data_dir,
    lat_tag=lat_tag,
    config_num=args.config_num,
    sm_tag=sm_tag,
)
