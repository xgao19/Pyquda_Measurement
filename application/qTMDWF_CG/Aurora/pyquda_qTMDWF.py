"""Aurora entry point for the shared pion CG qTMDWF runner."""

import argparse
import numpy as np
import sys
from pathlib import Path

from pyquda import init


def _vector(text):
    values = [int(item) for item in text.split(".")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("boost must be X.Y.Z")
    return values


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", default="1.1.1.1")
parser.add_argument("--stream", default="b")
parser.add_argument("--pzmin", type=int, default=4)
parser.add_argument("--pzmax", type=int, default=9)
parser.add_argument("--pos-boost", type=_vector, default=[0, 0, 4])
parser.add_argument("--neg-boost", type=_vector, default=[0, 0, -4])
parser.add_argument("--source-count", type=int, default=2)
args = parser.parse_args()

geometry = [int(item) for item in args.mpi_geometry.split(".")]
init(
    geometry,
    enable_mps=True,
    backend="dpnp",
    backend_target="sycl",
    resource_path=".cache",
)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyquda_utils import core, io
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.tools import srcLoc_distri_eq
from application.qTMDWF_CG.qTMDWF_runner import run_qtmdwf_sources


Ls = Lt = 80
lattice = [Ls, Ls, Ls, Lt]
lat_tag = "l80c80a050"
sm_tag = (
    f"1HYP_M140_GSRC_W70_kp{args.pos_boost[2]}_km{abs(args.neg_boost[2])}"
)
data_dir = f"/lus/flare/projects/StructNGB/xgao/run/l80c80a050/TMDWF_pyquda/data_{args.stream}"
gauge_path = (
    "/lus/flare/projects/StructNGB/xgao/ensembles/s8080b7596/gauge_fixed/"
    f"{args.stream}/l8080f21b7596m00101m0202{args.stream}."
    f"coulomb.1e-14.{args.config_num}"
)
latt_info = core.LatticeInfo(lattice, -1, 1.0)
gauge = io.readNERSCGauge(
    gauge_path, checksum=False, link_trace=False, plaquette=False
)
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
dirac = core.getClover(
    latt_info, -0.0386, 1e-10, 10000, 1.0, 1.03094, 1.03094,
    [[5, 4, 5, 4]],
)
measurement = pion_TMDWF_measurement({
    "eta": [0],
    "b_T": 30,
    "b_z": 30,
    "pzmin": args.pzmin,
    "pzmax": args.pzmax,
    "width": 7.0,
    "pos_boost": args.pos_boost,
    "neg_boost": args.neg_boost,
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
