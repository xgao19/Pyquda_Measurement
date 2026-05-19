import numpy as np
import datetime
import sys
import time

# load pyquda modules
from pyquda import init
from pyquda_utils import core, gamma, io
import subprocess
# from pyquda_measurement_utils import EMT_gluon_1pt
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT, GluonEMT
from pyquda_measurement_utils.tools import mpi_print

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Gobal parameters
data_dir="/lustre/orion/nph158/proj-shared/xgao/l64c64a076/EMT_meson_pyquda/data" # FIXME
lat_tag = "l64c64a076" # FIXME
GEN_SIMD_WIDTH = 64
sm_tag = "1HYP_GSRC_W10_k0" # NOTE

# --------------------------
# initiate quda
# --------------------------
init(mpi_geometry, enable_mps=True)


# --------------------------
# Setup parameters
# --------------------------

parameters = {

    "qext": [[x,y,z,0] for x in [-2,-1,0,1,2] for y in [-2,-1,0,1,2] for z in [-2,-1,0,1,2]], # momentum transfer for TMD, pf = pi + q
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [-2,-1,0,1,2] for y in [-2,-1,0,1,2] for z in [-2,-1,0,1,2]], # 2pt momentum

    "pos_boost" : [0,0,0], # boosted smearing for quark
    "neg_boost" : [0,0,0], # boosted smearing for anti-quark
    "width" : 1.0, # Gaussian smearing width

    "flow_type": "Wilson", # type of flow: Wilson, Zeuthen, Symanzik
    "flow_epsilon": 0.1, # flow time step size
    "flow_steps": 10, # number of flow steps
}

###################### load gauge ######################
Ls = 64
Lt = 64
# gauge = io.readNERSCGauge(f"/ccs/home/xiangg/latwork/l64c64a076/nucleon_TMD_noGPT/gauge/l6464f21b7130m00119m0322a.{conf}.coulomb.1e-14.HYP")
gauge = io.readNERSCGauge(f"/lustre/orion/nph158/proj-shared/jinchen/debug/nucleon_TMD/fixed_GLU/l6464f21b7130m00119m0322a.{conf}.coulomb.1e-14")
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
latt_info = gauge.latt_info

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")


# --------------------------
# Start measurements
# --------------------------

###################### gluonic EMT ######################
# gauge.toDevice()

# GluonEMT.flowed_1pt(
#     gauge,
#     stepsize=parameters["epsion"],
#     Nsteps=parameters["Nsteps"],
#     datfile=f"{data_dir}/EMTg/gEMT_{conf}",
#     n_max=3,
# )

gluon_emt = GluonEMT(parameters)

gauge.toDevice()
gluon_emt.flowed_1pt(
    gauge,
    datfile=f"{data_dir}/EMTg/gEMT_{conf}",
)
