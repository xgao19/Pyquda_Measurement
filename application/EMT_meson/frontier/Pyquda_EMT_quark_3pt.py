import numpy as np
import datetime
import sys
import time

# load pyquda modules
from pyquda import init
from pyquda.field import LatticeInfo, LatticeGauge, LatticeMom
from pyquda_utils import core, gamma, io
# from pyquda_measurement_utils import EMT_quark_3pt
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

    "CG_GaussSmear" : True, # whether to use CG-based Gaussian smearing
    "pos_boost" : [0,0,0], # boosted smearing for quark
    "neg_boost" : [0,0,0], # boosted smearing for anti-quark
    "width" : 1.0, # Gaussian smearing width

    "flow_type": "wilson", # type of flow: wilson, symanzik
    "flow_epsilon": 0.207936, # flow time step size
    "flow_steps": 10, # number of flow steps
}

# --------------------------
# Start measurements
# --------------------------

###################### load gauge ######################

gauge = io.readNERSCGauge(f"/lustre/orion/nph158/proj-shared/jinchen/debug/nucleon_TMD/fixed_GLU/l6464f21b7130m00119m0322a.{conf}.coulomb.1e-14")
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info

###################### test wilson flow: fermion part ######################

gaugePara = [0.076, conf, gauge]
invPara = [0.236, 1.0372, 1e-15, 300] # mf, csw, prec, cgMax
flowPara = [0.207936, 10, False, 1] # stepsize, Nsteps, improve, division
smearPara = [False, False, 4.8, 48]

# QuarkEMT.connected_3pt(
#     gaugePara,
#     invPara,
#     flowPara,
#     smearPara,
#     Nsrc=1,
#     sinkt_range=[2, 3],
#     spin=5,
#     datfile="/ccs/home/xiangg/latwork/l64c64a076/EMT_meson_pyquda/data/EMT3pt",
# )

quark_emt = QuarkEMT(parameters)

quark_emt.connected_3pt(
    gauge,
    invPara,
    src_pos=[0, 0, 0, 0],      # 这里改成你真正想要的源位置
    t_separations=[2, 3],
    spin=5,
    datfile="/ccs/home/xiangg/latwork/l64c64a076/EMT_meson_pyquda/data/EMT3pt",
)
