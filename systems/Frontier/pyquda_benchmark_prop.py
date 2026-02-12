'''
v2 means no mesonAllSinkTwoPoint, for non-CUDA environments
'''


# load python modules
import time
import os
import numpy as np

from pyquda import init, getMPIComm

# --------------------------
# initiate quda
# --------------------------
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
print(f"DEBUG: mpi_geometry from args: {mpi_geometry}")
init(mpi_geometry, enable_mps=True, grid_map="shared", resource_path=".cache")
#init(mpi_geometry, enable_mps=True, grid_map="default", backend="dpnp", backend_target="sycl", resource_path=".cache")

from pyquda_utils import core, phase, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda.field import LatticeGauge

# Global parameters
data_dir="/ccs/home/xiangg/latwork/l64c64a076/nucleon_TMD_noGPT/data" # NOTE
lat_tag = "l64c64a076" # NOTE
interpolation = "T5" # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_"+interpolation # NOTE
GEN_SIMD_WIDTH = 64

# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 64
Lt = 64
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.049 # kappa = 0.12623
csw_r = 1.0372
csw_t = 1.0372
multigrid = [[4, 4, 4, 4]]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

dirac = core.getDirac(latt_info, mass, 1e-10, 5000, xi_0, csw_r, csw_t, multigrid)
gauge = io.readNERSCGauge(f"/ccs/home/xiangg/latwork/l64c64a076/nucleon_TMD_noGPT/gauge/l6464f21b7130m00119m0322a.1050.coulomb.1e-14.HYP", checksum=False, link_trace=False, plaquette=False) #todo: done hyp by gpt

t0 = time.time()
pos = [1,11,16,17]
srcD = source.propagator(latt_info, "point", pos)
dirac.loadGauge(gauge)
b = core.invertPropagator(dirac, srcD, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version

a = io.readNPYPropagator("data/prop_10p10/x1y11z16t17.h5.npy")

a.toHost()
b.toHost()
print((a-b).norm2() ** 0.5,'/', b.norm2() ** 0.5)

#tag = data_dir + f"/prop/x{pos[0]}y{pos[1]}z{pos[2]}t{pos[3]}"
#propag.save(tag+".h5")

a.toDevice()
b.toDevice()
pion_mpi_a = contract(
    "wtzyxjiba,jk,wtzyxklba,li->t",
    a.data.conj(),
    G5 @ G5,
    a.data, #propagator.data.shape: (2, Lt, Lz, Ly, Lx // 2, Ns, Ns Nc, Nc)
    G5 @ G5,
)
pion_a = core.gatherLattice(pion_mpi_a.get(), [0, -1, -1, -1]) # gather mpi data into root rank
pion_mpi_b = contract(
    "wtzyxjiba,jk,wtzyxklba,li->t",
    b.data.conj(),
    G5 @ G5,
    b.data, #propagator.data.shape: (2, Lt, Lz, Ly, Lx // 2, Ns, Ns Nc, Nc)
    G5 @ G5,
)
pion_b = core.gatherLattice(pion_mpi_b.get(), [0, -1, -1, -1]) # gather mpi data into root rank
if latt_info.mpi_rank == 0:
    np.savetxt("data/c2pt/pion_a.dat", pion_a)
