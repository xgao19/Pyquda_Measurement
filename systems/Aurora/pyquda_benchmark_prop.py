'''
v2 means no mesonAllSinkTwoPoint, for non-CUDA environments
'''


# load python modules
import time
import os
import numpy as np
import dpnp as dnp

from pyquda import init, getMPIComm

from pyquda_utils import core, phase, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda.field import LatticeGauge

from utils.boosted_smearing_pyquda import boosted_smearing
#from utils.bw_seq_pyquda import create_bw_seq_pyquda
#from utils.proton_qTMD_pyquda import proton_TMD, my_pyquda_gammas
from utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMD_file_tag, save_qTMD_proton_hdf5_noRoll
from utils.tools import srcLoc_distri_eq, mpi_print

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
init(mpi_geometry, enable_mps=True, grid_map="shared", backend="dpnp", backend_target="sycl", resource_path=".cache")
#init(mpi_geometry, enable_mps=True, grid_map="default", backend="dpnp", backend_target="sycl", resource_path=".cache")

from pyquda_utils import gamma
G5 = gamma.gamma(15)

# Global parameters
data_dir="/lus/flare/projects/StructNGB/xgao/run/l64c64a076/full_TMD//data" # NOTE
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

dirac = core.getDirac(latt_info, mass, 1e-12, 5000, xi_0, csw_r, csw_t, multigrid)
gauge = io.readNERSCGauge(f"/lus/flare/projects/StructNGB/xgao/run/l64c64a076/full_TMD/gauge/l6464f21b7130m00119m0322a.1050.coulomb.1e-14.HYP", checksum=False, link_trace=False, plaquette=False) #todo: done hyp by gpt

pos = [1,11,16,17]
srcD = source.propagator(latt_info, "point", pos)
dirac.loadGauge(gauge)
t0 = time.time()
propag = core.invertPropagator(dirac, srcD, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
mpi_print(latt_info, f"TIME Pyquda: Forward propagator inversion {time.time() - t0}s")

tag = data_dir + f"/prop_10p12/x{pos[0]}y{pos[1]}z{pos[2]}t{pos[3]}"
propag.save(tag+".h5")


b = propag
a = io.readNPYPropagator("data/prop/x1y11z16t17.h5.npy")
t0 = time.time()
c = core.invertPropagator(dirac, srcD, 1, 0)
mpi_print(latt_info, f"TIME Pyquda: Forward propagator inversion {time.time() - t0}s")
d = io.readNPYPropagator("data_frontier/prop_10p12/x1y11z16t17.h5.npy")
a.toHost()
b.toHost()
c.toHost()
d.toHost()
mpi_print(latt_info,f"(a-b)/b,{(a-b).norm2() ** 0.5} / {b.norm2() ** 0.5}")
mpi_print(latt_info,f"(a-c)/c,{(a-c).norm2() ** 0.5} / {c.norm2() ** 0.5}")
mpi_print(latt_info,f"(b-c)/c,{(b-c).norm2() ** 0.5} / {c.norm2() ** 0.5}")
mpi_print(latt_info,f"(c-d)/d,{(c-d).norm2() ** 0.5} / {d.norm2() ** 0.5}")
mpi_print(latt_info,f"(a-d)/d,{(a-d).norm2() ** 0.5} / {d.norm2() ** 0.5}")

c.toDevice()
d.toDevice()
pion_mpi_c = dnp.einsum(
    "wtzyxjiba,jk,wtzyxklba,li->t",
    c.data.conj(),
    G5 @ G5,
    c.data, #propagator.data.shape: (2, Lt, Lz, Ly, Lx // 2, Ns, Ns Nc, Nc)
    G5 @ G5,
)
#core.gatherLattice(dnp.asnumpy(xp.einsum("qwtzyx, pgwtzyx -> pqgt", phases_pdf_pyq, temp)), [3, -1, -1, -1])
pion_c = core.gatherLattice(dnp.asnumpy(pion_mpi_c), [0, -1, -1, -1]) # gather mpi data into root rank
pion_mpi_d = dnp.einsum(
    "wtzyxjiba,jk,wtzyxklba,li->t",
    d.data.conj(),
    G5 @ G5,
    d.data, #propagator.data.shape: (2, Lt, Lz, Ly, Lx // 2, Ns, Ns Nc, Nc)
    G5 @ G5,
)
pion_d = core.gatherLattice(dnp.asnumpy(pion_mpi_d), [0, -1, -1, -1]) # gather mpi data into root rank
if latt_info.mpi_rank == 0:
    np.savetxt("data/c2pt/pion_c.dat", pion_c)
    np.savetxt("data/c2pt/pion_d.dat", pion_d)
