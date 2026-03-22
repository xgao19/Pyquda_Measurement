import numpy as np
from opt_einsum import contract
import cupy as cp

# load pyquda modules
from pyquda import getMPIComm
from pyquda.field import LatticeInfo, LatticeGauge, LatticePropagator, LatticeFermion
from pyquda_utils import core, gamma, source, phase, convert
from pyquda_comm.array import arrayIdentity, arrayZeros
from pyquda_measurement_utils.tools import mpi_print


GEN_SIMD_WIDTH = 64

D_gammas = [
    cp.asarray(gamma.gamma(1)),
    cp.asarray(gamma.gamma(2)),
    cp.asarray(gamma.gamma(4)),
    cp.asarray(gamma.gamma(8)),
]

# Please confirm this is gamma5 in your convention
G5 = cp.asarray(gamma.gamma(15))


def make_zn_noise_fermion(latt_info, n=2):

    xi = LatticeFermion(latt_info)
    r = cp.random.randint(0, n, size=xi.data.shape)
    xi.data[:] = cp.exp(2j * cp.pi * r / n).astype(xi.data.dtype)
    return xi

def _impose_P_Breit_slice(U: LatticeGauge, complex_field, n_max, realize=False):
    mpi_print(U.latt_info, f'impose_P_Breit_slice n_max = {n_max}')
    Nt = U.latt_info.global_size[3]
    results = np.zeros((n_max+1,n_max+1,n_max+1,Nt),dtype=np.complex128)
    for nx in range(n_max+1):
        for ny in range(n_max+1):
            for nz in range(n_max+1):
                qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]
                phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0,0,0,0])
                slice_t = core.gatherLattice(contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field).get(), [1, -1, -1, -1])
                slice_t = getMPIComm().bcast(slice_t, root=0)
                results[nx,ny,nz] += slice_t[0]
    return results


def _get_Tmunu_symmetrized_P_Breit_slice(U_f: LatticeGauge, xi: LatticeFermion, eta: LatticeFermion, n_max: int):
    Nt = U_f.latt_info.global_size[3]

    # s term
    CHI = np.zeros([2,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    
    dot_xi_eta = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), eta.data) 
    CHI[0] = _impose_P_Breit_slice(U_f, dot_xi_eta, n_max, realize=True)
    dot_xi_xi  = contract('etzyxbc,etzyxbc->etzyx', xi.data.conj(), xi.data) 
    CHI[1] = _impose_P_Breit_slice(U_f, dot_xi_xi, n_max, realize=True)

    # t term
    Tmunu = np.zeros([4,4,n_max+1,n_max+1,n_max+1,Nt], dtype=np.complex128)
    U_f.gauge_dirac.loadGauge(U_f)
    for mu in range(4):
        #\psi'(x)=U_\mu(x)\psi(x+\hat\mu)0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
        tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu+4) 
        pyquda_tmp = contract('...sc,...sc->...', tmp.data.conj(), xi.data)
        pyquda_tmp = _impose_P_Breit_slice(U_f, pyquda_tmp, n_max, realize=True)
        for nu in range(4):
            Y = contract('ab,...bc->...ac', cp.asarray(D_gammas[nu]), tmp.data)
            complex_field = contract('...sc,...sc->...', xi.data.conj(), Y)
            Tmunu[nu,mu] += -0.5*_impose_P_Breit_slice(U_f, complex_field, n_max, realize=True)

    # symmetrization
    for mu in range(4):
        for nu in range(mu+1,4):
            Tmunu[mu,nu] = ( Tmunu[mu,nu] + Tmunu[nu,mu] ) / 2
            Tmunu[nu,mu] = Tmunu[mu,nu]

    return Tmunu, CHI

def flowed_fermionic_EMT_pyquda(
    gaugePara, 
    invPara,
    flowPara, 
    randPara,
    stepsize: float = 0.1,
    Nsteps: int = 20,
    datfile: str = "",
    n_max: int = 0,
    improve: bool = False
):

    Nv, n_input, randseed = randPara
    a, conf_id, U = gaugePara
    stepsize, Nsteps, improve, division = flowPara
    latt_info = U.latt_info
    Nx, Ny, Nz, Nt = latt_info.global_size
    Lt = latt_info.Lt
    gt = latt_info.gt

    # --- 几何信息，从 PyQUDA 的 LatticeGauge 中取 ---
    global_size = U.latt_info.global_size
    Ns3 = global_size[0] * global_size[1] * global_size[2]
    Nt = global_size[3]

    mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
    dirac = core.getDirac(latt_info, invPara[0], invPara[2],  5000, 1.0, invPara[1], invPara[1], [[8, 8, 4, 4]])
    dirac.loadGauge(U)
    mpi_print(latt_info, "Multigrid inverter ready.")

    cp.random.seed(randseed)

    Tmunu = np.zeros([Nv,4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    CHI = np.zeros([Nv,2,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt], dtype=np.complex128)
    for vec_picked in range(Nv):

        mpi_print(U.latt_info, f'vec {vec_picked}')
        xi = make_zn_noise_fermion(latt_info, n=n_input)
        eta = dirac.invert(xi)

        U_f = U.copy()  # to avoid modifying U in place before flow
        U_f.setAntiPeriodicT()

        for step in range(Nsteps+1):
            mpi_print(U_f.latt_info, f'calc Tmunu, step = {step}')
            U_f.gauge_dirac.loadGauge(U_f)
            
            tmpt,tmps = _get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
            Tmunu[vec_picked,:,:,:,:,:,step,:] += tmpt
            CHI[vec_picked,:,:,:,:,step,:] += tmps

            if Nsteps > 0:

                temp = core.MultiLatticeFermion(U.latt_info, 2, cp.array([xi.data, eta.data]))
                temp_flow = U_f.gradientFlow(temp, "wilson", 1, stepsize)
                xi, eta = temp_flow[0], temp_flow[1]

    mpi_print(U.latt_info, f"random vectors done.")

    np.save(f'{datfile}/cTmunu_pervec.pyquda.npy', Tmunu)
    np.save(f'{datfile}/cCHI_pervec.pyquda.npy', CHI)
    
    Tmunu = np.mean(Tmunu,axis=0) / Ns3
    CHI = np.mean(CHI,axis=0) / Ns3
    for mu in range(4):
        for nu in range(mu,4):
            np.save(f'{datfile}/cT{mu+1}{nu+1}.pyquda.npy', Tmunu[mu,nu])
    np.save(f'{datfile}/cCHI.pyquda.npy', CHI)