import numpy as np
from opt_einsum import contract

# load pyquda modules
from pyquda.field import LatticeInfo, LatticeGauge
from pyquda_utils import core, gamma, source, phase
from pyquda_comm.array import arrayIdentity, arrayZeros
from pyquda_measurement_utils.tools import mpi_print

def _F_clover_traceless(U: LatticeGauge, mu: int, nu: int):
    """
    Clover 1x1 field-strength-like tensor F_{mu,nu}:
      - build 4 plaquettes (clover) from U.loop
      - average over corners
      - project to anti-Hermitian traceless
      - multiply overall factor (-i/2)
    Result is stored in-place in a LatticeGauge-like object, same type as U.
    """
    loops_one = [
        [mu,   nu,   mu+4, nu+4],
        [nu,   mu+4, nu+4, mu],
        [mu+4, nu+4, mu,   nu],
        [nu+4, mu,   nu,   mu+4],
    ]

    # 1) sum over clover corners
    F = U.loop([loops_one] * 4, coeff=[1.0, 1.0, 1.0, 1.0])   # same type as U
    data = F.data                                             # (..., Nc, Nc)

    # 2) anti-Hermitian part: A = 1/8 (F - F†) (including average clover factor 1/4)
    A = 0.125 * (data - data.swapaxes(-2, -1).conjugate())

    # 3) traceless: A -> A - tr(A)/Nc * I
    Nc = A.shape[-1]
    trA = contract('...ii->...', A)                  # (...,)
    I = arrayIdentity(Nc, A.dtype, F.location)       # (Nc, Nc)
    A -= trA[..., None, None] * I / Nc

    # 4) overall normalization: F_mu_nu = (-i) * A
    data[...] = (-1j) * A

    return F

def _all_F_clover_traceless(U: LatticeGauge,):
    F = [[None]*4 for _ in range(4)]
    planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    for mu, nu in planes:
        F_mu_nu = (_F_clover_traceless(U, mu, nu)).data[0]
        F[mu][nu] = F_mu_nu
        F[nu][mu] = -F_mu_nu

    return F

def flowed_gluonic_EMT_P_pyquda(
    U: LatticeGauge,
    stepsize: float = 0.1,
    Nsteps: int = 20,
    datfile: str = "",
    n_max: int = 0,
    improve: bool = False,
):
    """
    PyQUDA version of flowed_gluonic_EMT_P, using
    _F_clover_traceless / _all_F_clover_traceless and `contract`
    for all color traces and contractions.

    U: LatticeGauge (4D gauge field, with geometry in U.latt_info)
    Return value: T_{mu nu}(n_x,n_y,n_z,flow_step,t) numpy array
           Shape: (4,4,n_max+1,n_max+1,n_max+1,Nsteps+1,Nt)
    """
    # --- Geometry information extracted from PyQUDA's LatticeGauge ---
    latt_info = U.latt_info
    global_size = latt_info.global_size
    Lx, Ly, Lz, Lt = latt_info.size
    Ns3 = global_size[0] * global_size[1] * global_size[2]

    # Tmunu_t[mu,nu,nx,ny,nz,step,t]
    Tmunu_t = np.zeros(
        (4, 4, n_max + 1, n_max + 1, n_max + 1, Nsteps + 1, global_size[3]),
        dtype=np.complex128,
    )

    # To avoid modifying input U, create a copy (if in-place flow is acceptable, U can be used directly)
    U_flow = U.copy()

    for step in range(Nsteps + 1):
        mpi_print(latt_info, f"step {step} calculate F")

        # --------- 1. Compute all F_{mu,nu} (clover + traceless) ---------
        # _all_F_clover_traceless(U) returns F[mu][nu] = array(shape=(Nlat, Nc,Nc))
        F = _all_F_clover_traceless(U_flow)

        mpi_print(latt_info, f"step {step} calculate T")

        # --------- 2. Construct EMT T_{mu,nu}(x) from F and perform momentum projection ---------
        # Original code:
        # tmp(x) = sum_{rho != mu,nu} tr_c [ F_{mu,rho}(x) * F_{nu,rho}(x) ]
        # Then perform 3D sum of P(p) * tmp, slice over time
        for mu in range(4):
            for nu in range(mu, 4):
                # tmp: complex field on site, shape=(Nlat,)
                tmp = arrayZeros((2, Lt, Lz, Ly, Lx // 2), U.data.dtype, U.location)       # (Nc, Nc)

                for rho in range(4):
                    if rho == mu or rho == nu:
                        continue

                    # F_{mu,rho}(x), F_{nu,rho}(x), shape = (Nlat, Nc, Nc)
                    F_mr = F[mu][rho]
                    F_nr = F[nu][rho]

                    # color trace of matrix product:
                    # tr(F_mr * F_nr) = sum_{a,b} F_mr[a,b] * F_nr[b,a]
                    # Using einsum: '...ab,...ba->...'
                    tmp += contract("...ab,...ba->...", F_mr, F_nr)

                # ---- 3. Perform plane wave projection for each (n_x,n_y,n_z) and sum over space ----
                # Original code: P = g.exp_ixp(2π * [2nx,2ny,2nz,0]/L), then g.slice(P*tmp, 3)
                # Here P(x)=exp(i p·x) is constructed manually, then sum over (x,y,z) to get values for each t
                for nx in range(n_max + 1):
                    for ny in range(n_max + 1):
                        for nz in range(n_max + 1):
                            # p_mu = 2π * (2n_mu / L_mu), last component is fixed to 0
                            qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]

                            # phase(x) = p · x
                            phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0,0,0,0])
                            
                            # Sum over (x,y,z), keep t dimension, shape=(Lt,)
                            slice_t = core.gatherLattice(contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get(), [1, -1, -1, -1])

                            if U.latt_info.mpi_rank == 0:
                                Tmunu_t[mu, nu, nx, ny, nz, step, :] += 2.0 * slice_t[0]

        # --------- 4. Update U_flow using Wilson flow / Zeuthen flow ---------
        # TODO add improve option
        if Nsteps > 0:
            if step == 0:
                mpi_print(latt_info, f"wilsonFlow step = {step}")
                energy = U_flow.wilsonFlow(10, epsilon=stepsize / 10)
            elif step < Nsteps:
                mpi_print(latt_info, f"wilsonFlow step = {step}")
                energy = U_flow.wilsonFlow(1, epsilon=stepsize)

    # --------- 5. Normalize & save ---------
    # Original code finally divides by Ns3
    Tmunu_t /= Ns3

    for mu in range(4):
        for nu in range(mu, 4):
            suffix = f".T{mu+1}{nu+1}.n_max{n_max}.pyquda.npy"
            np.save(datfile + suffix, Tmunu_t[mu, nu])

    return Tmunu_t
