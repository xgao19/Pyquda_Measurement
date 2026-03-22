#!/usr/bin/env python3
import numpy as np
from opt_einsum import contract
import cupy as cp

from pyquda import getMPIComm
from pyquda.field import LatticeGauge, LatticePropagator, MultiLatticeFermion
from pyquda_utils import core, gamma, convert, source
from pyquda_utils.source import sequential12 

from pyquda_measurement_utils.tools import mpi_print
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing

GEN_SIMD_WIDTH = 64

D_gammas = [
    cp.asarray(gamma.gamma(1)),
    cp.asarray(gamma.gamma(2)),
    cp.asarray(gamma.gamma(4)),
    cp.asarray(gamma.gamma(8)),
]

# Please confirm this is gamma5 in your convention
G5 = cp.asarray(gamma.gamma(15))


def _covdev_sym_prop(U_f: LatticeGauge, prop: LatticePropagator, mu: int):
    """
    Symmetric covariant derivative on propagator:
        0.5 * (D_{+mu} - D_{-mu})

    Do it column by column in MultiLatticeFermion space, then convert back
    to propagator.
    """
    U_f.gauge_dirac.loadGauge(U_f)

    mf = convert.propagatorToMultiFermion(prop)
    mf_covdev = convert.propagatorToMultiFermion(prop)

    for spin in range(4):
        for color in range(3):
            idx = spin * 3 + color
            Dp = U_f.pure_gauge.covDev(mf[idx], mu)
            Dm = U_f.pure_gauge.covDev(mf[idx], mu + 4)
            mf_covdev[idx] = 0.5 * (Dp - Dm)

    return convert.multiFermionToPropagator(mf_covdev)


def _left_covdev_dst2_from_dsty(U_f: LatticeGauge, dst_y: LatticePropagator, mu: int):
    """
    Construct left-acting covariant derivative on
        dst2 = gamma5 * adj(dst_y) * gamma5

    using
        leftD(dst2) = gamma5 * adj(D dst_y) * gamma5
    """
    D_y = _covdev_sym_prop(U_f, dst_y, mu)
    D_y_dag = D_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7)
    leftD_dst2 = contract("ab,...bcij,cd->...adij", G5, D_y_dag, G5)
    return leftD_dst2


def _flow_two_props_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    stepsize: float,
    flow_type: str = "wilson",
):
    """
    Flow two propagators together using PyQUDA gradientFlow, while U_f
    is updated in place.
    """
    mf_y = convert.propagatorToMultiFermion(dst_y)
    mf_seq = convert.propagatorToMultiFermion(dst_seq)

    L5_y = mf_y.L5
    L5_seq = mf_seq.L5
    assert L5_y == L5_seq

    packed = MultiLatticeFermion(
        U_f.latt_info,
        L5_y + L5_seq,
        cp.concatenate([mf_y.data, mf_seq.data], axis=0),
    )

    packed_flow = U_f.gradientFlow(packed, flow_type, 1, stepsize)

    mf_y_flow = MultiLatticeFermion(
        U_f.latt_info,
        L5_y,
        packed_flow.data[:L5_y].copy(),
    )
    mf_seq_flow = MultiLatticeFermion(
        U_f.latt_info,
        L5_seq,
        packed_flow.data[L5_y:L5_y + L5_seq].copy(),
    )

    dst_y_flow = convert.multiFermionToPropagator(mf_y_flow)
    dst_seq_flow = convert.multiFermionToPropagator(mf_seq_flow)

    return dst_y_flow, dst_seq_flow


def get_C3_chi_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    t0: int,
):
    """
    C3_chi(t) = Tr[ dst2 * dst_seq ]
    with dst2 = gamma5 * adj(dst_y) * gamma5
    Local contraction reduces wtzyx+spin+color to t,
    then core.gatherLattice handles MPI combination.
    """
    dst2 = contract(
        "ab,wtzyxbcij,cd->wtzyxadij",
        G5,
        dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
        G5,
    )

    scalar_t = contract("wtzyxabij,wtzyxbaji->t", dst2, dst_seq.data)

    slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
    slice_t = getMPIComm().bcast(slice_t, root=0)

    return np.roll(np.array(slice_t.real), -t0)

def get_C3_Tmunu_symmetrized_pyquda(
    U_f: LatticeGauge,
    dst_y: LatticePropagator,
    dst_seq: LatticePropagator,
    t0: int,
):
    Nt = U_f.latt_info.global_size[3]
    C3_Tmunu = np.zeros((4, 4, Nt), dtype=np.float64)

    # dst2 = gamma5 * adj(dst_y) * gamma5
    # dst_y.data: wtzyxabij
    # adj(dst_y): wtzyxbaji
    # dst2:       wtzyxadij
    dst2 = contract(
        "ab,wtzyxbcij,cd->wtzyxadij",
        G5,
        dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
        G5,
    )

    # first term: +1/2 Tr[ dst2 * gamma_nu * D_mu(dst_seq) ]
    for mu in range(4):
        D_seq = _covdev_sym_prop(U_f, dst_seq, mu)   # now guaranteed propagator-shaped

        for nu in range(4):
            # gamma_D_seq: wtzyxadij
            gamma_D_seq = contract(
                "ab,wtzyxbdij->wtzyxadij",
                D_gammas[nu],
                D_seq.data,
            )

            # trace over a,d,i,j and local sum over w,z,y,x -> keep t
            scalar_t = 0.5 * contract(
                "wtzyxadij,wtzyxdaji->t",
                dst2,
                gamma_D_seq,
            )

            slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
            slice_t = getMPIComm().bcast(slice_t, root=0)
            C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

    # second term: -1/2 Tr[ leftD_mu(dst2) * gamma_nu * dst_seq ]
    for mu in range(4):
        leftD_dst2 = _left_covdev_dst2_from_dsty(U_f, dst_y, mu)   # wtzyxadij

        for nu in range(4):
            gamma_dst_seq = contract(
                "ab,wtzyxbdij->wtzyxadij",
                D_gammas[nu],
                dst_seq.data,
            )

            scalar_t = -0.5 * contract(
                "wtzyxadij,wtzyxdaji->t",
                leftD_dst2,
                gamma_dst_seq,
            )

            slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
            slice_t = getMPIComm().bcast(slice_t, root=0)
            C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

    for mu in range(4):
        for nu in range(mu + 1, 4):
            C3_Tmunu[mu, nu] = 0.5 * (C3_Tmunu[mu, nu] + C3_Tmunu[nu, mu])
            C3_Tmunu[nu, mu] = C3_Tmunu[mu, nu]

    return C3_Tmunu


def C3_con_EMT_pyquda(
    gaugePara,
    invPara,
    flowPara,
    smearPara,
    Nsrc,
    sinkt_range,
    spin,
    datfile,
):
    """
    Hybrid migration version of quark EMT 3pt.

    - source/smear/inversion/sequential source remain in GPT
    - gauge + propagator gradient flow is done by PyQUDA
    - C3_chi and C3_Tmunu contractions are done in PyQUDA style
    - NO flattening of spin/color indices
    """
    assert spin in [0, 1, 2, 5]
    N_sinkt = len(sinkt_range)

    a, conf_id, U = gaugePara
    stepsize, Nsteps, improve, division = flowPara
    to_sm_src, to_sm_dst, sm_sigma, sm_steps = smearPara
    latt_info = U.latt_info
    Nx, Ny, Nz, Nt = latt_info.global_size
    Lt = latt_info.Lt
    gt = latt_info.gt

    if improve:
        raise NotImplementedError(
            "Current PyQUDA flow path here only implements Wilson flow; improve=True not yet wired."
        )

    mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
    dirac = core.getDirac(latt_info, invPara[0], invPara[2],  5000, 1.0, invPara[1], invPara[1], [[8, 8, 4, 4]])
    dirac.loadGauge(U)
    mpi_print(latt_info, "Multigrid inverter ready.")

    C2 = np.zeros((Nsrc, Nt), dtype=np.float64)
    C3_chi = np.zeros((Nsrc, N_sinkt, Nsteps + 1, Nt), dtype=np.float64)
    C3_Tmunu = np.zeros((Nsrc, N_sinkt, Nsteps + 1, 4, 4, Nt), dtype=np.float64)
    src_locs = np.zeros((Nsrc, 4), dtype=np.uint16)

    for n_src in range(Nsrc):
        x0 = int(n_src % Nx)
        y0 = int(n_src % Ny)
        z0 = int(n_src % Nz)
        t0 = int(n_src % Nt)

        mpi_print(latt_info, f"src {n_src}: [{x0},{y0},{z0},{t0}]")
        src_locs[n_src] = np.array([x0, y0, z0, t0], dtype=np.uint16)

        pos = [x0, y0, z0, t0]
        src = source.propagator(latt_info, "point", pos)

        if to_sm_src:
            mpi_print(latt_info, "source smearing starts")
            src = boosted_smearing(src, w=sm_sigma, boost=[0,0,0])
            mpi_print(latt_info, "source smearing ends")

        dirac.loadGauge(U)
        dst_x = core.invertPropagator(dirac, src, 1, 0)

        del src

        dst_y_back = dst_x.copy()

        if to_sm_dst:
            mpi_print(latt_info, "first sink smearing starts")
            dst_x = boosted_smearing(dst_x, w=sm_sigma, boost=[0,0,0])

        bw_prop = contract("ij, wtzyxilab, kl -> wtzyxkjba", G5, dst_x.data.conj(), G5)
        bw_prop = contract("wtzyxjicf, im -> wtzyxjmcf", bw_prop, G5)
        scalar_t = contract("wtzyxjiab, wtzyxilba, lj -> t", bw_prop, dst_x.data, G5)
        slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
        slice_t = getMPIComm().bcast(slice_t, root=0)
        C2[n_src] += np.roll(slice_t.real,-t0)

        if to_sm_dst:
            mpi_print(latt_info, "second sink smearing starts")
            dst_x = boosted_smearing(dst_x, w=sm_sigma, boost=[0,0,0])
            mpi_print(latt_info, "sink smearings end")

        for n_t, sink_t in enumerate(sinkt_range):
            mpi_print(latt_info, f"create sequential source sink_t = {sink_t}")

            t_sink = (sink_t + t0) % Nt

            # Use pyquda_utils.source.sequential12 to handle time slicing/masking.
            # This operates directly on the LatticePropagator without manual reshape/masking.
            src_seq_sliced = sequential12(dst_x, t_sink)
            src_seq_data = contract("ij, wtzyxilab, kl -> wtzyxjkab", G5, src_seq_sliced.data, G5)
            src_seq = core.LatticePropagator(latt_info)
            src_seq.data = src_seq_data

            dirac.loadGauge(U)
            dst_seq_py = core.invertPropagator(dirac, src_seq, 1, 0)
            del src_seq, src_seq_data

            dst_y_py = dst_y_back.copy()

            U_f = U.copy()  # to avoid modifying U in place before flow
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(latt_info, f"contraction for step {step}")

                C3_chi[n_src, n_t, step] += get_C3_chi_pyquda(
                    U_f, dst_y_py, dst_seq_py, t0
                )

                C3_Tmunu[n_src, n_t, step] += get_C3_Tmunu_symmetrized_pyquda(
                    U_f, dst_y_py, dst_seq_py, t0
                )

                if step < Nsteps:
                    dst_y_py, dst_seq_py = _flow_two_props_pyquda(
                        U_f,
                        dst_y_py,
                        dst_seq_py,
                        stepsize,
                        flow_type="wilson",
                    )

            del U_f, dst_y_py, dst_seq_py

    np.save(f"{datfile}/C2_spin{spin}_HYP_SS_persrc.pyquda.npy", C2)
    np.save(f"{datfile}/C3_chi_spin{spin}_HYP_SS_persrc.pyquda.npy", C3_chi)
    np.save(f"{datfile}/C3_Tmunu_spin{spin}_HYP_SS_persrc.pyquda.npy", C3_Tmunu)
    np.save(f"{datfile}/src_locs_spin{spin}_HYP_SS.pyquda.npy", src_locs)

    C2 = np.mean(C2, axis=0)
    C3_chi = np.mean(C3_chi, axis=0)
    C3_Tmunu = np.mean(C3_Tmunu, axis=0)

    np.save(f"{datfile}/C2_spin{spin}_HYP_SS.pyquda.npy", C2)
    for n_t, sink_t in enumerate(sinkt_range):
        np.save(f"{datfile}/C3_chi_sinkt{sink_t}_spin{spin}_HYP_SS.pyquda.npy", C3_chi[n_t])
        np.save(f"{datfile}/C3_Tmunu_sinkt{sink_t}_spin{spin}_HYP_SS.pyquda.npy", C3_Tmunu[n_t])