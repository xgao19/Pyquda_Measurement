#!/usr/bin/env python3
"""Unified PyQUDA EMT measurements with detailed comments.

This module collects three measurements relevant for lattice-QCD extractions of
hadron matrix elements of the energy-momentum tensor (EMT):

1. Quark EMT 1pt (stochastic estimator)
2. Quark EMT 3pt (connected diagram with a sequential source)
3. Gluon EMT 1pt (flowed clover field strength)

The implementation is intentionally close to the user's working code.  The main
addition here is *physics-facing comments* explaining:

- what lattice field each object represents,
- why a given numerical step is needed,
- what the key contractions compute,
- and how the resulting arrays are organized.

Conventions used repeatedly below
--------------------------------
For a quark propagator-like object we use the data layout

    wtzyxabij

where
    w   : checkerboard / parity-like internal index used by PyQUDA
    t,z,y,x : local lattice coordinates on the current MPI rank
    a,b : spin row / spin column indices
    i,j : color row / color column indices

A propagator therefore carries *two* spin-color indices.  A fermion field only
carries one spin-color index and is stored as

    etzyxbc

with
    e : checkerboard / parity-like internal index
    b : spin index
    c : color index

Throughout, MPI communication is handled by ``core.gatherLattice``.  The code
usually performs the local contraction first and then lets ``gatherLattice``
combine data across MPI ranks.
"""

from __future__ import annotations

import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import (
    LatticeGauge,
    LatticePropagator,
    LatticeFermion,
    MultiLatticeFermion,
)
from pyquda_utils import core, gamma, source, phase, convert
from pyquda_utils.source import sequential12
from pyquda_comm.array import arrayIdentity, arrayZeros
from pyquda_measurement_utils.tools import mpi_print
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing


class QuarkEMT:
    """Quark EMT measurements.

    This class contains two logically different calculations:

    - ``flowed_fermionic_1pt``
        Stochastic 1-point functions used to estimate local quark EMT operators
        after fermion/gauge flow.

    - ``connected_3pt``
        Connected 3-point functions with an insertion of the quark EMT operator,
        using a fixed-sink sequential propagator.

    The gamma matrices are stored once as class attributes because the same
    objects are used in several contractions.
    """

    # Euclidean gamma_mu used in the EMT insertion.
    D_gammas = [
        cp.asarray(gamma.gamma(1)),
        cp.asarray(gamma.gamma(2)),
        cp.asarray(gamma.gamma(4)),
        cp.asarray(gamma.gamma(8)),
    ]

    # gamma_5.  Please keep this consistent with the convention in your code base.
    G5 = cp.asarray(gamma.gamma(15))

    def __init__(self, parameters):
        """Initialize shared measurement parameters for quark EMT.

        parameters:
          qext      : list of external momenta (q) for three-point functions
          pf        : final state momentum (usually pi + q)
          p_2pt     : two-point momenta list
          pos_boost : positive boost factors for smeared sources/sinks
          neg_boost : negative boost factors for smeared sources/sinks
          width     : smearing radius/width parameter
          flow_type : flow action type (e.g., "wilson")
          flow_epsion : flow epsilon for integration step size (typo kept for backward compatibility)
          flow_steps  : number of flow steps
        """

        # External momentum list for EMT insertion/momentum projection.
        self.qlist = parameters["qext"]

        # Final-state momentum and reference 2-pt momentum set.
        self.pf = parameters["pf"]  # momentum of final nucleon state; pf = pi + q
        self.pilist = parameters["p_2pt"]  # 2pt momentum

        # Source/sink boosted smearing parameters.
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]

        # Gradient flow parameters.
        self.flow_type = parameters["flow_type"]
        self.flow_epsion = parameters["flow_epsion"]
        self.flow_steps = parameters["flow_steps"]

    @staticmethod
    def make_zn_noise_fermion(latt_info, n: int = 2) -> LatticeFermion:
        """Create a single stochastic fermion source with Z_n phases.

        Physically, this is the stochastic source

            xi(x) in {exp(2 pi i k / n)}

        at each spin-color-space-time component.  In the stochastic 1pt method,
        one solves

            D eta = xi,

        and then forms bilinears such as

            xi^dagger Gamma D_mu eta.

        Parameters
        ----------
        latt_info : lattice geometry object
        n         : order of Z_n noise, e.g. n=2 or n=4
        """
        xi = LatticeFermion(latt_info)
        r = cp.random.randint(0, n, size=xi.data.shape)
        xi.data[:] = cp.exp(2j * cp.pi * r / n).astype(xi.data.dtype)
        return xi

    @staticmethod
    def _impose_P_Breit_slice(U: LatticeGauge, complex_field, n_max: int, realize: bool = False):
        """Project a local field onto Breit-frame spatial momenta and keep time.

        For each integer momentum label (n_x,n_y,n_z), this routine builds the
        phase

            exp(i q . x),   q = 2 pi (2n_x/L_x, 2n_y/L_y, 2n_z/L_z, 0),

        multiplies the local scalar field, sums over space, and returns the time
        dependence.

        Input ``complex_field`` is assumed to be a *site-wise scalar* with local
        layout ``wtzyx`` (or compatible).  The contraction

            contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field)

        performs the local sum over w,z,y,x and leaves the momentum index q and
        the time index t.  ``gatherLattice`` then combines the contributions from
        different MPI ranks.
        """
        mpi_print(U.latt_info, f"impose_P_Breit_slice n_max = {n_max}")
        Nt = U.latt_info.global_size[3]
        results = np.zeros((n_max + 1, n_max + 1, n_max + 1, Nt), dtype=np.complex128)
        for nx in range(n_max + 1):
            for ny in range(n_max + 1):
                for nz in range(n_max + 1):
                    qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]
                    phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
                    slice_t = core.gatherLattice(
                        contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field).get(),
                        [1, -1, -1, -1],
                    )
                    slice_t = getMPIComm().bcast(slice_t, root=0)
                    results[nx, ny, nz] += slice_t[0]
        return results

    @classmethod
    def _get_Tmunu_symmetrized_P_Breit_slice(
        cls,
        U_f: LatticeGauge,
        xi: LatticeFermion,
        eta: LatticeFermion,
        n_max: int,
    ):
        """Build flowed quark 1pt observables in the stochastic method.

        Physics content
        ---------------
        For the stochastic estimator one uses a random source ``xi`` and its
        solution ``eta = D^{-1} xi``.  The code forms two families of observables:

        1. Scalar bilinears

               CHI[0] ~ xi^dagger eta,
               CHI[1] ~ xi^dagger xi,

           each projected to definite spatial momentum.

        2. EMT-like bilinears

               T_{nu,mu}(q,t) ~ -1/2 * xi^dagger gamma_nu (D_mu - D_{-mu}) eta,

           followed by symmetrization in (mu,nu).

        Notes on the contractions
        -------------------------
        ``dot_xi_eta = contract("etzyxbc,etzyxbc->etzyx", ...)``
            sums over spin/color locally and leaves a scalar field on the lattice.

        ``Y = contract("ab,...bc->...ac", gamma_nu, tmp.data)``
            applies gamma_nu to the spin index of the fermion field ``tmp``.

        ``complex_field = contract("...sc,...sc->...", xi.data.conj(), Y)``
            computes xi^dagger (gamma_nu tmp) site by site.
        """
        Nt = U_f.latt_info.global_size[3]

        # CHI[0] and CHI[1] are momentum-projected scalar bilinears.
        CHI = np.zeros([2, n_max + 1, n_max + 1, n_max + 1, Nt], dtype=np.complex128)
        dot_xi_eta = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), eta.data)
        CHI[0] = cls._impose_P_Breit_slice(U_f, dot_xi_eta, n_max, realize=True)
        dot_xi_xi = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), xi.data)
        CHI[1] = cls._impose_P_Breit_slice(U_f, dot_xi_xi, n_max, realize=True)

        # T_{mu nu}(q,t) after momentum projection.
        Tmunu = np.zeros([4, 4, n_max + 1, n_max + 1, n_max + 1, Nt], dtype=np.complex128)
        U_f.gauge_dirac.loadGauge(U_f)
        for mu in range(4):
            # Symmetric covariant derivative acting on eta.
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            for nu in range(4):
                # Apply gamma_nu to the spin index.
                Y = contract("ab,...bc->...ac", cls.D_gammas[nu], tmp.data)
                # xi^dagger gamma_nu D_mu eta at each site.
                complex_field = contract("...sc,...sc->...", xi.data.conj(), Y)
                Tmunu[nu, mu] += -0.5 * cls._impose_P_Breit_slice(U_f, complex_field, n_max, realize=True)

        # EMT is symmetrized in Lorentz indices in the final observable.
        for mu in range(4):
            for nu in range(mu + 1, 4):
                Tmunu[mu, nu] = (Tmunu[mu, nu] + Tmunu[nu, mu]) / 2
                Tmunu[nu, mu] = Tmunu[mu, nu]

        return Tmunu, CHI

    @classmethod
    def flowed_fermionic_1pt(
        cls,
        gaugePara,
        invPara,
        flowPara,
        randPara,
        datfile: str = "",
        n_max: int = 0,
    ):
        """Compute quark flowed 1pt observables with stochastic sources.

        Workflow
        --------
        1. Build a multigrid Dirac solver.
        2. For each stochastic vector xi, solve eta = D^{-1} xi.
        3. Copy the gauge field and impose anti-periodic temporal BC for flowed
           fermions.
        4. At each flow time:
              - measure CHI and T_{mu nu}
              - flow xi and eta together using the same flowed gauge field
        5. Average over noise vectors and save the momentum-projected arrays.
        """
        Nv, n_input, randseed = randPara
        a, conf_id, U = gaugePara
        stepsize, Nsteps, improve, division = flowPara
        latt_info = U.latt_info

        global_size = latt_info.global_size
        Ns3 = global_size[0] * global_size[1] * global_size[2]
        Nt = global_size[3]

        mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
        dirac = core.getDirac(
            latt_info,
            invPara[0],
            invPara[2],
            5000,
            1.0,
            invPara[1],
            invPara[1],
            [[8, 8, 4, 4]],
        )
        dirac.loadGauge(U)
        mpi_print(latt_info, "Multigrid inverter ready.")

        cp.random.seed(randseed)

        # Per-noise-vector storage before averaging.
        Tmunu = np.zeros([Nv, 4, 4, n_max + 1, n_max + 1, n_max + 1, Nsteps + 1, Nt], dtype=np.complex128)
        CHI = np.zeros([Nv, 2, n_max + 1, n_max + 1, n_max + 1, Nsteps + 1, Nt], dtype=np.complex128)
        for vec_picked in range(Nv):
            mpi_print(U.latt_info, f"vec {vec_picked}")
            xi = cls.make_zn_noise_fermion(latt_info, n=n_input)
            eta = dirac.invert(xi)

            # The flowed fermion fields live on a flowed gauge background.
            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(U_f.latt_info, f"calc Tmunu, step = {step}")
                U_f.gauge_dirac.loadGauge(U_f)
                tmpt, tmps = cls._get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, n_max)
                Tmunu[vec_picked, :, :, :, :, :, step, :] += tmpt
                CHI[vec_picked, :, :, :, :, step, :] += tmps

                if Nsteps > 0 and step < Nsteps:
                    # Flow xi and eta simultaneously so they remain on the same
                    # flowed gauge background U_f(t).
                    temp = core.MultiLatticeFermion(U.latt_info, 2, cp.array([xi.data, eta.data]))
                    temp_flow = U_f.gradientFlow(temp, "wilson", 1, stepsize)
                    xi, eta = temp_flow[0], temp_flow[1]

        mpi_print(U.latt_info, "random vectors done.")

        np.save(f"{datfile}/cTmunu_pervec.pyquda.npy", Tmunu)
        np.save(f"{datfile}/cCHI_pervec.pyquda.npy", CHI)

        Tmunu = np.mean(Tmunu, axis=0) / Ns3
        CHI = np.mean(CHI, axis=0) / Ns3
        for mu in range(4):
            for nu in range(mu, 4):
                np.save(f"{datfile}/cT{mu+1}{nu+1}.pyquda.npy", Tmunu[mu, nu])
        np.save(f"{datfile}/cCHI.pyquda.npy", CHI)

        return Tmunu, CHI

    @staticmethod
    def _covdev_sym_prop(U_f: LatticeGauge, prop: LatticePropagator, mu: int):
        """Apply the symmetric covariant derivative to a propagator.

        A propagator carries 12 source spin-color columns.  PyQUDA's covDev acts
        naturally on a single fermion field, so the code:

            propagator -> MultiLatticeFermion -> act on each column -> propagator

        The result is

            1/2 (D_{+mu} - D_{-mu}) S .
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

    @classmethod
    def _left_covdev_dst2_from_dsty(cls, U_f: LatticeGauge, dst_y: LatticePropagator, mu: int):
        """Construct the left-acting derivative on ``dst2 = gamma5 S^dagger gamma5``.

        If

            dst2 = gamma5 * dst_y^dagger * gamma5,

        then a left derivative on ``dst2`` can be built from the derivative of
        ``dst_y`` followed by Hermitian conjugation and the same gamma5 wrapping.
        The returned object has propagator-like index structure ``wtzyxadij``.
        """
        D_y = cls._covdev_sym_prop(U_f, dst_y, mu)
        D_y_dag = D_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7)
        leftD_dst2 = contract("ab,wtzyxbcij,cd->wtzyxadij", cls.G5, D_y_dag, cls.G5)
        return leftD_dst2

    @staticmethod
    def _flow_two_props_pyquda(U_f: LatticeGauge, dst_y: LatticePropagator, dst_seq: LatticePropagator, stepsize: float, flow_type: str = "wilson"):
        """Flow two propagators simultaneously on the same flowed gauge background.

        The propagators are packed into one MultiLatticeFermion object so that a
        single call to ``gradientFlow`` advances both objects together.
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

        mf_y_flow = MultiLatticeFermion(U_f.latt_info, L5_y, packed_flow.data[:L5_y].copy())
        mf_seq_flow = MultiLatticeFermion(U_f.latt_info, L5_seq, packed_flow.data[L5_y:L5_y + L5_seq].copy())

        dst_y_flow = convert.multiFermionToPropagator(mf_y_flow)
        dst_seq_flow = convert.multiFermionToPropagator(mf_seq_flow)
        return dst_y_flow, dst_seq_flow

    @classmethod
    def get_C3_chi(cls, U_f: LatticeGauge, dst_y: LatticePropagator, dst_seq: LatticePropagator, t0: int):
        """Compute the connected 3pt scalar insertion C3_chi(t).

        The basic object is

            dst2 = gamma5 * dst_y^dagger * gamma5 ,

        which is the standard backward-line object appearing in meson
        contractions.  The contraction

            contract("wtzyxabij,wtzyxbaji->t", dst2, dst_seq.data)

        does the local trace over spin/color and the local sum over w,z,y,x,
        leaving only the time dependence t.  ``gatherLattice`` then combines MPI
        ranks, and the result is rolled by ``-t0`` so the source sits at time 0.
        """
        dst2 = contract(
            "ab,wtzyxbcij,cd->wtzyxadij",
            cls.G5,
            dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
            cls.G5,
        )
        scalar_t = contract("wtzyxadij,wtzyxdaji->t", dst2, dst_seq.data)
        slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.array(slice_t.real), -t0)

    @classmethod
    def get_C3_Tmunu_symmetrized(cls, U_f: LatticeGauge, dst_y: LatticePropagator, dst_seq: LatticePropagator, t0: int):
        """Compute the connected quark EMT 3pt function C3_{mu nu}(t).

        The operator insertion is built from two pieces:

        1. ``+ 1/2 Tr[ dst2 * gamma_nu * D_mu(dst_seq) ]``
        2. ``- 1/2 Tr[ (left D_mu dst2) * gamma_nu * dst_seq ]``

        These are the two standard terms obtained when the symmetric derivative in
        the EMT acts on the forward and backward quark lines.

        Important contractions
        ----------------------
        ``gamma_D_seq = contract("ab,wtzyxbdij->wtzyxadij", ...)``
            left-multiplies the propagator by gamma_nu in spin space.

        ``contract("wtzyxadij,wtzyxdaji->t", ...)``
            performs the local spin-color trace and the local sum over w,z,y,x,
            leaving only t on the current MPI rank.
        """
        Nt = U_f.latt_info.global_size[3]
        C3_Tmunu = np.zeros((4, 4, Nt), dtype=np.float64)

        dst2 = contract(
            "ab,wtzyxbcij,cd->wtzyxadij",
            cls.G5,
            dst_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
            cls.G5,
        )

        # First derivative term: derivative acts on the forward/sequential line.
        for mu in range(4):
            D_seq = cls._covdev_sym_prop(U_f, dst_seq, mu)
            for nu in range(4):
                gamma_D_seq = contract("ab,wtzyxbdij->wtzyxadij", cls.D_gammas[nu], D_seq.data)
                scalar_t = 0.5 * contract("wtzyxadij,wtzyxdaji->t", dst2, gamma_D_seq)
                slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

        # Second derivative term: derivative acts on the backward line.
        for mu in range(4):
            leftD_dst2 = cls._left_covdev_dst2_from_dsty(U_f, dst_y, mu)
            for nu in range(4):
                gamma_dst_seq = contract("ab,wtzyxbdij->wtzyxadij", cls.D_gammas[nu], dst_seq.data)
                scalar_t = -0.5 * contract("wtzyxadij,wtzyxdaji->t", leftD_dst2, gamma_dst_seq)
                slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[mu, nu] += np.roll(np.array(slice_t.real), -t0)

        # Enforce T_{mu nu} = T_{nu mu} at the measured level.
        for mu in range(4):
            for nu in range(mu + 1, 4):
                C3_Tmunu[mu, nu] = 0.5 * (C3_Tmunu[mu, nu] + C3_Tmunu[nu, mu])
                C3_Tmunu[nu, mu] = C3_Tmunu[mu, nu]

        return C3_Tmunu

    @classmethod
    def connected_3pt(
        cls,
        gaugePara,
        invPara,
        flowPara,
        smearPara,
        Nsrc,
        sinkt_range,
        spin,
        datfile,
    ):
        """Compute connected quark EMT 3pt functions with a fixed-sink method.

        High-level algorithm
        --------------------
        For each source position x0 and each sink separation t_sink:

        1. Build a point source propagator and invert to obtain ``dst_x``.
        2. Optionally smear source and sink.
        3. Build a sequential source on the chosen sink time slice.
        4. Invert again to obtain the sequential propagator ``dst_seq``.
        5. Starting from the unflowed gauge field, flow both ``dst_y`` and
           ``dst_seq`` together and measure C2, C3_chi, and C3_Tmunu at each flow
           time.

        Notes on special objects
        ------------------------
        ``dst_x``
            forward point-to-all propagator from x0.

        ``dst_y_back``
            copy of the unsmeared forward propagator used on the backward line in
            the insertion contraction.

        ``src_seq``
            fixed-sink sequential source.  The precise gamma structure here is
            kept the same as the user's current implementation.
        """
        assert spin in [0, 1, 2, 5]
        N_sinkt = len(sinkt_range)

        a, conf_id, U = gaugePara
        stepsize, Nsteps, improve, division = flowPara
        to_sm_src, to_sm_dst, sm_sigma, sm_steps = smearPara
        latt_info = U.latt_info
        Nx, Ny, Nz, Nt = latt_info.global_size

        if improve:
            raise NotImplementedError("Current PyQUDA flow path here only implements Wilson flow; improve=True not yet wired.")

        mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
        dirac = core.getDirac(
            latt_info,
            invPara[0],
            invPara[2],
            5000,
            1.0,
            invPara[1],
            invPara[1],
            [[8, 8, 4, 4]],
        )
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
                src = boosted_smearing(src, w=sm_sigma, boost=[0, 0, 0])
                mpi_print(latt_info, "source smearing ends")

            # Point-to-all propagator from the chosen source position.
            dirac.loadGauge(U)
            dst_x = core.invertPropagator(dirac, src, 1, 0)
            del src

            # This copy is used later for the backward line in the 3pt contraction.
            dst_y_back = dst_x.copy()

            if to_sm_dst:
                mpi_print(latt_info, "first sink smearing starts")
                dst_x = boosted_smearing(dst_x, w=sm_sigma, boost=[0, 0, 0])

            # 2pt correlator.  The contraction structure is kept as in the user's
            # current working version; comments are left intentionally conservative
            # here because the exact gamma-index convention is user-specific.
            bw_prop = contract("ij, wtzyxilab, kl -> wtzyxkjba", cls.G5, dst_x.data.conj(), cls.G5)
            bw_prop = contract("wtzyxjicf, im -> wtzyxjmcf", bw_prop, cls.G5)
            scalar_t = contract("wtzyxjiab, wtzyxilba, lj -> t", bw_prop, dst_x.data, cls.G5)
            slice_t = core.gatherLattice(scalar_t.get(), [0, -1, -1, -1])
            slice_t = getMPIComm().bcast(slice_t, root=0)
            C2[n_src] += np.roll(slice_t.real, -t0)

            if to_sm_dst:
                mpi_print(latt_info, "second sink smearing starts")
                dst_x = boosted_smearing(dst_x, w=sm_sigma, boost=[0, 0, 0])
                mpi_print(latt_info, "sink smearings end")

            for n_t, sink_t in enumerate(sinkt_range):
                mpi_print(latt_info, f"create sequential source sink_t = {sink_t}")

                # Pick the sink time slice t = t0 + sink_t and build the sequential source.
                t_sink = (sink_t + t0) % Nt
                src_seq_sliced = sequential12(dst_x, t_sink)

                # The gamma structure below is kept identical to the user's current
                # implementation.  In physics terms this step inserts the sink
                # interpolating-operator Dirac structure before the second inversion.
                src_seq_data = contract("ij, wtzyxilab, kl -> wtzyxjkab", cls.G5, src_seq_sliced.data, cls.G5)
                src_seq = LatticePropagator(latt_info)
                src_seq.data = src_seq_data

                # Sequential inversion.
                dirac.loadGauge(U)
                dst_seq_py = core.invertPropagator(dirac, src_seq, 1, 0)
                del src_seq, src_seq_data

                dst_y_py = dst_y_back.copy()
                U_f = U.copy()
                U_f.setAntiPeriodicT()

                for step in range(Nsteps + 1):
                    mpi_print(latt_info, f"contraction for step {step}")
                    C3_chi[n_src, n_t, step] += cls.get_C3_chi(U_f, dst_y_py, dst_seq_py, t0)
                    C3_Tmunu[n_src, n_t, step] += cls.get_C3_Tmunu_symmetrized(U_f, dst_y_py, dst_seq_py, t0)

                    if step < Nsteps:
                        # Advance both propagators and the gauge field to the next
                        # flow time.
                        dst_y_py, dst_seq_py = cls._flow_two_props_pyquda(U_f, dst_y_py, dst_seq_py, stepsize, flow_type="wilson")

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

        return C2, C3_chi, C3_Tmunu


class GluonEMT:
    """Gluon EMT measurements: flowed gluonic 1pt functions.

    The gluonic EMT here is built from a flowed clover field strength

        F_{mu nu}(t, x),

    projected to its anti-Hermitian traceless part.  The measured local building
    block is

        sum_{rho != mu,nu} Tr_c[ F_{mu rho}(x) F_{nu rho}(x) ],

    which is then projected to spatial momentum and summed over the spatial
    volume.
    """

    def __init__(self, parameters):
        """Initialize the same shared parameters used by quark EMT routines.

        This constructor is intentionally compatible with `QuarkEMT` to allow
        shared experiment configuration objects.
        """

        # External momentum list for momentum-projected gluonic observables.
        self.qlist = parameters["qext"]

        # Final-state momentum and two-point momentum set.
        self.pf = parameters["pf"]  # momentum of final nucleon state; pf = pi + q
        self.pilist = parameters["p_2pt"]  # 2pt momentum

        # Source/sink boosted smearing parameters (if used in mixed workflows).
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]

        # Gradient flow parameters.
        self.flow_type = parameters["flow_type"]
        self.flow_epsion = parameters["flow_epsion"]
        self.flow_steps = parameters["flow_steps"]

    @staticmethod
    def _F_clover_traceless(U: LatticeGauge, mu: int, nu: int):
        """Construct the traceless clover field strength F_{mu nu}.

        Steps
        -----
        1. Build the four 1x1 plaquettes around the (mu,nu) plane.
        2. Average them to form the clover combination.
        3. Project to the anti-Hermitian traceless Lie-algebra element.
        4. Multiply by -i so the result matches the usual Euclidean field-strength
           convention used in the user's original code.
        """
        loops_one = [
            [mu, nu, mu + 4, nu + 4],
            [nu, mu + 4, nu + 4, mu],
            [mu + 4, nu + 4, mu, nu],
            [nu + 4, mu, nu, mu + 4],
        ]

        F = U.loop([loops_one] * 4, coeff=[1.0, 1.0, 1.0, 1.0])
        data = F.data
        A = 0.125 * (data - data.swapaxes(-2, -1).conjugate())

        Nc = A.shape[-1]
        trA = contract("...ii->...", A)
        I = arrayIdentity(Nc, A.dtype, F.location)
        A -= trA[..., None, None] * I / Nc
        data[...] = (-1j) * A
        return F

    @classmethod
    def _all_F_clover_traceless(cls, U: LatticeGauge):
        """Build all independent F_{mu nu} and fill the antisymmetric table.

        The output is a 4x4 list with

            F[mu][nu] = - F[nu][mu].
        """
        F = [[None] * 4 for _ in range(4)]
        planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for mu, nu in planes:
            F_mu_nu = cls._F_clover_traceless(U, mu, nu).data[0]
            F[mu][nu] = F_mu_nu
            F[nu][mu] = -F_mu_nu
        return F

    @classmethod
    def flowed_1pt(
        cls,
        U: LatticeGauge,
        stepsize: float = 0.1,
        Nsteps: int = 20,
        datfile: str = "",
        n_max: int = 0,
        improve: bool = False,
    ):
        """Compute flowed gluon 1pt EMT observables.

        Workflow
        --------
        For each flow step:

        1. Build all clover field strengths F_{mu nu}(t,x).
        2. For each EMT component, form

               sum_{rho != mu,nu} Tr_c[F_{mu rho}(x) F_{nu rho}(x)]

           as a site-wise scalar field.
        3. Project to Breit-frame spatial momenta with phase factors.
        4. Sum over the spatial volume and store the time dependence.
        5. Advance the gauge field by Wilson flow.
        """
        latt_info = U.latt_info
        global_size = latt_info.global_size
        Lx, Ly, Lz, Lt = latt_info.size
        Ns3 = global_size[0] * global_size[1] * global_size[2]

        Tmunu_t = np.zeros(
            (4, 4, n_max + 1, n_max + 1, n_max + 1, Nsteps + 1, global_size[3]),
            dtype=np.complex128,
        )

        U_flow = U.copy()

        for step in range(Nsteps + 1):
            mpi_print(latt_info, f"step {step} calculate F")
            F = cls._all_F_clover_traceless(U_flow)
            mpi_print(latt_info, f"step {step} calculate T")

            for mu in range(4):
                for nu in range(mu, 4):
                    # Local scalar field that will become the EMT building block.
                    tmp = arrayZeros((2, Lt, Lz, Ly, Lx // 2), U.data.dtype, U.location)

                    for rho in range(4):
                        if rho == mu or rho == nu:
                            continue
                        F_mr = F[mu][rho]
                        F_nr = F[nu][rho]
                        # Color trace Tr_c[F_{mu rho} F_{nu rho}] at each site.
                        tmp += contract("...ab,...ba->...", F_mr, F_nr)

                    for nx in range(n_max + 1):
                        for ny in range(n_max + 1):
                            for nz in range(n_max + 1):
                                qext_xyz = [[2 * nx, 2 * ny, 2 * nz]]
                                phases_3pt = phase.MomentumPhase(U.latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
                                slice_t = core.gatherLattice(
                                    contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get(),
                                    [1, -1, -1, -1],
                                )
                                if U.latt_info.mpi_rank == 0:
                                    # Factor 2 is kept from the user's original
                                    # normalization convention.
                                    Tmunu_t[mu, nu, nx, ny, nz, step, :] += 2.0 * slice_t[0]

            if Nsteps > 0:
                if step == 0:
                    mpi_print(latt_info, f"wilsonFlow step = {step}")
                    # First step is subdivided into 10 smaller flow steps, again
                    # following the user's original implementation.
                    U_flow.wilsonFlow(10, epsilon=stepsize / 10)
                elif step < Nsteps:
                    mpi_print(latt_info, f"wilsonFlow step = {step}")
                    U_flow.wilsonFlow(1, epsilon=stepsize)

        Tmunu_t /= Ns3
        for mu in range(4):
            for nu in range(mu, 4):
                suffix = f".T{mu+1}{nu+1}.n_max{n_max}.pyquda.npy"
                np.save(datfile + suffix, Tmunu_t[mu, nu])

        return Tmunu_t
