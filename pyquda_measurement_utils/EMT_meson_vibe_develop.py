"""Meson EMT measurement formulas and conventions.

This module implements flowed quark/gluon EMT observables for meson matrix
elements.  The code currently uses the "convention B" connected-meson
contraction with ``meson_sign = +1``.

Notation
--------
``S_f`` denotes the ordinary forward quark propagator from the source to the
insertion point.  ``S_seq`` denotes the fixed-sink sequential propagator after
inverting the meson sequential source.  A backward meson line is built through
gamma5 hermiticity,

    S_b(x) = gamma5 * S_seq(x)^dagger * gamma5.

Momentum projection is applied as

    C(q, t) = sum_x exp(i q dot (x - x_src)) C(x, t),

using the phase convention provided by ``pyquda_utils.phase.MomentumPhase``.

Inversion path
--------------
For connected meson 3pt functions, the code:

1. Builds point sources at ``src_pos`` and optionally applies Gaussian/boosted
   source smearing.
2. Inverts the Dirac operator to get source-smeared point-sink forward and
   backward propagators, ``S_f`` and ``S_b``.  If the forward/backward source
   smearing is identical, only one inversion is used and the propagator is
   copied.
3. Applies sink smearing to the forward propagator used by the fixed-sink
   sequential source.
4. Builds the meson sequential source at
   ``t_sink = src_t + t_sep`` with sink momentum ``pf`` and spin structure

       gamma_seq = gamma5 * Gamma_sink^dagger * gamma5.

5. Inverts this sequential source to obtain ``S_seq``.

Meson 2pt contraction
---------------------
The meson 2pt function scans all 16 sink gamma structures while keeping the
source gamma fixed:

    C2[Gamma_sink, p, t]
      = sum_x exp(-i p dot (x - x_src))
        Tr_sc[ Gamma_sink S_b(x, src) Gamma_src S_f(x, src) ],

where the backward line is reconstructed with gamma5 hermiticity from the
anti-quark propagator.  This is the same structural convention used for the
pion two-point function.

Connected quark 3pt contractions
--------------------------------
The scalar connected insertion is

    C3_chi(q, t)
      = sum_x exp(i q dot (x - x_src))
        Tr_sc[ S_b(x) S_f(x) Gamma_src ].

The quark EMT insertion is implemented as the symmetrized Euclidean bilinear

    T_{mu nu}^q = 1/2 * [ gamma_nu D_mu - left_D_mu gamma_nu ],

where ``D_mu`` is the symmetric covariant derivative acting on the forward
line and ``left_D_mu`` is the corresponding left-acting derivative on the
backward sequential line.  In code this is evaluated as

    +1/2 Tr_sc[ S_b(x) gamma_nu D_mu S_f(x) Gamma_src ]
    -1/2 Tr_sc[ (left_D_mu S_b)(x) gamma_nu S_f(x) Gamma_src ],

then projected to momentum ``q`` and symmetrized under ``mu <-> nu``.

Stochastic quark 1pt contraction
--------------------------------
For random noise ``xi`` and solution ``eta = D^{-1} xi``, the code estimates

    CHI[0](q, t) = sum_x exp(i q dot x) xi^dagger(x) eta(x),
    CHI[1](q, t) = sum_x exp(i q dot x) xi^dagger(x) xi(x),

and the flowed quark EMT building block

    T_{nu mu}^q(q, t)
      = -1/2 sum_x exp(i q dot x)
        xi^dagger(x) gamma_nu [D_{+mu} - D_{-mu}] eta(x),

followed by symmetrization in ``mu`` and ``nu`` and averaging over noise
vectors.  The overall volume normalization is applied after the noise average.

Gluon 1pt contraction
---------------------
The gluon field strength is a traceless anti-Hermitian clover operator with an
extra factor of ``-i`` to match the original Euclidean convention.  The measured
building block is

    T_{mu nu}^g(q, t)
      = 2 / V3 * sum_x exp(i q dot x)
        sum_{rho != mu,nu} Tr_c[ F_{mu rho}(x) F_{nu rho}(x) ].

This code intentionally measures the full gluonic EMT building block used for
gradient-flow renormalization.  It does not impose a traceless projection at
the EMT level.  The traceless projection in ``_F_clover_traceless`` is only the
standard projection of each clover field-strength matrix onto the su(3) gauge
algebra.

Gradient flow convention
------------------------
For every observable, the code measures first and then advances the fields to
the next flow time.  Therefore output index ``step = 0`` is the unflowed
measurement, and later indices correspond to flowed fields.  Gauge links and
fermion fields are flowed together so contractions at a given output index use
a common flowed background.

Gradient flow smooths UV fluctuations at flow radius roughly ``sqrt(8 t)``.
The EMT observables are intended to be analyzed as flowed composite operators:
one studies their flow-time dependence and applies the appropriate
small-flow-time/renormalization treatment outside this contraction kernel.

Formula review notes
--------------------
The connected meson 2pt and 3pt contractions are internally consistent with the
current convention B + ``meson_sign = +1`` choice, and q=0 regression tests have
matched the previous baseline to roundoff after refactors.

The quark and gluon flow schedules are aligned: because the code measures
before each flow update, the first interval is subdivided into 10 small flow
steps immediately after measuring ``step == 0``.  Thus output index ``step`` is
intended to correspond to flow time ``step * flow_epsilon``.
"""

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
from pyquda_comm.array import arrayIdentity, arrayZeros
from pyquda_measurement_utils.io_corr import (
    save_emt_quark_1pt_hdf5,
    save_emt_quark_3pt_hdf5,
    save_emt_meson_2pt_hdf5,
    save_emt_gluon_1pt_hdf5,
)
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_meson_bw_seq_pyquda

_VALID_FLOW_TYPES = {"wilson", "symanzik"}
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]
my_pyquda_gammas = [gamma.gamma(idx) for idx in pyquda_gammas_order]
D_GAMMA_IDS = [1, 2, 4, 8]
D_gammas = [gamma.gamma(idx) for idx in D_GAMMA_IDS]
G5 = gamma.gamma(15)


def _normalize_flow_type(flow_type: str) -> str:
    flow = str(flow_type).strip().lower()
    if flow not in _VALID_FLOW_TYPES:
        raise ValueError(f"flow_type should be one of {_VALID_FLOW_TYPES}, got {flow_type!r}")
    return flow

"""
================================================================================
                                  QuarkEMT
================================================================================
"""
class QuarkEMT:
    def __init__(self, parameters):
        self.qlist = parameters["qext"]
        self.pf = parameters["pf"]
        self.pilist = parameters["p_2pt"]

        self.CG_GaussSmear = parameters.get("CG_GaussSmear", False)
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]

        self.flow_type = _normalize_flow_type(parameters["flow_type"])
        self.flow_epsilon = parameters["flow_epsilon"]
        self.flow_steps = parameters["flow_steps"]

    @staticmethod
    def _gamma5_for(ref_arr):
        return _asarray_on_queue(G5, _get_xp_from_array(ref_arr), ref_arr)

    @staticmethod
    def _gamma_stack_for(ref_arr):
        return _asarray_on_queue(my_pyquda_gammas, _get_xp_from_array(ref_arr), ref_arr)

    @staticmethod
    def _dirac_gammas_for(ref_arr):
        return _asarray_on_queue(D_gammas, _get_xp_from_array(ref_arr), ref_arr)

    @classmethod
    def _get_interpolator_gamma_for(cls, interpolator, ref_arr):
        if interpolator not in my_gammas:
            raise ValueError(f"Unsupported interpolator {interpolator!r}. Expected one of {my_gammas}.")
        return _asarray_on_queue(my_pyquda_gammas[my_gammas.index(interpolator)], _get_xp_from_array(ref_arr), ref_arr)

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
    def _impose_P_Breit_slice(U: LatticeGauge, complex_field, phases_3pt):
        """Project a local field onto Breit-frame spatial momenta and keep time.

        Input ``complex_field`` is assumed to be a *site-wise scalar* with local
        layout ``wtzyx`` (or compatible).  The prebuilt ``phases_3pt`` carries all
        requested spatial momenta.  The contraction

            contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field)

        performs the local sum over w,z,y,x and leaves the momentum index q and
        the time index t.  ``gatherLattice`` then combines the contributions from
        different MPI ranks.
        """
        slice_t = core.gatherLattice(
            contract("qwtzyx, wtzyx -> qt", phases_3pt, complex_field).get(),
            [1, -1, -1, -1],
        )
        return getMPIComm().bcast(slice_t, root=0)

    def _get_Tmunu_symmetrized_P_Breit_slice(
        self,
        U_f: LatticeGauge,
        xi: LatticeFermion,
        eta: LatticeFermion,
        qlist,
        phases_3pt,
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
        CHI = np.zeros([2, len(qlist), Nt], dtype=np.complex128)
        dot_xi_eta = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), eta.data)
        CHI[0] = self._impose_P_Breit_slice(U_f, dot_xi_eta, phases_3pt)
        dot_xi_xi = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), xi.data)
        CHI[1] = self._impose_P_Breit_slice(U_f, dot_xi_xi, phases_3pt)

        # T_{mu nu}(q,t) after momentum projection.
        Tmunu = np.zeros([4, 4, len(qlist), Nt], dtype=np.complex128)
        U_f.gauge_dirac.loadGauge(U_f)
        D_gammas_local = self._dirac_gammas_for(eta.data)
        for mu in range(4):
            # Symmetric covariant derivative acting on eta.
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            for nu in range(4):
                # Apply gamma_nu to the spin index.
                Y = contract("ab,...bc->...ac", D_gammas_local[nu], tmp.data)
                # xi^dagger gamma_nu D_mu eta at each site.
                complex_field = contract("...sc,...sc->...", xi.data.conj(), Y)
                Tmunu[nu, mu] += -0.5 * self._impose_P_Breit_slice(U_f, complex_field, phases_3pt)

        # EMT is symmetrized in Lorentz indices in the final observable.
        for mu in range(4):
            for nu in range(mu + 1, 4):
                Tmunu[mu, nu] = (Tmunu[mu, nu] + Tmunu[nu, mu]) / 2
                Tmunu[nu, mu] = Tmunu[mu, nu]

        return Tmunu, CHI

    def flowed_fermionic_1pt(
        self,
        gauge: LatticeGauge,
        invPara,
        randPara,
        tag: str = "",
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
        n_vec, n_zn, randseed = randPara
        mass, csw, tol, maxiter = invPara
        U = gauge
        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps
        latt_info = U.latt_info

        global_size = latt_info.global_size
        Ns3 = global_size[0] * global_size[1] * global_size[2]
        Nt = global_size[3]

        mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
        dirac = core.getDirac(
            latt_info,
            mass,
            tol,
            maxiter,
            1.0,
            csw,
            csw,
            [[8, 8, 4, 4]],
        )
        dirac.loadGauge(U)
        mpi_print(latt_info, "Multigrid inverter ready.")

        cp.random.seed(randseed)

        # Per-noise-vector storage before averaging.
        Tmunu = np.zeros([n_vec, 4, 4, len(self.qlist), Nsteps + 1, Nt], dtype=np.complex128)
        CHI = np.zeros([n_vec, 2, len(self.qlist), Nsteps + 1, Nt], dtype=np.complex128)
        qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
        phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
        for vec_picked in range(n_vec):
            mpi_print(U.latt_info, f"vec {vec_picked}")
            xi = self.make_zn_noise_fermion(latt_info, n=n_zn)
            eta = dirac.invert(xi)

            # The flowed fermion fields live on a flowed gauge background.
            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(U_f.latt_info, f"calc Tmunu, step = {step}")
                U_f.gauge_dirac.loadGauge(U_f)
                tmpt, tmps = self._get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, self.qlist, phases_3pt)
                Tmunu[vec_picked, :, :, :, step, :] += tmpt
                CHI[vec_picked, :, :, step, :] += tmps

                if Nsteps > 0 and step == 0:
                    # First step is subdivided into 10 smaller flow steps to better preserve the initial condition.
                    temp = core.MultiLatticeFermion(U.latt_info, 2, cp.array([xi.data, eta.data]))
                    temp_flow = U_f.gradientFlow(temp, self.flow_type, 10, stepsize/10)
                    xi, eta = temp_flow[0], temp_flow[1]

                elif Nsteps > 0 and step < Nsteps:
                    # Flow xi and eta simultaneously so they remain on the same
                    # flowed gauge background U_f(t).
                    temp = core.MultiLatticeFermion(U.latt_info, 2, cp.array([xi.data, eta.data]))
                    temp_flow = U_f.gradientFlow(temp, self.flow_type, 1, stepsize)
                    xi, eta = temp_flow[0], temp_flow[1]

        mpi_print(U.latt_info, "random vectors done.")

        attrs = {
            "measurement": "quark_1pt",
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "n_vec": n_vec,
            "n_zn": n_zn,
        }
        Tmunu_avg = np.mean(Tmunu, axis=0) / Ns3
        CHI_avg = np.mean(CHI, axis=0) / Ns3
        save_emt_quark_1pt_hdf5(tag, Tmunu, CHI, Tmunu_avg, CHI_avg, attrs=attrs)

        return Tmunu_avg, CHI_avg

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
    def _make_dst2(cls, prop: LatticePropagator):
        """Build the backward meson line gamma5 * prop^dagger * gamma5."""
        G5_local = cls._gamma5_for(prop.data)
        return contract(
            "ab,wtzyxbcij,cd->wtzyxadij",
            G5_local,
            prop.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
            G5_local,
        )

    @classmethod
    def _left_covdev_dst2_from_prop(cls, U_f: LatticeGauge, prop: LatticePropagator, mu: int):
        """Construct the left-acting derivative on ``dst2 = gamma5 S^dagger gamma5``.

        If

            dst2 = gamma5 * prop^dagger * gamma5,

        then a left derivative on ``dst2`` can be built from the derivative of
        ``prop`` followed by Hermitian conjugation and the same gamma5 wrapping.
        The returned object has propagator-like index structure ``wtzyxadij``.
        """
        D_y = cls._covdev_sym_prop(U_f, prop, mu)
        D_y_dag = D_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7)
        G5_local = cls._gamma5_for(prop.data)
        leftD_dst2 = contract("ab,wtzyxbcij,cd->wtzyxadij", G5_local, D_y_dag, G5_local)
        return leftD_dst2

    @staticmethod
    def _flow_two_props_pyquda(U_f: LatticeGauge, prop_a: LatticePropagator, prop_b: LatticePropagator, stepsize: float, Nsteps: int, flow_type: str = "wilson"):
        """Flow two propagators simultaneously on the same flowed gauge background.

        The propagators are packed into one MultiLatticeFermion object so that a
        single call to ``gradientFlow`` advances both objects together.
        """
        mf_a = convert.propagatorToMultiFermion(prop_a)
        mf_b = convert.propagatorToMultiFermion(prop_b)

        L5_a = mf_a.L5
        L5_b = mf_b.L5
        assert L5_a == L5_b

        packed = MultiLatticeFermion(
            U_f.latt_info,
            L5_a + L5_b,
            cp.concatenate([mf_a.data, mf_b.data], axis=0),
        )

        packed_flow = U_f.gradientFlow(packed, flow_type, Nsteps, stepsize)

        mf_a_flow = MultiLatticeFermion(U_f.latt_info, L5_a, packed_flow.data[:L5_a].copy())
        mf_b_flow = MultiLatticeFermion(U_f.latt_info, L5_b, packed_flow.data[L5_a:L5_a + L5_b].copy())

        prop_a_flow = convert.multiFermionToPropagator(mf_a_flow)
        prop_b_flow = convert.multiFermionToPropagator(mf_b_flow)
        return prop_a_flow, prop_b_flow

    def _advance_flowed_props(self, U_f, prop_fw_flow, seq_bw_prop_flow, step, stepsize, Nsteps):
        """Advance the flowed propagators using the existing quark-flow schedule."""
        if Nsteps > 0 and step == 0:
            return self._flow_two_props_pyquda(
                U_f,
                prop_fw_flow,
                seq_bw_prop_flow,
                stepsize / 10,
                Nsteps=10,
                flow_type=self.flow_type,
            )
        if Nsteps > 0 and step < Nsteps:
            return self._flow_two_props_pyquda(
                U_f,
                prop_fw_flow,
                seq_bw_prop_flow,
                stepsize,
                Nsteps=1,
                flow_type=self.flow_type,
            )
        return prop_fw_flow, seq_bw_prop_flow

    def _make_meson_source_props(self, dirac, U, src_pos):
        """Build source-smeared point-sink forward/backward meson propagators."""
        latt_info = U.latt_info
        src_fw = source.propagator(latt_info, "point", src_pos)

        if self.CG_GaussSmear and self.pos_boost == self.neg_boost:
            mpi_print(latt_info, f"source smearing starts, boost = {self.pos_boost}")
            src_fw = boosted_smearing(src_fw, w=self.width, boost=self.pos_boost)
            mpi_print(latt_info, "source smearing ends")

            dirac.loadGauge(U)
            prop_fw_SP = core.invertPropagator(dirac, src_fw, 1, 0)
            prop_bw_SP = prop_fw_SP.copy()
            del src_fw
            return prop_fw_SP, prop_bw_SP

        if self.CG_GaussSmear:
            mpi_print(latt_info, f"forward source smearing starts, boost = {self.pos_boost}")
            src_fw = boosted_smearing(src_fw, w=self.width, boost=self.pos_boost)
            mpi_print(latt_info, "forward source smearing ends")

            src_bw = source.propagator(latt_info, "point", src_pos)
            mpi_print(latt_info, f"backward source smearing starts, boost = {self.neg_boost}")
            src_bw = boosted_smearing(src_bw, w=self.width, boost=self.neg_boost)
            mpi_print(latt_info, "backward source smearing ends")

            dirac.loadGauge(U)
            prop_fw_SP = core.invertPropagator(dirac, src_fw, 1, 0)
            prop_bw_SP = core.invertPropagator(dirac, src_bw, 1, 0)
            del src_fw, src_bw
            return prop_fw_SP, prop_bw_SP

        dirac.loadGauge(U)
        prop_fw_SP = core.invertPropagator(dirac, src_fw, 1, 0)
        prop_bw_SP = prop_fw_SP.copy()
        del src_fw
        return prop_fw_SP, prop_bw_SP

    @classmethod
    def get_C3_chi(
        cls,
        U_f: LatticeGauge,
        prop_fw: LatticePropagator,
        seq_bw_prop: LatticePropagator,
        src_gamma,
        phases_3pt,
        t0: int,
    ):
        """Compute C3_chi with the meson sequential-source layout.

        The ordinary forward propagator is the source-to-insertion line, and the
        sequential backward propagator is the sink-to-insertion line.  The
        backward line is formed as

            dst2 = gamma5 * seq_bw_prop^dagger * gamma5

        before tracing with the source interpolator.
        """
        dst2 = cls._make_dst2(seq_bw_prop)
        scalar_field = contract("wtzyxabij,wtzyxbcji,ca->wtzyx", dst2, prop_fw.data, src_gamma)
        slice_t = core.gatherLattice(
            contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
            [1, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.array(slice_t), -t0, axis=-1)

    @classmethod
    def get_C3_Tmunu_symmetrized(
        cls,
        U_f: LatticeGauge,
        prop_fw: LatticePropagator,
        seq_bw_prop: LatticePropagator,
        src_gamma,
        phases_3pt,
        t0: int,
    ):
        """Compute connected quark EMT 3pt functions in the meson convention."""
        Nq = len(phases_3pt)
        Nt = U_f.latt_info.global_size[3]
        C3_Tmunu = np.zeros((Nq, 4, 4, Nt), dtype=np.complex128)

        dst2 = cls._make_dst2(seq_bw_prop)
        D_gammas_local = cls._dirac_gammas_for(prop_fw.data)

        # First derivative term: derivative acts on the forward quark line.
        for mu in range(4):
            D_fw = cls._covdev_sym_prop(U_f, prop_fw, mu)
            for nu in range(4):
                gamma_D_fw = contract("ab,wtzyxbdij->wtzyxadij", D_gammas_local[nu], D_fw.data)
                scalar_field = 0.5 * contract(
                    "wtzyxabij,wtzyxbcji,ca->wtzyx",
                    dst2,
                    gamma_D_fw,
                    src_gamma,
                )
                slice_t = core.gatherLattice(
                    contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
                    [1, -1, -1, -1],
                )
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[:, mu, nu] += np.roll(np.array(slice_t), -t0, axis=-1)

        # Second derivative term: derivative acts on the sequential backward line.
        for mu in range(4):
            leftD_dst2 = cls._left_covdev_dst2_from_prop(U_f, seq_bw_prop, mu)
            for nu in range(4):
                gamma_fw = contract("ab,wtzyxbdij->wtzyxadij", D_gammas_local[nu], prop_fw.data)
                scalar_field = -0.5 * contract(
                    "wtzyxabij,wtzyxbcji,ca->wtzyx",
                    leftD_dst2,
                    gamma_fw,
                    src_gamma,
                )
                slice_t = core.gatherLattice(
                    contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
                    [1, -1, -1, -1],
                )
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[:, mu, nu] += np.roll(np.array(slice_t), -t0, axis=-1)

        # Enforce T_{mu nu} = T_{nu mu} at the measured level.
        for mu in range(4):
            for nu in range(mu + 1, 4):
                C3_Tmunu[:, mu, nu] = 0.5 * (C3_Tmunu[:, mu, nu] + C3_Tmunu[:, nu, mu])
                C3_Tmunu[:, nu, mu] = C3_Tmunu[:, mu, nu]

        return C3_Tmunu

    def contract_meson_2pt(
        self,
        latt_info,
        prop_fw,
        prop_bw,
        src_gamma,
        src_pos,
        tag=None,
        attrs=None,
    ):
        """Contract meson 2pt functions with one source gamma and all sink gammas.

        The contraction follows the pion two-point convention

            Tr[Gamma_sink S_bw Gamma_src S_fw]

        where the backward line is built with gamma5 hermiticity from the
        antiquark propagator.  The sink side scans all 16 gamma structures in
        ``my_gammas`` while the source side keeps the requested
        interpolating gamma fixed.
        """
        if self.CG_GaussSmear:
            mpi_print(latt_info, f"2pt forward sink smearing starts, boost = {self.pos_boost}")
            prop_fw = boosted_smearing(prop_fw, w=self.width, boost=self.pos_boost)
            mpi_print(latt_info, f"2pt backward sink smearing starts, boost = {self.neg_boost}")
            prop_bw = boosted_smearing(prop_bw, w=self.width, boost=self.neg_boost)
            mpi_print(latt_info, "2pt sink smearing ends")

        sink_gammas = self._gamma_stack_for(prop_fw.data)
        G5_local = self._gamma5_for(prop_bw.data)
        p_2pt_xyz = [[-p[0], -p[1], -p[2]] for p in self.pilist]
        phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz, src_pos)

        bw_prop = contract("ij, wtzyxilab, kl -> wtzyxkjba", G5_local, prop_bw.data.conj(), G5_local)
        bw_prop = contract("wtzyxjicf, gim -> gwtzyxjmcf", bw_prop, sink_gammas)
        scalar = contract("gwtzyxjiab, wtzyxilba, lj -> gwtzyx", bw_prop, prop_fw.data, src_gamma)
        C2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> gqt", phases_2pt, scalar).get(), [2, -1, -1, -1])
        C2 = getMPIComm().bcast(C2, root=0)
        C2 = np.roll(np.array(C2), -src_pos[3], axis=-1)

        if tag is not None and latt_info.mpi_rank == 0:
            save_emt_meson_2pt_hdf5(tag, C2, my_gammas, self.pilist, attrs=attrs)
        return C2

    def connected_3pt(
        self,
        gauge: LatticeGauge,
        invPara,
        src_pos,
        t_separations,
        spin,
        tag,
        c2_tag=None,
        src_interpolator="5",
        sink_interpolator="5",
    ):
        """Compute connected quark EMT 3pt functions with a fixed-sink method.

        High-level algorithm
        --------------------
        For a given source position src_pos and each sink separation t_sink:

        1. Build point sources and invert to obtain source-smeared point-sink
           forward/backward propagators.
        2. Optionally apply sink smearing to the forward propagator.
        3. Build and invert the meson fixed-sink sequential source with
           ``create_meson_bw_seq_pyquda``.
        4. Starting from the unflowed gauge field, flow both the forward
           propagator and sequential backward propagator together and measure
           C2, C3_chi(q,t), and C3_Tmunu(q,t) at each flow time.

        Notes on special objects
        ------------------------
        ``prop_fw_SP`` / ``prop_bw_SP``
            source-smeared, point-sink forward/backward propagators.

        ``prop_fw_SS``
            source-smeared and sink-smeared forward propagator used to build
            the fixed-sink sequential source.

        ``seq_bw_prop``
            fixed-sink backward sequential propagator.  The underlying source
            is built by applying the sink momentum phase and the standard meson
            gamma structure gamma5 * Gamma_sink^dagger * gamma5.
        """
        assert spin in [0, 1, 2, 5]
        N_ts = len(t_separations)

        U = gauge
        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps
        latt_info = U.latt_info
        Nt = latt_info.global_size[3]
        mass, csw, tol, maxiter = invPara

        x0, y0, z0, t0 = src_pos
        mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
        dirac = core.getDirac(
            latt_info,
            mass,
            tol,
            maxiter,
            1.0,
            csw,
            csw,
            [[8, 8, 4, 4]],
        )
        dirac.loadGauge(U)
        mpi_print(latt_info, "Multigrid inverter ready.")

        C2 = np.zeros((len(my_gammas), len(self.pilist), Nt), dtype=np.complex128)
        Nq = len(self.qlist)
        C3_chi = np.zeros((N_ts, Nsteps + 1, Nq, Nt), dtype=np.complex128)
        C3_Tmunu = np.zeros((N_ts, Nsteps + 1, Nq, 4, 4, Nt), dtype=np.complex128)

        mpi_print(latt_info, f"src [{x0},{y0},{z0},{t0}]")

        prop_fw_SP, prop_bw_SP = self._make_meson_source_props(dirac, U, src_pos)
        src_gamma = self._get_interpolator_gamma_for(src_interpolator, prop_fw_SP.data)

        c2_attrs = {
            "measurement": "meson_2pt",
            "src_t": t0,
            "src_interpolator": src_interpolator,
            "sink_gamma_scan": "all_16",
        }
        C2 += self.contract_meson_2pt(
            latt_info,
            prop_fw_SP.copy(),
            prop_bw_SP.copy(),
            src_gamma,
            src_pos,
            tag=c2_tag,
            attrs=c2_attrs,
        )
        sink_gamma_idx = my_gammas.index(sink_interpolator)
        zero_mom_idx = self.pilist.index([0, 0, 0, 0]) if [0, 0, 0, 0] in self.pilist else 0
        C2_selected = C2[sink_gamma_idx, zero_mom_idx]

        if self.CG_GaussSmear:
            mpi_print(latt_info, f"first sink smearing starts, boost = {self.pos_boost}")
            prop_fw_SS = boosted_smearing(prop_fw_SP.copy(), w=self.width, boost=self.pos_boost)
        else:
            prop_fw_SS = prop_fw_SP.copy()
        sink_gamma = self._get_interpolator_gamma_for(sink_interpolator, prop_fw_SS.data)

        qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
        phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, src_pos)

        for n_ts, t_sep in enumerate(t_separations):
            mpi_print(latt_info, f"create sequential source sink_t = {t_sep}")

            dirac.loadGauge(U)
            seq_bw_prop = create_meson_bw_seq_pyquda(
                dirac,
                prop_fw_SS,
                src_pos,
                self.pf,
                t_sep,
                sink_gamma,
            )

            prop_fw_flow = prop_fw_SP.copy()
            seq_bw_prop_flow = seq_bw_prop.copy()
            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(latt_info, f"contraction for step {step}")
                C3_chi[n_ts, step] += self.get_C3_chi(U_f, prop_fw_flow, seq_bw_prop_flow, src_gamma, phases_3pt, t0)
                C3_Tmunu[n_ts, step] += self.get_C3_Tmunu_symmetrized(U_f, prop_fw_flow, seq_bw_prop_flow, src_gamma, phases_3pt, t0)

                prop_fw_flow, seq_bw_prop_flow = self._advance_flowed_props(
                    U_f,
                    prop_fw_flow,
                    seq_bw_prop_flow,
                    step,
                    stepsize,
                    Nsteps,
                )

            del U_f, prop_fw_flow, seq_bw_prop_flow, seq_bw_prop

        attrs = {
            "measurement": "quark_3pt",
            "spin": spin,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "n_t_separations": N_ts,
            "src_t": t0,
            "src_interpolator": src_interpolator,
            "sink_interpolator": sink_interpolator,
            "contraction_convention": "B",
            "meson_sign": 1,
            "n_qext": Nq,
            "c2_selected_momentum_index": zero_mom_idx,
            "c2_selected_momentum": self.pilist[zero_mom_idx],
        }
        save_emt_quark_3pt_hdf5(tag, C2_selected, C3_chi, C3_Tmunu, momentum_transfer_list=self.qlist, attrs=attrs)
        return C2, C3_chi, C3_Tmunu


"""
================================================================================
                                  GluonEMT
================================================================================
"""
class GluonEMT:

    def __init__(self, parameters):
        self.qlist = parameters["qext"]
        self.pf = parameters["pf"]
        self.pilist = parameters["p_2pt"]

        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]

        self.flow_type = _normalize_flow_type(parameters["flow_type"])
        self.flow_epsilon = parameters["flow_epsilon"]
        self.flow_steps = parameters["flow_steps"]

    def _advance_flowed_gauge(self, U_flow, step, stepsize, Nsteps):
        """Advance the flowed gauge field using the existing gluon-flow schedule."""
        if Nsteps > 0 and step == 0:
            if self.flow_type == "wilson":
                U_flow.wilsonFlow(10, epsilon=stepsize / 10)
            elif self.flow_type == "symanzik":
                U_flow.symanzikFlow(10, epsilon=stepsize / 10)
        elif Nsteps > 0 and step < Nsteps:
            if self.flow_type == "wilson":
                U_flow.wilsonFlow(1, epsilon=stepsize)
            elif self.flow_type == "symanzik":
                U_flow.symanzikFlow(1, epsilon=stepsize)

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

    def _all_F_clover_traceless(self, U: LatticeGauge):
        """Build all independent F_{mu nu} and fill the antisymmetric table.

        The output is a 4x4 list with

            F[mu][nu] = - F[nu][mu].
        """
        F = [[None] * 4 for _ in range(4)]
        planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for mu, nu in planes:
            F_mu_nu = self._F_clover_traceless(U, mu, nu).data[0]
            F[mu][nu] = F_mu_nu
            F[nu][mu] = -F_mu_nu
        return F

    def flowed_1pt(
        self,
        U: LatticeGauge,
        tag: str = "",
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

        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps

        Tmunu_t = np.zeros(
            (4, 4, len(self.qlist), Nsteps + 1, global_size[3]),
            dtype=np.complex128,
        )

        U_flow = U.copy()
        qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
        phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])

        for step in range(Nsteps + 1):
            mpi_print(latt_info, f"step {step} calculate F")
            F = self._all_F_clover_traceless(U_flow)
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

                    slice_t = core.gatherLattice(
                        contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get(),
                        [1, -1, -1, -1],
                    )
                    if U.latt_info.mpi_rank == 0:
                        # Factor 2 is kept from the user's original normalization convention.
                        Tmunu_t[mu, nu, :, step, :] += 2.0 * slice_t

            mpi_print(latt_info, f"{self.flow_type}Flow step = {step}")
            self._advance_flowed_gauge(U_flow, step, stepsize, Nsteps)

        Tmunu_t /= Ns3
        attrs = {
            "measurement": "gluon_1pt",
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
        }
        save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=attrs)

        return Tmunu_t
