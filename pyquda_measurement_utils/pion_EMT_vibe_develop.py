"""Meson EMT measurement formulas and conventions.

This module implements flowed quark/gluon EMT observables for meson matrix
elements.  The connected meson three-point code uses the "convention B"
contraction with ``meson_sign = +1``.

Common notation
---------------
The source is ``x0 = (t0, x0)``.  The current or EMT insertion is
``x = (tau, x)``.  The fixed sink is ``y = (tsep, y)``.  ``S_q(a, b)`` denotes
the quark propagator from ``b`` to ``a``.  Antiquark-like lines are represented
with gamma5 hermiticity,

    S_anti(a, b) = gamma5 S_q(a, b)^dagger gamma5.

The code uses ``pyquda_utils.phase.MomentumPhase`` for all Fourier phases.  To
avoid hiding sign conventions in prose, write these phases as

    Phi_k(r - x0),

where ``k`` is exactly the momentum list passed to ``MomentumPhase``.

Meson two-point correlation function
------------------------------------
The physical connected meson two-point function with source interpolator
``Gamma_src`` and sink interpolator ``Gamma_sink`` is

    C2_{Gamma_sink}(p, t)
      = sum_x Phi_{-p}(x - x0)
        Tr_sc[
            S_anti(x, x0) Gamma_sink
            S_q(x, x0) Gamma_src
        ].

The implementation scans all 16 sink gamma structures while keeping
``Gamma_src`` fixed.  In code, the antiquark line is made from the independent
``prop_bw`` source propagator,

    bw_prop(x) = gamma5 prop_bw(x, x0)^dagger gamma5,

then the contraction is

    Tr_sc[ bw_prop(x) Gamma_sink prop_fw(x, x0) Gamma_src ].

Connected meson three-point function before the sequential trick
----------------------------------------------------------------
For a local quark bilinear insertion ``O_g(x)`` and final sink momentum ``pf``,
the connected three-point function starts as

    C3_g(q, tau; pf, tsep)
      = sum_x sum_y Phi_q(x - x0) Phi_pf(y - x0)
        Tr_sc[
            S_anti(y, x0) Gamma_sink
            S_q(y, x) O_g(x) S_q(x, x0) Gamma_src
        ].

For the scalar diagnostic insertion, ``O_g(x) = 1``.  For the connected quark
EMT, ``O_g(x)`` is the symmetrized derivative bilinear described below.  The
``q`` list is ``parameters["qext"]`` and the final sink momentum is
``parameters["pf"]``.

Fixed-sink meson sequential source
----------------------------------
The sink sum over ``y`` is absorbed into a fixed-sink sequential propagator.
The application first builds a source-smeared and sink-smeared forward
propagator ``prop_fw_SS``.  ``create_meson_bw_seq_pyquda`` restricts it to the
sink time slice and builds

    eta_seq(y; pf, tsep)
      = delta_{t_y,t0+tsep}
        Phi_pf(y - x0)
        Gamma_seq prop_fw_SS(y, x0),

    Gamma_seq = gamma5 Gamma_sink^dagger gamma5.

The sequential inversion solves

    D S_seq = eta_seq.

The contraction routines then form the backward sequential meson line

    S_seq_anti(x; pf, tsep) = gamma5 S_seq(x)^dagger gamma5.

After this step, the code evaluates the sink-summed three-point function as

    C3_g(q, tau; pf, tsep)
      = sum_x Phi_q(x - x0)
        Tr_sc[
            S_seq_anti(x; pf, tsep)
            O_g(x) S_q(x, x0) Gamma_src
        ].

Production evaluates this formula through the shared primitive-bilinear
kernel, then derives ``C3_chi`` and ``C3_Tmunu`` from those primitives.

Connected quark EMT insertion
-----------------------------
The Euclidean quark EMT insertion is represented as the symmetrized bilinear

    T_{mu nu}^q(x)
      = 1/2 [ gamma_nu D_mu - left_D_mu gamma_nu ],

where ``D_mu`` is the symmetric covariant derivative acting on the forward
source-to-insertion line and ``left_D_mu`` is the corresponding derivative
acting on the backward sequential line.  The code evaluates

    C3_{mu nu}^{q,first}(q, tau)
      = +1/2 sum_x Phi_q(x - x0)
        Tr_sc[
            S_seq_anti(x)
            gamma_nu D_mu S_q(x, x0)
            Gamma_src
        ],

    C3_{mu nu}^{q,second}(q, tau)
      = -1/2 sum_x Phi_q(x - x0)
        Tr_sc[
            (left_D_mu S_seq_anti)(x)
            gamma_nu S_q(x, x0)
            Gamma_src
        ].

The final measured connected quark EMT is

    C3_{mu nu}^q = C3_{mu nu}^{q,first} + C3_{mu nu}^{q,second},

followed by explicit symmetrization under ``mu <-> nu``.

Connected scalar diagnostic insertion
-------------------------------------
The scalar diagnostic three-point function is the same sequential-source
contraction with ``O_g(x) = 1``:

    C3_chi(q, tau)
      = sum_x Phi_q(x - x0)
        Tr_sc[
            S_seq_anti(x)
            S_q(x, x0)
            Gamma_src
        ].

Stochastic quark one-point contraction
--------------------------------------
The quark one-point part estimates traces with random noise.  For a Z_n noise
field ``xi`` and solution ``eta = D^{-1} xi``, the code measures

    L_I(q,t) = sum_x Phi_q(x) xi^dagger(x) eta(x),
    N_xi(q,t) = sum_x Phi_q(x) xi^dagger(x) xi(x),

and the flowed quark EMT building block

    T_{nu mu}^q(q, t)
      = -1/2 sum_x Phi_q(x)
        xi^dagger(x) gamma_nu [D_{+mu} - D_{-mu}] eta(x).

The result is symmetrized in ``mu`` and ``nu``, averaged over noise vectors, and
volume-normalized after the average.

The same saved ``Tmunu`` data also contains the flowed-fermion kinetic
bilinear used to construct ringed fermion fields.  At zero momentum, the
diagonal trace satisfies

    sum_mu T_{mu mu}^q(0, t)
      = -1/2 sum_x
        xi^dagger(x) gamma_mu [D_{+mu} - D_{-mu}] eta(x)

      = -1/2 < bar_chi(t, x) overleftrightarrow{not D} chi(t, x) >,

up to the same lattice derivative and stochastic-estimator conventions used in
the code.  This quantity is the normalization input commonly denoted by the
flowed fermion kinetic expectation value in ringed-fermion constructions.  In
the saved quark 1pt HDF5 file it can be reconstructed from the averaged
datasets

    avg/Tmunu/T11, avg/Tmunu/T22, avg/Tmunu/T33, avg/Tmunu/T44

at the ``q = 0`` momentum index.  The identity local channel and explicitly
named ``flowed_noise_norm`` are diagnostics; they are not the standard
``bar_chi overleftrightarrow{not D} chi`` normalization by themselves.

Gluon one-point contraction
---------------------------
The gluon field strength is built from a clover operator and projected onto the
traceless anti-Hermitian su(3) algebra, with an extra factor of ``-i`` matching
the original Euclidean convention.  The measured gluon EMT building block is

    T_{mu nu}^g(q, t)
      = 2 / V3 sum_x Phi_q(x)
        sum_{rho != mu,nu} Tr_c[ F_{mu rho}(x) F_{nu rho}(x) ].

This intentionally measures the full gluonic EMT building block used for
gradient-flow renormalization.  The code does not impose a traceless projection
on the final EMT tensor.  The traceless projection in ``_F_clover_traceless``
is only the standard projection of each clover field-strength matrix onto the
su(3) gauge algebra.

How the one-point data is used
------------------------------
The connected meson three-point output gives matrix elements with a connected
quark EMT insertion on the valence line.  The stochastic quark 1pt output gives
the vacuum/disconnected quark EMT building block and the ringed-fermion
normalization information described above.  The gluon 1pt output gives the
flowed gluonic EMT building block.  These pieces are saved separately because
the final renormalized gradient-flow EMT is usually assembled in analysis, not
inside this contraction kernel.

Schematically, the analysis stage combines flowed operators as

    T_{mu nu}^{ren}(x)
      = c_g(t) O_{mu nu}^g(t, x)
        + c_q(t) O_{mu nu}^q(t, x)
        + mixing/trace terms,

with coefficients and possible vacuum subtractions determined outside this
file.  The quark 1pt and gluon 1pt measurements provide the flowed operator
building blocks and normalization inputs needed for that construction, while
the connected 3pt files provide the hadron matrix-element part.

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

Future upgrade targets
----------------------
The shared stochastic quark 1pt estimator in
``Disconnected_1pt_EMT_vibe_develop.py`` uses plain Z_n noise.  Two natural
variance-reduction upgrades should be considered before large production runs:

1. Hierarchical probing, following arXiv:1302.4018.  The EMT 1pt trace
   estimator is a trace of an inverse Dirac operator with local gamma/derivative
   insertions.  Replacing or supplementing purely random noise vectors with
   distance-ordered Hadamard/coloring vectors on the four-dimensional torus
   should cancel near-diagonal contributions of ``D^{-1}`` more deterministically
   and reduce stochastic variance at fixed inversion count.

2. Frequency splitting / propagator-decomposition variance reduction, following
   the strategies reviewed in arXiv:2605.00643.  The flowed EMT loop could be
   decomposed into low/infrared and high/ultraviolet components, or into
   frequency-filtered estimator pieces, so that expensive noise averaging is
   focused on the component with the largest residual variance.  This should be
   designed together with the gradient-flow radius because flow already smooths
   ultraviolet modes.

These are not implemented yet.  When added to the shared 1pt module, they
should preserve the existing HDF5 schema or add explicit estimator metadata so
disconnected diagram analysis can combine old and new 1pt data safely.
"""

import numpy as np
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import (
    LatticeGauge,
    LatticePropagator,
)
from pyquda_utils import core, source, phase
from pyquda_measurement_utils.io_corr import (
    save_emt_quark_3pt_hdf5,
    save_emt_meson_2pt_hdf5,
)
from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    EMT_OPERATOR_SCHEMA_VERSION,
    _flow_times,
    my_gammas,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.fermion_bilinear_basis import (
    IDENTITY_GAMMA_POSITION,
    basis_attrs,
    symmetric_vector_emt,
)
from pyquda_measurement_utils.tools import mpi_print
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_meson_bw_seq_pyquda


def _save_connected_3pt_rank0(latt_info, *args, **kwargs):
    """Keep the serial HDF5 writer completely unopened on non-root ranks."""
    if latt_info.mpi_rank == 0:
        save_emt_quark_3pt_hdf5(*args, **kwargs)

"""
================================================================================
                                  QuarkEMT
================================================================================
"""
class QuarkEMT(EMTDisconnectedQuark1pt):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.pf = parameters["pf"]
        self.pilist = parameters["p_2pt"]
        self.CG_GaussSmear = bool(parameters.get("CG_GaussSmear", False))
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]
        self.source_interpolator = parameters.get("source_interpolator", "5")
        self.sink_interpolator = parameters.get("sink_interpolator", "5")

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
    def _project_gamma_scalar_fields(cls, scalar_fields, phases_3pt, t0):
        projected = contract("qwtzyx,gwtzyx->gqt", phases_3pt, scalar_fields)
        slice_t = core.gatherLattice(array_to_numpy(projected), [2, -1, -1, -1])
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.asarray(slice_t), -t0, axis=-1)

    @classmethod
    def get_C3_primitive_bilinears(
        cls,
        U_f: LatticeGauge,
        prop_fw: LatticePropagator,
        seq_bw_prop: LatticePropagator,
        src_gamma,
        phases_3pt,
        t0: int,
    ):
        """Compute all 16 local and 16x4 one-derivative meson bilinears."""
        dst2 = cls._make_dst2(seq_bw_prop)
        gamma_ls = cls._gamma_stack_for(prop_fw.data)
        local_fields = contract(
            "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
            dst2, gamma_ls, prop_fw.data, src_gamma,
        )
        local = cls._project_gamma_scalar_fields(local_fields, phases_3pt, t0)
        del local_fields

        derivative = np.zeros(
            (16, 4, len(phases_3pt), U_f.latt_info.global_size[3]),
            dtype=np.complex128,
        )
        for mu in range(4):
            D_fw = cls._covdev_sym_prop(U_f, prop_fw, mu)
            right_fields = 0.5 * contract(
                "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
                dst2, gamma_ls, D_fw.data, src_gamma,
            )
            derivative[:, mu] += cls._project_gamma_scalar_fields(
                right_fields, phases_3pt, t0
            )
            del right_fields, D_fw

            leftD_dst2 = cls._left_covdev_dst2_from_prop(U_f, seq_bw_prop, mu)
            left_fields = -0.5 * contract(
                "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
                leftD_dst2, gamma_ls, prop_fw.data, src_gamma,
            )
            derivative[:, mu] += cls._project_gamma_scalar_fields(
                left_fields, phases_3pt, t0
            )
            del left_fields, leftD_dst2
        del dst2, gamma_ls
        return np.asarray(local), derivative

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
        C3_local_bilinear = np.zeros(
            (N_ts, 16, Nq, Nsteps + 1, Nt), dtype=np.complex128
        )
        C3_derivative_bilinear = np.zeros(
            (N_ts, 16, 4, Nq, Nsteps + 1, Nt), dtype=np.complex128
        )

        mpi_print(latt_info, f"src [{x0},{y0},{z0},{t0}]")

        prop_fw_SP, prop_bw_SP = self._make_meson_source_props(dirac, U, src_pos)
        src_gamma = self._get_interpolator_gamma_for(src_interpolator, prop_fw_SP.data)

        c2_attrs = {
            "measurement": "meson_2pt",
            "config_num": self.config_num,
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "source_position": np.asarray(src_pos, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "src_t": t0,
            "src_interpolator": src_interpolator,
            "sink_interpolator": "all_16_gamma_scan",
            "sink_gamma_scan": "all_16",
            "gaussian_smearing": self.CG_GaussSmear,
            "smearing_width": self.width,
            "source_boost": np.asarray(self.pos_boost, dtype=np.int32),
            "sink_boost": np.asarray(self.neg_boost, dtype=np.int32),
            "dataset_axes": "gamma,p,t",
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
                self.width if self.CG_GaussSmear else None,
                self.neg_boost if self.CG_GaussSmear else None,
            )

            prop_fw_flow = prop_fw_SP.copy()
            seq_bw_prop_flow = seq_bw_prop.copy()
            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(latt_info, f"contraction for step {step}")
                local_step, derivative_step = self.get_C3_primitive_bilinears(
                    U_f, prop_fw_flow, seq_bw_prop_flow, src_gamma, phases_3pt, t0
                )
                C3_local_bilinear[n_ts, :, :, step] += local_step
                C3_derivative_bilinear[n_ts, :, :, :, step] += derivative_step
                C3_chi[n_ts, step] += local_step[IDENTITY_GAMMA_POSITION]
                tensor_step = symmetric_vector_emt(
                    derivative_step, gamma_axis=0, derivative_axis=1
                )
                C3_Tmunu[n_ts, step] += np.moveaxis(
                    tensor_step, (0, 1), (1, 2)
                )

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
            "emt_operator_schema_version": EMT_OPERATOR_SCHEMA_VERSION,
            "config_num": self.config_num,
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "source_position": np.asarray(src_pos, dtype=np.int32),
            "pf": np.asarray(self.pf, dtype=np.int32),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "gaussian_smearing": self.CG_GaussSmear,
            "smearing_width": self.width,
            "source_boost": np.asarray(self.pos_boost, dtype=np.int32),
            "sink_boost": np.asarray(self.neg_boost, dtype=np.int32),
            "spin": spin,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "n_t_separations": N_ts,
            "src_t": t0,
            "src_interpolator": src_interpolator,
            "sink_interpolator": sink_interpolator,
            "contraction_convention": "B",
            "meson_sign": 1,
            "n_qext": Nq,
            "operator_normalization": "unringed_flowed_bilinear",
            "ringed_normalization_applied": False,
            "ringed_factor_source": "analysis_from_quark_1pt_kinetic",
            "quark_flow_scope": "inserted_operator_quark_legs_only",
            "hadron_interpolator_flowed": False,
            "derivative_convention": "symmetric_covdev_0p5_Dplus_minus_Dminus",
            "primitive_local_axes": "tsep,gamma,q,flow,t",
            "primitive_derivative_axes": "tsep,gamma,derivative,q,flow,t",
            "primitive_derivative_unsymmetrized": True,
            "derived_emt_axes": "tsep,flow,q,mu,nu,t",
            "C3_chi_axes": "tsep,flow,q,t",
        }
        attrs.update(basis_attrs())
        _save_connected_3pt_rank0(
            latt_info,
            tag,
            C3_chi,
            C3_Tmunu,
            C3_local_bilinear,
            C3_derivative_bilinear,
            momentum_transfer_list=self.qlist,
            attrs=attrs,
        )
        return C2, C3_chi, C3_Tmunu
