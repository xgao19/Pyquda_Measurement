"""Connected pion EMT production measurement."""
import numpy as np
from time import perf_counter
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import (
    LatticeGauge,
    LatticePropagator,
)
from pyquda_utils import core, phase
from pyquda_measurement_utils.io_corr import (
    save_emt_quark_3pt_hdf5,
    save_emt_meson_2pt_hdf5,
)
from pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop import (
    EMT_OPERATOR_SCHEMA_VERSION,
    FlowedFermionBilinearKernel,
    flow_times as _flow_times,
    my_gammas,
    parse_multigrid_blocks,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.fermion_bilinear_basis import (
    IDENTITY_GAMMA_POSITION,
    basis_attrs,
    symmetric_vector_emt,
)
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    build_pion_source_propagators,
    contract_pion_2pt,
)
from pyquda_measurement_utils.tools import mpi_print, mpi_timer_print
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
class QuarkEMT(FlowedFermionBilinearKernel):

    def __init__(self, parameters):
        super().__init__(parameters["flow_type"])
        self.qlist = parameters["qext"]
        self.flow_epsilon = parameters["flow_epsilon"]
        self.flow_steps = parameters["flow_steps"]
        self.config_num = parameters.get("config_num")
        self.gauge_preprocessing = parameters.get(
            "gauge_preprocessing", "unspecified"
        )
        multigrid = parameters.get("multigrid", [[8, 8, 4, 4]])
        self.multigrid_blocks = (
            None if multigrid is None else parse_multigrid_blocks(multigrid)
        )
        self.pf = parameters["pf"]
        self.pilist = parameters["p_2pt"]
        self.CG_GaussSmear = bool(parameters.get("CG_GaussSmear", False))
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.width = parameters["width"]
        self.source_interpolator = parameters.get("source_interpolator", "5")
        self.sink_interpolator = parameters.get("sink_interpolator", "5")

    def _make_meson_source_props(
        self, dirac, U, src_pos, *, restore_original_gauge=True
    ):
        """Build source-smeared point-sink forward/backward meson propagators."""
        latt_info = U.latt_info
        if restore_original_gauge:
            restore_t0 = perf_counter()
            dirac.loadGauge(U, thin_update_only=True)
            mpi_timer_print(latt_info, "pion_emt_source_restore", perf_counter() - restore_t0)
        invert_t0 = perf_counter()
        prop_fw_SP, prop_bw_SP = build_pion_source_propagators(
            dirac,
            latt_info,
            src_pos,
            gaussian_smearing=self.CG_GaussSmear,
            width=self.width,
            pos_boost=self.pos_boost,
            neg_boost=self.neg_boost,
        )
        mpi_timer_print(latt_info, "pion_emt_source_inversion", perf_counter() - invert_t0)
        return prop_fw_SP, prop_bw_SP

    @classmethod
    def _project_gamma_scalar_fields(cls, scalar_fields, phases_3pt, t0):
        projected = contract("qwtzyx,gwtzyx->gqt", phases_3pt, scalar_fields)
        slice_t = core.gatherLattice(array_to_numpy(projected), [2, -1, -1, -1])
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.asarray(slice_t), -t0, axis=-1)

    def get_C3_primitive_bilinears(
        self,
        U_f: LatticeGauge,
        gauge_dirac,
        prop_fw: LatticePropagator,
        seq_bw_prop: LatticePropagator,
        src_gamma,
        phases_3pt,
        t0: int,
    ):
        """Compute all 16 local and 16x4 one-derivative meson bilinears."""
        dst2 = self._make_dst2(seq_bw_prop)
        gamma_ls = self._gamma_stack_for(prop_fw.data)
        local_fields = contract(
            "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
            dst2, gamma_ls, prop_fw.data, src_gamma,
        )
        local = self._project_gamma_scalar_fields(local_fields, phases_3pt, t0)
        del local_fields

        derivative = np.zeros(
            (16, 4, len(phases_3pt), U_f.latt_info.global_size[3]),
            dtype=np.complex128,
        )
        for mu in range(4):
            D_fw = self._covdev_sym_prop(gauge_dirac, prop_fw, mu)
            right_fields = 0.5 * contract(
                "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
                dst2, gamma_ls, D_fw.data, src_gamma,
            )
            derivative[:, mu] += self._project_gamma_scalar_fields(
                right_fields, phases_3pt, t0
            )
            del right_fields, D_fw

            leftD_dst2 = self._left_covdev_dst2_from_prop(
                gauge_dirac, seq_bw_prop, mu
            )
            left_fields = -0.5 * contract(
                "wtzyxabij,gbn,wtzyxncji,ca->gwtzyx",
                leftD_dst2, gamma_ls, prop_fw.data, src_gamma,
            )
            derivative[:, mu] += self._project_gamma_scalar_fields(
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
        src_gamma_label,
        src_pos,
        tag=None,
        attrs=None,
    ):
        """Smear and contract the shared pion C2 kernel for EMT output."""
        if self.CG_GaussSmear:
            mpi_print(latt_info, f"2pt forward sink smearing starts, boost = {self.pos_boost}")
            prop_fw = boosted_smearing(prop_fw, w=self.width, boost=self.pos_boost)
            mpi_print(latt_info, f"2pt backward sink smearing starts, boost = {self.neg_boost}")
            prop_bw = boosted_smearing(prop_bw, w=self.width, boost=self.neg_boost)
            mpi_print(latt_info, "2pt sink smearing ends")

        p_2pt_xyz = [[-p[0], -p[1], -p[2]] for p in self.pilist]
        phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz, src_pos)
        C2 = contract_pion_2pt(
            latt_info,
            prop_fw,
            prop_bw,
            phases_2pt,
            src_gamma=src_gamma_label,
        )
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
        2. Optionally apply positive-boost sink smearing to the spectator.
        3. Build and invert the meson fixed-sink sequential source with
           ``create_meson_bw_seq_pyquda``.
        4. Starting from the unflowed gauge field, flow the negative-boost
           active propagator and sequential backward propagator together and
           measure C2, C3_chi(q,t), and C3_Tmunu(q,t) at each flow time.

        Notes on special objects
        ------------------------
        ``prop_fw_SP`` / ``prop_bw_SP``
            Source-smeared, point-sink propagators for the positive-boost
            spectator line and negative-boost active line, respectively.

        ``prop_fw_SS``
            Source- and sink-smeared positive-boost spectator propagator used
            to build the fixed-sink sequential source.  The outer smearing of
            that source uses ``neg_boost`` for the active line at the sink.

        ``seq_bw_prop``
            fixed-sink backward sequential propagator.  The underlying source
            is built by applying the sink momentum phase and the standard meson
            gamma structure gamma5 * Gamma_sink^dagger * gamma5.
        """
        N_ts = len(t_separations)

        U = gauge
        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps
        latt_info = U.latt_info
        Nt = latt_info.global_size[3]
        mass, csw, tol, maxiter = invPara

        x0, y0, z0, t0 = src_pos
        total_t0 = perf_counter()
        mpi_print(latt_info, f"t_boundary = {latt_info.t_boundary}")
        dirac = core.getDirac(
            latt_info,
            mass,
            tol,
            maxiter,
            1.0,
            csw,
            csw,
            self.multigrid_blocks,
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

        prop_fw_SP, prop_bw_SP = self._make_meson_source_props(
            dirac, U, src_pos, restore_original_gauge=False
        )
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
            "pos_boost": np.asarray(self.pos_boost, dtype=np.int32),
            "neg_boost": np.asarray(self.neg_boost, dtype=np.int32),
            "dataset_axes": "gamma,p,t",
        }
        C2 += self.contract_meson_2pt(
            latt_info,
            prop_fw_SP.copy(),
            prop_bw_SP.copy(),
            src_interpolator,
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

            if n_ts > 0:
                restore_t0 = perf_counter()
                dirac.loadGauge(U, thin_update_only=True)
                mpi_timer_print(
                    latt_info, "pion_emt_sequential_restore",
                    perf_counter() - restore_t0, t_sep=t_sep,
                )
            inversion_t0 = perf_counter()
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
            mpi_timer_print(
                latt_info, "pion_emt_sequential_inversion",
                perf_counter() - inversion_t0, t_sep=t_sep,
            )

            # The sequential line was built from the positive-boost spectator
            # propagator.  The EMT insertion therefore contracts with the
            # independent negative-boost active propagator.
            active_prop_flow = prop_bw_SP.copy()
            seq_bw_prop_flow = seq_bw_prop.copy()
            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(latt_info, f"contraction for step {step}")
                primitive_t0 = perf_counter()
                with U_f.use() as gauge_dirac:
                    local_step, derivative_step = self.get_C3_primitive_bilinears(
                        U_f, gauge_dirac, active_prop_flow, seq_bw_prop_flow,
                        src_gamma, phases_3pt, t0,
                    )
                mpi_timer_print(
                    latt_info, "pion_emt_primitive",
                    perf_counter() - primitive_t0, t_sep=t_sep, step=step,
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

                flow_t0 = perf_counter()
                active_prop_flow, seq_bw_prop_flow = self._advance_flowed_props(
                    U_f,
                    active_prop_flow,
                    seq_bw_prop_flow,
                    step,
                    stepsize,
                    Nsteps,
                )
                if step < Nsteps:
                    mpi_timer_print(
                        latt_info, "pion_emt_flow", perf_counter() - flow_t0,
                        t_sep=t_sep, step=f"{step}_to_{step + 1}",
                    )

            del U_f, active_prop_flow, seq_bw_prop_flow, seq_bw_prop

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
            "pos_boost": np.asarray(self.pos_boost, dtype=np.int32),
            "neg_boost": np.asarray(self.neg_boost, dtype=np.int32),
            "operator_insertion_line": "neg_boost",
            "boost_line_convention": "pos_spectator_neg_active",
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
            "one_derivative_operator": (
                "0.5*bar_chi*Gamma_A*(rightD_mu-leftD_mu)*chi"
            ),
            "derivative_closed_fermion_loop_sign_included": False,
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
        mpi_timer_print(
            latt_info, "pion_emt_total", perf_counter() - total_t0,
            source_position=src_pos,
        )
        return C2, C3_chi, C3_Tmunu
