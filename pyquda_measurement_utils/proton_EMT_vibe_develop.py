"""Proton EMT measurement formulas and conventions.

This module is the proton analogue of ``pion_EMT_vibe_develop.py``.  It keeps
the flowed quark/gluon EMT definitions from the meson code, but replaces the
connected fixed-sink contraction with the baryon sequential-source convention
already used by ``proton_qTMD_pyquda.py``.

Common notation
---------------
The source is ``x0 = (t0, x0)``.  The EMT insertion is ``x = (tau, x)``.  The
fixed proton sink is ``y = (tsep, y)``.  The proton interpolating field is
schematically

    chi_alpha(y) =
        epsilon_abc [u_a^T(y) C Gamma_interp d_b(y)] u_{c,alpha}(y).

The sink spin projection is supplied by ``PolProjections`` in
``bw_seq_pyquda.py``.  Momentum phases are written as ``Phi_k(r - x0)``; the
actual sign convention is the one used by ``MomentumPhase``.

Connected proton three-point function before the sequential trick
-----------------------------------------------------------------
For a connected insertion on flavor ``f`` in {U, D}, the target correlator is

    C3_f(q, tau; pf, tsep, P)
      = sum_x sum_y Phi_q(x - x0) Phi_pf(y - x0)
        P_{alpha alpha'}
        < chi_alpha(y) O_f(x) bar_chi_{alpha'}(x0) >_connected.

After Wick contraction the sink-side baryon contraction, spin projection,
final momentum, and sink time are absorbed into a flavor-dependent fixed-sink
sequential source built by ``create_bw_seq_pyquda``.

Proton sequential-source contraction
------------------------------------
``create_bw_seq_pyquda`` returns a sequential backward object with shape

    [polarization, w, t, z, y, x_cb, spin_a, spin_b, color_a, color_b].

For qTMD/PDF insertions the mature proton code contracts it as

    Seq_f(p, x) Gamma_g S_q(x, x0).

The EMT contractions here use the same object and replace ``Gamma_g S_q`` by
the local derivative bilinear.  The scalar diagnostic is

    C3_chi_f(q, tau)
      = sum_x Phi_q(x - x0)
        Seq_f(x) S_q(x, x0).

The connected quark EMT insertion is evaluated as

    C3_{f,mu nu}^{first}(q, tau)
      = +1/2 sum_x Phi_q(x - x0)
        Seq_f(x) gamma_nu D_mu S_q(x, x0),

    C3_{f,mu nu}^{second}(q, tau)
      = -1/2 sum_x Phi_q(x - x0)
        (left_D_mu Seq_f)(x) gamma_nu S_q(x, x0).

The measured tensor is the sum of these two terms, symmetrized under
``mu <-> nu``.  This is the direct proton counterpart of the meson convention B
used in ``pion_EMT_vibe_develop.py``.

One-point and gradient-flow data
--------------------------------
The stochastic quark 1pt, ringed-fermion kinetic normalization, and gluon 1pt
building blocks are inherited from the shared
``Disconnected_1pt_EMT_vibe_develop.py`` implementation.  The quark 1pt output
contains ``avg/Tmunu/T11`` through ``T44`` for reconstructing the zero-momentum
``bar_chi overleftrightarrow{not D} chi`` normalization, and the gluon 1pt
output provides the flowed gluonic EMT building block.  The final renormalized
gradient-flow EMT is assembled in analysis from connected 3pt, quark 1pt, and
gluon 1pt data.

Limitations
-----------
This module computes connected proton EMT three-point functions for U and D
insertions.  Disconnected diagrams, renormalization coefficients, vacuum
subtractions, and flavor mixing are intentionally left to separate workflows or
analysis code.

Future upgrade targets
----------------------
The proton connected 3pt contractions use sequential propagators, while the
disconnected quark EMT contribution is inherited from the shared stochastic
quark 1pt loop estimator.  The next variance-reduction upgrades should
therefore target ``Disconnected_1pt_EMT_vibe_develop.py``:

1. Hierarchical probing, following arXiv:1302.4018.  Structured probing vectors
   on the toroidal lattice can reduce the stochastic variance of trace
   estimators by canceling near-neighbor contributions of the inverse Dirac
   operator more systematically than independent noise alone.

2. Frequency splitting / propagator-decomposition variance reduction, following
   the strategies reviewed in arXiv:2605.00643.  Splitting the quark propagator
   or loop estimator into frequency components can let the code treat low-mode,
   high-mode, and flowed ultraviolet-suppressed pieces with different estimator
   budgets.

These upgrades are roadmap items only.  They should be implemented in the
shared quark 1pt path and recorded in HDF5 metadata before being mixed with
connected proton EMT data in disconnected-diagram analyses.
"""

import gc
import numpy as np
from time import perf_counter
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda_utils import core, source, phase

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    EMT_OPERATOR_SCHEMA_VERSION,
    _flow_times,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_raw_pyquda
from pyquda_measurement_utils.io_corr import (
    save_emt_quark_3pt_hdf5,
    save_proton_c2pt_hdf5,
)
from pyquda_measurement_utils.proton_utils_vibe_develop import (
    contract_proton_c2,
    proton_interpolator_matrix,
)
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    IDENTITY_GAMMA_POSITION,
    basis_attrs,
    symmetric_vector_emt,
)
from pyquda_measurement_utils.tools import mpi_print, mpi_timer_print


class ProtonQuarkEMT(EMTDisconnectedQuark1pt):
    """Connected proton EMT plus inherited stochastic quark 1pt utilities."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.pf = parameters["pf"]
        self.pilist = parameters["p_2pt"]
        self.CG_GaussSmear = bool(parameters.get("CG_GaussSmear", False))
        self.width = parameters["width"]
        self.pol_list = parameters["pol"]
        self.t_insert = parameters["t_insert"]
        self.t_separations = [int(t_sep) for t_sep in parameters.get("t_separations", [self.t_insert])]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]

    @staticmethod
    def _wait_sycl_queues(*objects):
        """Wait for visible SYCL queues before dropping large dpnp/dpctl arrays."""
        seen = set()

        def wait_one(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                for value in obj.values():
                    wait_one(value)
                return
            if isinstance(obj, (list, tuple)):
                for value in obj:
                    wait_one(value)
                return
            data = getattr(obj, "data", obj)
            queue = getattr(data, "sycl_queue", None)
            if queue is None:
                return
            queue_id = id(queue)
            if queue_id in seen:
                return
            seen.add(queue_id)
            queue.wait()

        for obj in objects:
            wait_one(obj)

    @classmethod
    def _cleanup_source_objects(cls, *objects):
        cls._wait_sycl_queues(*objects)
        gc.collect()

    def _make_source_prop(
        self, dirac, U, src_pos, *, restore_original_gauge=True
    ):
        latt_info = U.latt_info
        src = source.propagator(latt_info, "point", src_pos)
        if self.CG_GaussSmear:
            mpi_print(latt_info, f"proton source smearing starts, boost = {self.boost_in}")
            src = boosted_smearing(src, w=self.width, boost=self.boost_in)
            mpi_print(latt_info, "proton source smearing ends")

        if restore_original_gauge:
            restore_t0 = perf_counter()
            dirac.loadGauge(U, thin_update_only=True)
            mpi_timer_print(
                latt_info, "proton_emt_source_restore", perf_counter() - restore_t0
            )
        inversion_t0 = perf_counter()
        prop = core.invertPropagator(dirac, src, 1, 0)
        mpi_timer_print(
            latt_info, "proton_emt_source_inversion", perf_counter() - inversion_t0
        )
        del src
        return prop

    def contract_proton_2pt(
        self, latt_info, prop, src_pos, tag=None, interpolator="5", attrs=None
    ):
        p_2pt_xyz = [[-p[0], -p[1], -p[2]] for p in self.pilist]
        phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz, src_pos)
        gamma_matrices = self._gamma_stack_for(prop.data)
        proton_interpolator = self._cached_backend_matrix(
            f"proton_interpolator:{interpolator}",
            proton_interpolator_matrix(interpolator),
            prop.data,
        )
        C2 = contract_proton_c2(
            latt_info,
            prop,
            phases_2pt,
            interpolator=interpolator,
            sink_smearing=self.CG_GaussSmear,
            smearing_width=self.width,
            smearing_boost=self.boost_out,
            gamma_matrices=gamma_matrices,
            interpolator_matrix=proton_interpolator,
        )
        if tag is not None and latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(
                C2, tag, list(GAMMA_LABELS), self.pilist, attrs=attrs
            )
        return getMPIComm().bcast(C2, root=0)

    @classmethod
    def _raw_seq_gamma_scalar_fields(cls, raw_seq_prop, prop_fw, G5_gamma_stack):
        spin_trace = contract(
            "wtzyxajfc,wtzyxbjfc->wtzyxab",
            raw_seq_prop.data.conj(), prop_fw.data,
        )
        return contract("wtzyxab,gab->gwtzyx", spin_trace, G5_gamma_stack)

    @classmethod
    def _project_gamma_scalars_to_qt(cls, scalar_fields, phases_3pt, t0):
        slice_t = core.gatherLattice(
            array_to_numpy(contract("qwtzyx,gwtzyx->gqt", phases_3pt, scalar_fields)),
            [2, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.asarray(slice_t), -t0, axis=-1)

    def get_C3_primitive_bilinears_proton(
        self, U_f, gauge_dirac, prop_fw, raw_seq_prop, phases_3pt, t0
    ):
        """Compute all local and one-derivative proton insertion bilinears."""
        gamma_ls = self._gamma_stack_for(prop_fw.data)
        G5_local = self._gamma5_for(prop_fw.data)
        G5_gamma_stack = contract("ai,gib->gab", G5_local, gamma_ls)

        local_fields = self._raw_seq_gamma_scalar_fields(
            raw_seq_prop, prop_fw, G5_gamma_stack
        )
        local = self._project_gamma_scalars_to_qt(local_fields, phases_3pt, t0)
        del local_fields
        derivative = np.zeros(
            (16, 4, len(phases_3pt), U_f.latt_info.global_size[3]),
            dtype=np.complex128,
        )
        for mu in range(4):
            D_fw = self._covdev_sym_prop(gauge_dirac, prop_fw, mu)
            right_fields = 0.5 * self._raw_seq_gamma_scalar_fields(
                raw_seq_prop, D_fw, G5_gamma_stack
            )
            derivative[:, mu] += self._project_gamma_scalars_to_qt(
                right_fields, phases_3pt, t0
            )
            del D_fw, right_fields

            D_raw_seq = self._covdev_sym_prop(gauge_dirac, raw_seq_prop, mu)
            left_fields = -0.5 * self._raw_seq_gamma_scalar_fields(
                D_raw_seq, prop_fw, G5_gamma_stack
            )
            derivative[:, mu] += self._project_gamma_scalars_to_qt(
                left_fields, phases_3pt, t0
            )
            del D_raw_seq, left_fields
        del gamma_ls, G5_local, G5_gamma_stack
        return np.asarray(local), derivative

    def _connected_3pt_one_source(
        self,
        dirac,
        gauge,
        source_job,
        interpolator="5",
    ):
        """Compute and write connected proton U/D quark EMT 3pt functions for one source."""
        U = gauge
        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps
        latt_info = U.latt_info
        Nt = latt_info.global_size[3]
        src_pos = source_job["src_pos"]
        tags = source_job["tags"]
        c2_tag = source_job.get("c2_tag")
        t0 = src_pos[3]
        t_separations = self.t_separations

        prop_fw = None
        C3_chi = None
        C3_Tmunu = None
        C3_local_bilinear = None
        C3_derivative_bilinear = None
        phases_3pt = None
        raw_seq_bw = None
        U_f = None
        prop_fw_flow = None
        raw_seq_prop_flow = None
        try:
            prop_fw = self._make_source_prop(
                dirac,
                U,
                src_pos,
                restore_original_gauge=source_job.get("restore_source_gauge", True),
            )
            mass, csw, tol, maxiter = self._connected_invPara
            c2_attrs = {
                "measurement": "proton_2pt",
                "config_num": source_job.get("config"),
                "mass": mass,
                "csw": csw,
                "tol": tol,
                "maxiter": maxiter,
                "gauge_preprocessing": self.gauge_preprocessing,
                "t_boundary": latt_info.t_boundary,
                "source_position": np.asarray(src_pos, dtype=np.int32),
                "p_2pt": np.asarray(self.pilist, dtype=np.int32),
                "source_interpolator": interpolator,
                "sink_interpolator": "all_16_gamma_scan",
                "gaussian_smearing": self.CG_GaussSmear,
                "source_smearing": self.CG_GaussSmear,
                "sink_smearing": self.CG_GaussSmear,
                "sequential_smearing": self.CG_GaussSmear,
                "smearing_width": self.width,
                "source_boost": np.asarray(self.boost_in, dtype=np.int32),
                "sink_boost": np.asarray(self.boost_out, dtype=np.int32),
                "dataset_axes": "gamma,p,t",
            }
            self.contract_proton_2pt(
                latt_info,
                prop_fw.copy(),
                src_pos,
                tag=c2_tag,
                interpolator=interpolator,
                attrs=c2_attrs,
            )

            qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
            phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, src_pos)

            Nflavor = 2
            Npol = len(self.pol_list)
            Nq = len(self.qlist)

            for t_sep in t_separations:
                Ninsert = int(t_sep) + 2
                C3_chi = np.zeros((Nflavor, Npol, Nsteps + 1, Nq, Ninsert), dtype=np.complex128)
                C3_Tmunu = np.zeros((Nflavor, Npol, Nsteps + 1, Nq, 4, 4, Ninsert), dtype=np.complex128)
                C3_local_bilinear = np.zeros(
                    (Nflavor, Npol, 16, Nq, Nsteps + 1, Ninsert),
                    dtype=np.complex128,
                )
                C3_derivative_bilinear = np.zeros(
                    (Nflavor, Npol, 16, 4, Nq, Nsteps + 1, Ninsert),
                    dtype=np.complex128,
                )
                for flavor_idx, flavor in enumerate([1, 2]):
                    flavor_name = "U" if flavor == 1 else "D"
                    mpi_print(latt_info, f"create proton sequential source flavor={flavor_name} t_sep={t_sep}")
                    first_sequential = t_sep == t_separations[0] and flavor_idx == 0
                    if not first_sequential:
                        restore_t0 = perf_counter()
                        dirac.loadGauge(U, thin_update_only=True)
                        mpi_timer_print(
                            latt_info,
                            "proton_emt_sequential_restore",
                            perf_counter() - restore_t0,
                            flavor=flavor_name,
                            t_sep=t_sep,
                        )
                    inversion_t0 = perf_counter()
                    raw_seq_bw = create_bw_seq_raw_pyquda(
                        dirac,
                        prop_fw.copy(),
                        src_pos,
                        self.width if self.CG_GaussSmear else None,
                        self.boost_out if self.CG_GaussSmear else None,
                        self.pf,
                        t_sep,
                        self.pol_list,
                        flavor,
                        interpolator,
                    )
                    mpi_timer_print(
                        latt_info,
                        "proton_emt_sequential_inversion",
                        perf_counter() - inversion_t0,
                        flavor=flavor_name,
                        t_sep=t_sep,
                    )

                    for pol_idx, pol in enumerate(self.pol_list):
                        prop_fw_flow = prop_fw.copy()
                        raw_seq_prop_flow = raw_seq_bw[pol_idx].copy()
                        U_f = U.copy()
                        U_f.setAntiPeriodicT()

                        for step in range(Nsteps + 1):
                            step_t0 = perf_counter()
                            mpi_print(
                                latt_info,
                                f"proton EMT contraction flavor={flavor_name} pol={pol} t_sep={t_sep} step={step}",
                            )
                            primitive_t0 = perf_counter()
                            with U_f.use() as gauge_dirac:
                                local_step, derivative_step = self.get_C3_primitive_bilinears_proton(
                                    U_f,
                                    gauge_dirac,
                                    prop_fw_flow,
                                    raw_seq_prop_flow,
                                    phases_3pt,
                                    t0,
                                )
                            C3_local_bilinear[
                                flavor_idx, pol_idx, :, :, step
                            ] += local_step[..., :Ninsert]
                            C3_derivative_bilinear[
                                flavor_idx, pol_idx, :, :, :, step
                            ] += derivative_step[..., :Ninsert]
                            C3_chi[flavor_idx, pol_idx, step] += local_step[
                                IDENTITY_GAMMA_POSITION, ..., :Ninsert
                            ]
                            tensor_step = symmetric_vector_emt(
                                derivative_step, gamma_axis=0, derivative_axis=1
                            )
                            C3_Tmunu[flavor_idx, pol_idx, step] += np.moveaxis(
                                tensor_step, (0, 1), (1, 2)
                            )[..., :Ninsert]
                            primitive_seconds = perf_counter() - primitive_t0

                            flow_seconds = 0.0
                            if step < Nsteps:
                                flow_t0 = perf_counter()
                                prop_fw_flow, raw_seq_prop_flow = self._advance_flowed_props(
                                    U_f,
                                    prop_fw_flow,
                                    raw_seq_prop_flow,
                                    step,
                                    stepsize,
                                    Nsteps,
                                )
                                flow_seconds = perf_counter() - flow_t0
                            mpi_timer_print(
                                latt_info,
                                "proton_emt_step",
                                perf_counter() - step_t0,
                                flavor=flavor_name,
                                pol=pol,
                                t_sep=t_sep,
                                step=step,
                                primitive_s=primitive_seconds,
                                flow_s=flow_seconds,
                            )

                        self._wait_sycl_queues(U_f, prop_fw_flow, raw_seq_prop_flow)
                        del U_f, prop_fw_flow, raw_seq_prop_flow
                        U_f = None
                        prop_fw_flow = None
                        raw_seq_prop_flow = None
                    self._wait_sycl_queues(raw_seq_bw)
                    del raw_seq_bw
                    raw_seq_bw = None

                attrs = {
                    "measurement": "proton_quark_3pt",
                    "emt_operator_schema_version": EMT_OPERATOR_SCHEMA_VERSION,
                    "config_num": source_job.get("config"),
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
                    "source_smearing": self.CG_GaussSmear,
                    "sink_smearing": self.CG_GaussSmear,
                    "sequential_smearing": self.CG_GaussSmear,
                    "smearing_width": self.width,
                    "source_boost": np.asarray(self.boost_in, dtype=np.int32),
                    "sink_boost": np.asarray(self.boost_out, dtype=np.int32),
                    "flow_type": self.flow_type,
                    "flow_epsilon": self.flow_epsilon,
                    "flow_steps": self.flow_steps,
                    "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
                    "t_sep": int(t_sep),
                    "src_t": t0,
                    "time_insertion_range": "0..t_sep+1",
                    "time_insertion_count": Ninsert,
                    "interpolator": interpolator,
                    "source_interpolator": interpolator,
                    "sink_interpolator": interpolator,
                    "flavor_axis": "0=U,1=D",
                    "polarization_axis": ",".join(self.pol_list),
                    "n_qext": Nq,
                    "connected_only": True,
                    "operator_normalization": "unringed_flowed_bilinear",
                    "ringed_normalization_applied": False,
                    "ringed_factor_source": "analysis_from_quark_1pt_kinetic",
                    "quark_flow_scope": "inserted_operator_quark_legs_only",
                    "nucleon_interpolator_flowed": False,
                    "derivative_convention": "symmetric_covdev_0p5_Dplus_minus_Dminus",
                    "left_derivative_convention": "raw_seq_gamma5_hermiticity",
                    "C3_chi_axes": "flavor,polarization,flow,q,t",
                    "primitive_local_axes": "flavor,polarization,gamma,q,flow,t",
                    "primitive_derivative_axes": "flavor,polarization,gamma,derivative,q,flow,t",
                    "primitive_derivative_unsymmetrized": True,
                    "derived_emt_axes": "flavor,polarization,flow,q,mu,nu,t",
                }
                attrs.update(basis_attrs())
                if latt_info.mpi_rank == 0:
                    save_emt_quark_3pt_hdf5(
                        tags[t_sep],
                        C3_chi,
                        C3_Tmunu,
                        C3_local_bilinear,
                        C3_derivative_bilinear,
                        momentum_transfer_list=self.qlist,
                        attrs=attrs,
                    )
                C3_chi = None
                C3_Tmunu = None
                C3_local_bilinear = None
                C3_derivative_bilinear = None
                self._cleanup_source_objects()
            return {"src_pos": list(src_pos), "tags": tags, "c2_tag": c2_tag}
        finally:
            self._cleanup_source_objects(
                U_f,
                prop_fw_flow,
                raw_seq_prop_flow,
                raw_seq_bw,
                phases_3pt,
                prop_fw,
            )
            del C3_chi, C3_Tmunu, C3_local_bilinear, C3_derivative_bilinear

    def connected_3pt(self, gauge, invPara, source_jobs, interpolator="5", on_source_done=None):
        """Compute connected proton U/D quark EMT 3pt functions for source jobs."""
        U = gauge
        latt_info = U.latt_info
        mass, csw, tol, maxiter = invPara
        self._connected_invPara = tuple(invPara)
        mpi_print(latt_info, f"Proton EMT multigrid block: {self.multigrid_blocks}")
        dirac = core.getDirac(
            latt_info, mass, tol, maxiter, 1.0, csw, csw,
            self.multigrid_blocks,
        )
        dirac.loadGauge(U)
        mpi_print(latt_info, "Proton EMT inverter ready.")

        results = []
        for source_job in source_jobs:
            source_job = dict(source_job)
            source_job["restore_source_gauge"] = bool(results)
            src_idx = source_job.get("src_idx", len(results))
            src_pos = source_job["src_pos"]
            source_t0 = perf_counter()
            mpi_print(latt_info, f"--source_start index={src_idx} src_pos={src_pos}")
            result = self._connected_3pt_one_source(dirac, U, source_job, interpolator=interpolator)
            results.append(result)
            if on_source_done is not None:
                on_source_done(source_job, result)
            timer_fields = {"src_idx": src_idx, "src_pos": src_pos}
            if "config" in source_job:
                timer_fields["config"] = source_job["config"]
            mpi_timer_print(
                latt_info,
                "proton_emt_source",
                perf_counter() - source_t0,
                **timer_fields,
            )
            self._cleanup_source_objects()
        return results

__all__ = ["ProtonQuarkEMT"]
