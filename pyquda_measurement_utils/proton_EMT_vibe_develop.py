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
import os
from time import perf_counter
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import LatticePropagator
from pyquda_utils import core, gamma, source, phase

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedGluon1pt,
    EMTDisconnectedQuark1pt,
    _flow_times,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_raw_pyquda
from pyquda_measurement_utils.io_corr import save_emt_quark_3pt_hdf5
from pyquda_measurement_utils.proton_qTMD_pyquda import my_gammas, proton_TMD
from pyquda_measurement_utils.tools import mpi_print, mpi_timer_print


def _parse_mg_block(default):
    text = os.environ.get("EMT_PROTON_MG_BLOCK")
    if not text:
        return default
    text = text.strip()
    if text.lower() in {"none", "off", "false", "0"}:
        return None
    if any(sep in text for sep in [";", "/", "|"]):
        block_texts = [part.strip() for part in text.replace("/", ";").replace("|", ";").split(";")]
    elif "," in text and "." in text:
        block_texts = [part.strip() for part in text.split(",")]
    else:
        block_texts = [text]

    blocks = []
    for block_text in block_texts:
        if not block_text:
            continue
        block = [int(v) for v in block_text.replace(",", ".").split(".") if v]
        if len(block) != 4:
            raise ValueError(
                "EMT_PROTON_MG_BLOCK blocks must contain four integers, "
                "e.g. 4.4.4.4 or 4.4.4.4;4.4.2.2"
            )
        blocks.append(block)
    if not blocks:
        raise ValueError("EMT_PROTON_MG_BLOCK did not contain any multigrid blocks")
    return blocks


class ProtonQuarkEMT(EMTDisconnectedQuark1pt):
    """Connected proton EMT plus inherited stochastic quark 1pt utilities."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.pol_list = parameters["pol"]
        self.t_insert = parameters["t_insert"]
        self.t_separations = parameters.get("t_separations", [self.t_insert])
        self.save_propagators = parameters.get("save_propagators", False)
        self.boost_in = parameters.get("boost_in", self.pos_boost)
        self.boost_out = parameters.get("boost_out", self.pos_boost)

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

    def _make_source_prop(self, dirac, U, src_pos):
        latt_info = U.latt_info
        src = source.propagator(latt_info, "point", src_pos)
        if self.CG_GaussSmear:
            mpi_print(latt_info, f"proton source smearing starts, boost = {self.boost_in}")
            src = boosted_smearing(src, w=self.width, boost=self.boost_in)
            mpi_print(latt_info, "proton source smearing ends")

        dirac.loadGauge(U)
        prop = core.invertPropagator(dirac, src, 1, 0)
        del src
        return prop

    def contract_proton_2pt(self, latt_info, prop, src_pos, tag=None, interpolator="5"):
        p_2pt_xyz = [[-p[0], -p[1], -p[2]] for p in self.pilist]
        phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz, src_pos)
        helper = proton_TMD(
            {
                "eta": [0],
                "b_z": 0,
                "b_T": 0,
                "qext": self.qlist,
                "qext_PDF": self.qlist,
                "pf": self.pf,
                "p_2pt": self.pilist,
                "boost_in": self.boost_in,
                "boost_out": self.boost_out,
                "width": self.width,
                "pol": self.pol_list,
                "t_insert": self.t_insert,
                "save_propagators": self.save_propagators,
            }
        )
        C2 = helper.contract_2pt_TMD(latt_info, prop, phases_2pt, tag, interpolator=interpolator)
        return getMPIComm().bcast(C2, root=0)

    @staticmethod
    def _seq_to_prop(latt_info, seq_data):
        seq_prop = LatticePropagator(latt_info)
        seq_prop.data = seq_data
        return seq_prop

    @classmethod
    def _final_seq_from_raw_prop(cls, raw_seq_prop):
        G5_local = cls._gamma5_for(raw_seq_prop.data)
        return contract("wtzyxijfc,ik->wtzyxjkcf", raw_seq_prop.data.conj(), G5_local)

    @classmethod
    def _left_covdev_seq_from_raw_prop(cls, U_f, raw_seq_prop, mu):
        D_raw_seq = cls._covdev_sym_prop(U_f, raw_seq_prop, mu)
        return cls._final_seq_from_raw_prop(D_raw_seq)

    @classmethod
    def _project_scalar_to_qt(cls, scalar_field, phases_3pt, t0):
        slice_t = core.gatherLattice(
            array_to_numpy(contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field)),
            [1, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.array(slice_t), -t0, axis=-1)

    @classmethod
    def _raw_seq_scalar_field(cls, raw_seq_prop, prop_fw):
        G5_local = cls._gamma5_for(prop_fw.data)
        spin_trace = contract("wtzyxajfc,wtzyxijfc->wtzyxai", raw_seq_prop.data.conj(), prop_fw.data)
        return contract("wtzyxai,ai->wtzyx", spin_trace, G5_local)

    @classmethod
    def _raw_seq_gamma_scalar_field(cls, raw_seq_prop, prop_fw, G5_gamma):
        spin_trace = contract("wtzyxajfc,wtzyxbjfc->wtzyxab", raw_seq_prop.data.conj(), prop_fw.data)
        return contract("wtzyxab,ab->wtzyx", spin_trace, G5_gamma)

    @classmethod
    def get_C3_chi_proton(cls, latt_info, prop_fw, raw_seq_prop, phases_3pt, t0):
        scalar_field = cls._raw_seq_scalar_field(raw_seq_prop, prop_fw)
        return cls._project_scalar_to_qt(scalar_field, phases_3pt, t0)

    @classmethod
    def get_C3_Tmunu_symmetrized_proton(cls, U_f, prop_fw, raw_seq_prop, phases_3pt, t0):
        Nq = len(phases_3pt)
        Nt = U_f.latt_info.global_size[3]
        C3_Tmunu = np.zeros((Nq, 4, 4, Nt), dtype=np.complex128)
        D_gammas_local = cls._dirac_gammas_for(prop_fw.data)
        G5_local = cls._gamma5_for(prop_fw.data)

        for mu in range(4):
            D_fw = cls._covdev_sym_prop(U_f, prop_fw, mu)
            for nu in range(4):
                G5_gamma = contract("ai,ib->ab", G5_local, D_gammas_local[nu])
                scalar_field = 0.5 * cls._raw_seq_gamma_scalar_field(raw_seq_prop, D_fw, G5_gamma)
                C3_Tmunu[:, mu, nu] += cls._project_scalar_to_qt(scalar_field, phases_3pt, t0)
                del scalar_field, G5_gamma
            del D_fw

        for mu in range(4):
            D_raw_seq = cls._covdev_sym_prop(U_f, raw_seq_prop, mu)
            for nu in range(4):
                G5_gamma = contract("ai,ib->ab", G5_local, D_gammas_local[nu])
                scalar_field = -0.5 * cls._raw_seq_gamma_scalar_field(D_raw_seq, prop_fw, G5_gamma)
                C3_Tmunu[:, mu, nu] += cls._project_scalar_to_qt(scalar_field, phases_3pt, t0)
                del scalar_field, G5_gamma
            del D_raw_seq

        for mu in range(4):
            for nu in range(mu + 1, 4):
                C3_Tmunu[:, mu, nu] = 0.5 * (C3_Tmunu[:, mu, nu] + C3_Tmunu[:, nu, mu])
                C3_Tmunu[:, nu, mu] = C3_Tmunu[:, mu, nu]

        return C3_Tmunu

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
        tag = source_job["tag"]
        c2_tag = source_job.get("c2_tag")
        t0 = src_pos[3]
        t_separations = self.t_separations

        prop_fw = None
        C2 = None
        C3_chi = None
        C3_Tmunu = None
        phases_3pt = None
        raw_seq_bw = None
        U_f = None
        prop_fw_flow = None
        raw_seq_prop_flow = None
        try:
            prop_fw = self._make_source_prop(dirac, U, src_pos)
            C2 = self.contract_proton_2pt(latt_info, prop_fw.copy(), src_pos, tag=c2_tag, interpolator=interpolator)
            zero_mom_idx = self.pilist.index([0, 0, 0, 0]) if [0, 0, 0, 0] in self.pilist else 0
            sink_gamma_idx = my_gammas.index(interpolator)
            C2_selected = C2[sink_gamma_idx, zero_mom_idx]

            qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
            phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, src_pos)

            Nflavor = 2
            Npol = len(self.pol_list)
            Nts = len(t_separations)
            Nq = len(self.qlist)
            C3_chi = np.zeros((Nflavor, Npol, Nts, Nsteps + 1, Nq, Nt), dtype=np.complex128)
            C3_Tmunu = np.zeros((Nflavor, Npol, Nts, Nsteps + 1, Nq, 4, 4, Nt), dtype=np.complex128)

            for n_ts, t_sep in enumerate(t_separations):
                for flavor_idx, flavor in enumerate([1, 2]):
                    flavor_name = "U" if flavor == 1 else "D"
                    mpi_print(latt_info, f"create proton sequential source flavor={flavor_name} t_sep={t_sep}")
                    dirac.loadGauge(U)
                    raw_seq_bw = create_bw_seq_raw_pyquda(
                        dirac,
                        prop_fw.copy(),
                        src_pos,
                        self.width,
                        self.boost_out,
                        self.pf,
                        t_sep,
                        self.pol_list,
                        flavor,
                        interpolator,
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
                            chi_t0 = perf_counter()
                            C3_chi[flavor_idx, pol_idx, n_ts, step] += self.get_C3_chi_proton(
                                latt_info,
                                prop_fw_flow,
                                raw_seq_prop_flow,
                                phases_3pt,
                                t0,
                            )
                            chi_seconds = perf_counter() - chi_t0
                            tmunu_t0 = perf_counter()
                            C3_Tmunu[flavor_idx, pol_idx, n_ts, step] += self.get_C3_Tmunu_symmetrized_proton(
                                U_f,
                                prop_fw_flow,
                                raw_seq_prop_flow,
                                phases_3pt,
                                t0,
                            )
                            tmunu_seconds = perf_counter() - tmunu_t0

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
                                chi_s=chi_seconds,
                                tmunu_s=tmunu_seconds,
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
                "flow_type": self.flow_type,
                "flow_epsilon": self.flow_epsilon,
                "flow_steps": self.flow_steps,
                "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
                "src_t": t0,
                "interpolator": interpolator,
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
                "c2_selected_momentum_index": zero_mom_idx,
                "c2_selected_momentum": self.pilist[zero_mom_idx],
            }
            if latt_info.mpi_rank == 0:
                save_emt_quark_3pt_hdf5(
                    tag,
                    C2_selected,
                    C3_chi,
                    C3_Tmunu,
                    momentum_transfer_list=self.qlist,
                    attrs=attrs,
                )
            return {"src_pos": list(src_pos), "tag": tag, "c2_tag": c2_tag}
        finally:
            self._cleanup_source_objects(
                U_f,
                prop_fw_flow,
                raw_seq_prop_flow,
                raw_seq_bw,
                phases_3pt,
                prop_fw,
            )
            del C2, C3_chi, C3_Tmunu

    def connected_3pt(self, gauge, invPara, source_jobs, interpolator="5"):
        """Compute connected proton U/D quark EMT 3pt functions for source jobs."""
        U = gauge
        latt_info = U.latt_info
        mass, csw, tol, maxiter = invPara
        multigrid = _parse_mg_block([[8, 8, 4, 4]])
        mpi_print(latt_info, f"Proton EMT multigrid block: {multigrid}")
        dirac = core.getDirac(latt_info, mass, tol, maxiter, 1.0, csw, csw, multigrid)
        dirac.loadGauge(U)
        mpi_print(latt_info, "Proton EMT inverter ready.")

        results = []
        for source_job in source_jobs:
            src_idx = source_job.get("src_idx", len(results))
            src_pos = source_job["src_pos"]
            source_t0 = perf_counter()
            mpi_print(latt_info, f"--source_start index={src_idx} src_pos={src_pos}")
            results.append(self._connected_3pt_one_source(dirac, U, source_job, interpolator=interpolator))
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


class ProtonGluonEMT(EMTDisconnectedGluon1pt):
    """Proton workflow alias for the shared flowed gluon EMT 1pt code."""


__all__ = ["ProtonQuarkEMT", "ProtonGluonEMT"]
