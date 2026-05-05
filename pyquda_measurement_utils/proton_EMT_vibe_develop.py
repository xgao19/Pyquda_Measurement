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
building blocks are inherited from the meson EMT implementation.  Thus the
same comments in ``pion_EMT_vibe_develop.py`` apply: the quark 1pt output
contains ``avg/Tmunu/T11`` through ``T44`` for reconstructing the zero-momentum
``bar_chi overleftrightarrow{not D} chi`` normalization, and the gluon 1pt
output provides the flowed gluonic EMT building block.  The final
renormalized gradient-flow EMT is assembled in analysis from connected 3pt,
quark 1pt, and gluon 1pt data.

Limitations
-----------
This module computes connected proton EMT three-point functions for U and D
insertions.  Disconnected diagrams, renormalization coefficients, vacuum
subtractions, and flavor mixing are intentionally left to separate workflows or
analysis code.
"""

import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import LatticePropagator
from pyquda_utils import core, gamma, source, phase

from pyquda_measurement_utils.pion_EMT_vibe_develop import GluonEMT, QuarkEMT
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_bw_seq_pyquda
from pyquda_measurement_utils.io_corr import save_emt_quark_3pt_hdf5
from pyquda_measurement_utils.proton_qTMD_pyquda import my_gammas, proton_TMD
from pyquda_measurement_utils.tools import mpi_print


class ProtonQuarkEMT(QuarkEMT):
    """Connected proton EMT plus inherited stochastic quark 1pt utilities."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.pol_list = parameters["pol"]
        self.t_insert = parameters["t_insert"]
        self.save_propagators = parameters.get("save_propagators", False)
        self.boost_in = parameters.get("boost_in", self.pos_boost)
        self.boost_out = parameters.get("boost_out", self.pos_boost)

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
        return helper.contract_2pt_TMD(latt_info, prop, phases_2pt, tag, interpolator=interpolator)

    @staticmethod
    def _seq_to_prop(latt_info, seq_data):
        seq_prop = LatticePropagator(latt_info)
        seq_prop.data = seq_data
        return seq_prop

    @classmethod
    def _covdev_sym_seq(cls, U_f, seq_data, mu):
        seq_prop = cls._seq_to_prop(U_f.latt_info, seq_data)
        return cls._covdev_sym_prop(U_f, seq_prop, mu).data

    @classmethod
    def get_C3_chi_proton(cls, latt_info, prop_fw, seq_data, phases_3pt, t0):
        scalar_field = contract("wtzyxjicf,wtzyxijfc->wtzyx", seq_data, prop_fw.data)
        slice_t = core.gatherLattice(
            contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
            [1, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.roll(np.array(slice_t), -t0, axis=-1)

    @classmethod
    def get_C3_Tmunu_symmetrized_proton(cls, U_f, prop_fw, seq_data, phases_3pt, t0):
        Nq = len(phases_3pt)
        Nt = U_f.latt_info.global_size[3]
        C3_Tmunu = np.zeros((Nq, 4, 4, Nt), dtype=np.complex128)
        D_gammas_local = cls._dirac_gammas_for(prop_fw.data)

        for mu in range(4):
            D_fw = cls._covdev_sym_prop(U_f, prop_fw, mu)
            for nu in range(4):
                gamma_D_fw = contract("ab,wtzyxbdij->wtzyxadij", D_gammas_local[nu], D_fw.data)
                scalar_field = 0.5 * contract("wtzyxjicf,wtzyxijfc->wtzyx", seq_data, gamma_D_fw)
                slice_t = core.gatherLattice(
                    contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
                    [1, -1, -1, -1],
                )
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[:, mu, nu] += np.roll(np.array(slice_t), -t0, axis=-1)

        for mu in range(4):
            leftD_seq = cls._covdev_sym_seq(U_f, seq_data, mu)
            for nu in range(4):
                gamma_fw = contract("ab,wtzyxbdij->wtzyxadij", D_gammas_local[nu], prop_fw.data)
                scalar_field = -0.5 * contract("wtzyxjicf,wtzyxijfc->wtzyx", leftD_seq, gamma_fw)
                slice_t = core.gatherLattice(
                    contract("qwtzyx,wtzyx->qt", phases_3pt, scalar_field).get(),
                    [1, -1, -1, -1],
                )
                slice_t = getMPIComm().bcast(slice_t, root=0)
                C3_Tmunu[:, mu, nu] += np.roll(np.array(slice_t), -t0, axis=-1)

        for mu in range(4):
            for nu in range(mu + 1, 4):
                C3_Tmunu[:, mu, nu] = 0.5 * (C3_Tmunu[:, mu, nu] + C3_Tmunu[:, nu, mu])
                C3_Tmunu[:, nu, mu] = C3_Tmunu[:, mu, nu]

        return C3_Tmunu

    def connected_3pt(
        self,
        gauge,
        invPara,
        src_pos,
        t_separations,
        spin,
        tag,
        c2_tag=None,
        interpolator="5",
    ):
        """Compute connected proton U/D quark EMT 3pt functions."""
        assert spin in [0, 1, 2, 5]
        U = gauge
        stepsize = self.flow_epsilon
        Nsteps = self.flow_steps
        latt_info = U.latt_info
        Nt = latt_info.global_size[3]
        mass, csw, tol, maxiter = invPara
        t0 = src_pos[3]

        dirac = core.getDirac(latt_info, mass, tol, maxiter, 1.0, csw, csw, [[8, 8, 4, 4]])
        dirac.loadGauge(U)
        mpi_print(latt_info, "Proton EMT inverter ready.")

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
                seq_bw = create_bw_seq_pyquda(
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
                    seq_prop_flow = self._seq_to_prop(latt_info, seq_bw[pol_idx].copy())
                    U_f = U.copy()
                    U_f.setAntiPeriodicT()

                    for step in range(Nsteps + 1):
                        mpi_print(
                            latt_info,
                            f"proton EMT contraction flavor={flavor_name} pol={pol} t_sep={t_sep} step={step}",
                        )
                        C3_chi[flavor_idx, pol_idx, n_ts, step] += self.get_C3_chi_proton(
                            latt_info,
                            prop_fw_flow,
                            seq_prop_flow.data,
                            phases_3pt,
                            t0,
                        )
                        C3_Tmunu[flavor_idx, pol_idx, n_ts, step] += self.get_C3_Tmunu_symmetrized_proton(
                            U_f,
                            prop_fw_flow,
                            seq_prop_flow.data,
                            phases_3pt,
                            t0,
                        )

                        if step < Nsteps:
                            prop_fw_flow, seq_prop_flow = self._advance_flowed_props(
                                U_f,
                                prop_fw_flow,
                                seq_prop_flow,
                                step,
                                stepsize,
                                Nsteps,
                            )

                    del U_f, prop_fw_flow, seq_prop_flow
                del seq_bw

        attrs = {
            "measurement": "proton_quark_3pt",
            "spin": spin,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "src_t": t0,
            "interpolator": interpolator,
            "flavor_axis": "0=U,1=D",
            "polarization_axis": ",".join(self.pol_list),
            "n_qext": Nq,
            "connected_only": True,
            "c2_selected_momentum_index": zero_mom_idx,
            "c2_selected_momentum": self.pilist[zero_mom_idx],
        }
        save_emt_quark_3pt_hdf5(tag, C2_selected, C3_chi, C3_Tmunu, momentum_transfer_list=self.qlist, attrs=attrs)
        return C2, C3_chi, C3_Tmunu


class ProtonGluonEMT(GluonEMT):
    """Proton workflow alias for the shared flowed gluon EMT 1pt code."""


__all__ = ["ProtonQuarkEMT", "ProtonGluonEMT"]
