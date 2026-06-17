"""Shared flowed EMT one-point loop measurements.

This module contains the hadron-independent quark and gluon 1pt pieces used by
both pion and proton EMT workflows.  These loops are the building blocks for
disconnected diagrams in analysis:

    C3_disc = < C2 L > - < C2 > < L >.

They also provide the flowed-fermion kinetic expectation value used for ringed
fermion normalization through the diagonal quark ``Tmunu`` components.
"""

import numpy as np
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import LatticeGauge, LatticePropagator, LatticeFermion, MultiLatticeFermion
from pyquda_utils import core, gamma, phase, convert
from pyquda_comm.array import arrayIdentity, arrayZeros

from pyquda_measurement_utils.io_corr import save_emt_quark_1pt_hdf5, save_emt_gluon_1pt_hdf5
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    effective_n_inversions,
    iter_noise_sources,
    normalize_noise_scheme,
    source_bookkeeping_arrays,
    validate_hierarchical_probing_options,
)

_VALID_FLOW_TYPES = {"wilson", "symanzik"}
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]
my_pyquda_gammas = [gamma.gamma(idx) for idx in pyquda_gammas_order]
D_GAMMA_IDS = [1, 2, 4, 8]
D_gammas = [gamma.gamma(idx) for idx in D_GAMMA_IDS]
G5 = gamma.gamma(15)


def _gamma_matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def _array_on_backend(val, ref_arr):
    xp = _get_xp_from_array(ref_arr)
    if type(val).__module__.split(".")[0] == xp.__name__:
        return val
    if hasattr(val, "get"):
        val = val.get()
    return _asarray_on_queue(val, xp, ref_arr)


def _gamma_list_on_backend(gamma_list, ref_arr):
    xp = _get_xp_from_array(ref_arr)
    gamma_arrays = [_array_on_backend(_gamma_matrix(gamma_item), ref_arr) for gamma_item in gamma_list]
    return xp.stack(gamma_arrays)


def _normalize_flow_type(flow_type: str) -> str:
    flow = str(flow_type).strip().lower()
    if flow not in _VALID_FLOW_TYPES:
        raise ValueError(f"flow_type should be one of {_VALID_FLOW_TYPES}, got {flow_type!r}")
    return flow


def _flow_times(flow_epsilon, flow_steps):
    return np.arange(flow_steps + 1, dtype=np.float64) * float(flow_epsilon)


class EMTDisconnectedQuark1pt:
    """Hadron-independent stochastic flowed quark EMT loop measurement."""

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
        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "global_xyzt_gray_projected_to_evenodd")
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)

    @staticmethod
    def _gamma5_for(ref_arr):
        return _array_on_backend(_gamma_matrix(G5), ref_arr)

    @staticmethod
    def _gamma_stack_for(ref_arr):
        return _gamma_list_on_backend(my_pyquda_gammas, ref_arr)

    @staticmethod
    def _dirac_gammas_for(ref_arr):
        return _gamma_list_on_backend(D_gammas, ref_arr)

    @classmethod
    def _get_interpolator_gamma_for(cls, interpolator, ref_arr):
        if interpolator not in my_gammas:
            raise ValueError(f"Unsupported interpolator {interpolator!r}. Expected one of {my_gammas}.")
        return _array_on_backend(_gamma_matrix(my_pyquda_gammas[my_gammas.index(interpolator)]), ref_arr)

    @staticmethod
    def _impose_P_Breit_slice(U: LatticeGauge, complex_field, phases_3pt):
        """Project a local scalar field to spatial momenta and keep time."""
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
        """Build flowed quark 1pt EMT and scalar diagnostics."""
        Nt = U_f.latt_info.global_size[3]

        CHI = np.zeros([2, len(qlist), Nt], dtype=np.complex128)
        dot_xi_eta = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), eta.data)
        CHI[0] = self._impose_P_Breit_slice(U_f, dot_xi_eta, phases_3pt)
        dot_xi_xi = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), xi.data)
        CHI[1] = self._impose_P_Breit_slice(U_f, dot_xi_xi, phases_3pt)

        Tmunu = np.zeros([4, 4, len(qlist), Nt], dtype=np.complex128)
        U_f.gauge_dirac.loadGauge(U_f)
        D_gammas_local = self._dirac_gammas_for(eta.data)
        for mu in range(4):
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            for nu in range(4):
                Y = contract("ab,...bc->...ac", D_gammas_local[nu], tmp.data)
                complex_field = contract("...sc,...sc->...", xi.data.conj(), Y)
                Tmunu[nu, mu] += -0.5 * self._impose_P_Breit_slice(U_f, complex_field, phases_3pt)

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
        """Compute quark flowed 1pt observables with stochastic sources."""
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

        rng_probe = LatticeFermion(latt_info)
        xp = _get_xp_from_array(rng_probe.data)
        xp.random.seed(randseed)

        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors)
        Tmunu = np.zeros([n_eff, 4, 4, len(self.qlist), Nsteps + 1, Nt], dtype=np.complex128)
        CHI = np.zeros([n_eff, 2, len(self.qlist), Nsteps + 1, Nt], dtype=np.complex128)
        source_bookkeeping = source_bookkeeping_arrays(n_eff)
        qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
        phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
        for vec_picked, base_idx, hp_idx, xi in iter_noise_sources(latt_info, n_vec, n_zn, self.noise_scheme, self.hp_num_vectors, self.hp_ordering):
            mpi_print(U.latt_info, f"vec {vec_picked} base {base_idx} hp {hp_idx}")
            source_bookkeeping["base_noise_index"][vec_picked] = base_idx
            source_bookkeeping["hp_index"][vec_picked] = hp_idx
            dirac.loadGauge(U)
            eta = dirac.invert(xi)

            U_f = U.copy()
            U_f.setAntiPeriodicT()

            for step in range(Nsteps + 1):
                mpi_print(U_f.latt_info, f"calc Tmunu, step = {step}")
                U_f.gauge_dirac.loadGauge(U_f)
                tmpt, tmps = self._get_Tmunu_symmetrized_P_Breit_slice(U_f, xi, eta, self.qlist, phases_3pt)
                Tmunu[vec_picked, :, :, :, step, :] += tmpt
                CHI[vec_picked, :, :, step, :] += tmps

                if Nsteps > 0 and step == 0:
                    temp = convert.multiField([xi, eta])
                    temp_flow = U_f.gradientFlow(temp, self.flow_type, 10, stepsize / 10)
                    xi, eta = temp_flow[0], temp_flow[1]

                elif Nsteps > 0 and step < Nsteps:
                    temp = convert.multiField([xi, eta])
                    temp_flow = U_f.gradientFlow(temp, self.flow_type, 1, stepsize)
                    xi, eta = temp_flow[0], temp_flow[1]

        mpi_print(U.latt_info, "random vectors done.")

        attrs = {
            "measurement": "quark_1pt",
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "pf": np.asarray(self.pf, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "volume_norm": Ns3,
            "upper_triangle_only": True,
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
            "n_vec": n_vec,
            "n_base_noise": n_vec,
            "effective_n_inversions": n_eff,
            "n_zn": n_zn,
            "rand_seed": randseed,
            "noise_scheme": self.noise_scheme,
            "hp_num_vectors": self.hp_num_vectors,
            "hp_ordering": self.hp_ordering,
        }
        Tmunu_avg = np.mean(Tmunu, axis=0) / Ns3
        CHI_avg = np.mean(CHI, axis=0) / Ns3
        save_emt_quark_1pt_hdf5(
            tag,
            Tmunu,
            CHI,
            Tmunu_avg,
            CHI_avg,
            attrs=attrs,
            source_bookkeeping=source_bookkeeping,
        )

        return Tmunu_avg, CHI_avg

    @staticmethod
    def _covdev_sym_prop(U_f: LatticeGauge, prop: LatticePropagator, mu: int):
        """Apply the symmetric covariant derivative to a propagator."""
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
        """Construct the left-acting derivative on ``dst2 = gamma5 S^dagger gamma5``."""
        D_y = cls._covdev_sym_prop(U_f, prop, mu)
        D_y_dag = D_y.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7)
        G5_local = cls._gamma5_for(prop.data)
        leftD_dst2 = contract("ab,wtzyxbcij,cd->wtzyxadij", G5_local, D_y_dag, G5_local)
        return leftD_dst2

    @staticmethod
    def _flow_two_props_pyquda(U_f: LatticeGauge, prop_a: LatticePropagator, prop_b: LatticePropagator, stepsize: float, Nsteps: int, flow_type: str = "wilson"):
        """Flow two propagators simultaneously on the same flowed gauge background."""
        mf_a = convert.propagatorToMultiFermion(prop_a)
        mf_b = convert.propagatorToMultiFermion(prop_b)

        L5_a = mf_a.L5
        L5_b = mf_b.L5
        assert L5_a == L5_b

        fields = [mf_a[idx] for idx in range(L5_a)] + [mf_b[idx] for idx in range(L5_b)]
        packed = convert.multiField(fields)
        del fields, mf_a, mf_b, prop_a, prop_b

        packed_flow = U_f.gradientFlow(packed, flow_type, Nsteps, stepsize)
        del packed

        mf_a_flow = MultiLatticeFermion(U_f.latt_info, L5_a, packed_flow.data[:L5_a])
        mf_b_flow = MultiLatticeFermion(U_f.latt_info, L5_b, packed_flow.data[L5_a:L5_a + L5_b])

        prop_a_flow = convert.multiFermionToPropagator(mf_a_flow)
        prop_b_flow = convert.multiFermionToPropagator(mf_b_flow)
        prop_a_flow._packed_flow_owner = packed_flow
        prop_b_flow._packed_flow_owner = packed_flow
        del mf_a_flow, mf_b_flow
        return prop_a_flow, prop_b_flow

    def _advance_flowed_props(self, U_f, prop_fw_flow, seq_bw_prop_flow, step, stepsize, Nsteps):
        """Advance the flowed propagators using the quark-flow schedule."""
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


class EMTDisconnectedGluon1pt:
    """Hadron-independent flowed gluon EMT loop measurement."""

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
        """Advance the flowed gauge field using the gluon-flow schedule."""
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
        """Construct the traceless clover field strength F_{mu nu}."""
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
        """Build all independent F_{mu nu} and fill the antisymmetric table."""
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
        """Compute flowed gluon 1pt EMT observables."""
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
                    tmp = arrayZeros((2, Lt, Lz, Ly, Lx // 2), U.data.dtype, U.location)

                    for rho in range(4):
                        if rho == mu or rho == nu:
                            continue
                        F_mr = F[mu][rho]
                        F_nr = F[nu][rho]
                        tmp += contract("...ab,...ba->...", F_mr, F_nr)

                    slice_t = core.gatherLattice(
                        contract("qwtzyx, wtzyx -> qt", phases_3pt, tmp).get(),
                        [1, -1, -1, -1],
                    )
                    if U.latt_info.mpi_rank == 0:
                        Tmunu_t[mu, nu, :, step, :] += 2.0 * slice_t

            mpi_print(latt_info, f"{self.flow_type}Flow step = {step}")
            self._advance_flowed_gauge(U_flow, step, stepsize, Nsteps)

        Tmunu_t /= Ns3
        attrs = {
            "measurement": "gluon_1pt",
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "pf": np.asarray(self.pf, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "volume_norm": Ns3,
            "upper_triangle_only": True,
        }
        save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=attrs)

        return Tmunu_t


__all__ = [
    "EMTDisconnectedQuark1pt",
    "EMTDisconnectedGluon1pt",
    "my_gammas",
]
