r"""Disconnected qTMD/PDF one-point loop measurements.

This module measures hadron-independent stochastic quark loops for qTMD/PDF
operators.  The loops are intended to be combined with pion or proton two-point
functions in downstream analysis:

    C3_disc = < C2_H L_qTMD > - < C2_H > < L_qTMD >.

The first implementation intentionally mirrors the connected qTMD operator
choices already used in ``pion_qTMD_vibe_develop.py``:

    CG_qTMD: coordinate-gauge style spatial displacement, no explicit links.
    CG_PDF:  straight-z displacement, no explicit links.
    GI_PDF:  straight-z covariant displacement through gauge links.

The stochastic estimator uses eta = D^{-1} xi and the loop convention

    L_Gamma,b(q,tau) =
        sum_x exp(i q.x) eta^\dagger(x) Gamma O_b xi(x).

The nonlocal operator O_b acts on the source/noise side.  This convention is
minimal and keeps the first disconnected implementation easy to compare against
the connected PDF/local limits.  A fully gauge-invariant staple qTMD operator is
not implemented here yet.
"""

import numpy as np

from pyquda import getMPIComm
from pyquda.field import LatticeFermion, LatticeGauge
from pyquda_utils import core, phase

from pyquda_measurement_utils.io_corr import save_disconnected_qTMD_1pt_hdf5
from pyquda_measurement_utils.pion_utils_vibe_develop import gamma_stack, my_gammas
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    array_to_numpy,
    effective_n_inversions,
    iter_noise_sources,
    normalize_noise_scheme,
    source_bookkeeping_arrays,
    validate_hierarchical_probing_options,
)

_VALID_OPERATOR_KINDS = {"CG_qTMD", "CG_PDF", "GI_PDF"}


def gi_qtmd_staple_segments(W_index):
    """Return signed nearest-neighbor path segments for a future GI qTMD staple.

    Each segment is ``(direction, signed_steps)`` with direction ``0, 1, 2``
    corresponding to x, y, z.  The path convention is
    z(eta), transverse(b_T), z(b_z - eta).  The actual covariant-shift
    implementation is intentionally not enabled yet.
    """
    b_T, b_z, eta, transverse_direction = [int(round(v)) for v in W_index]
    return [
        (2, eta),
        (transverse_direction, b_T),
        (2, b_z - eta),
    ]


class DisconnectedQuarkqTMD1pt:
    """Hadron-independent stochastic disconnected qTMD/PDF loop measurement."""

    def __init__(self, parameters):
        self.eta = parameters.get("eta", [0])
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.qlist = parameters["qext"]
        self.qlist_PDF = parameters.get("qext_PDF", self.qlist)

        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "global_xyzt_gray_projected_to_evenodd")
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)

    def create_TMD_Wilsonline_index_list_CG(self):
        """Create the connected-code-compatible CG qTMD displacement list."""
        index_list_trans0 = []
        index_list_trans1 = []

        for current_bz in range(0, self.b_z + 1):
            for current_b_T in range(0, self.b_T + 1):
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])

        return self._reorder_wilson_indices(index_list_trans0), self._reorder_wilson_indices(index_list_trans1)

    def _reorder_wilson_indices(self, index_list):
        sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
        reordered = []
        i = 0
        while i < len(sorted_list) - 1:
            current = sorted_list[i]
            next_index = sorted_list[i + 1]
            if abs(current[0] - next_index[0]) > 1 or abs(current[1] - next_index[1]) > 1:
                best_match = next_index
                best_diff = max(abs(current[0] - next_index[0]), abs(current[1] - next_index[1]))
                for candidate in sorted_list[i + 2 :]:
                    diff = max(abs(current[0] - candidate[0]), abs(current[1] - candidate[1]))
                    if diff < best_diff:
                        best_match = candidate
                        best_diff = diff
                if best_match != next_index:
                    best_index = sorted_list.index(best_match)
                    sorted_list[i + 1], sorted_list[best_index] = sorted_list[best_index], sorted_list[i + 1]
            reordered.append(current)
            i += 1

        if i < len(sorted_list):
            reordered.append(sorted_list[-1])
        return reordered

    def create_PDF_Wilsonline_index_list(self):
        """Create the straight-z PDF displacement list."""
        index_list = []

        for current_bz in range(0, self.b_z + 1):
            index_list.append([0, current_bz, 0, 0])

        for current_bz in range(0, self.b_z + 1):
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])

        return index_list

    @staticmethod
    def create_fermion_TMD_CG(fermion, W_index, W_index_previous):
        """Apply the coordinate-gauge qTMD displacement to a fermion field."""
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3]
        z_direction = 2

        previous_b_T = W_index_previous[0]
        previous_bz = W_index_previous[1]

        return fermion.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), z_direction)

    @staticmethod
    def create_fermion_PDF_GI(gauge: LatticeGauge, fermion, W_index, W_index_previous):
        """Apply a straight-z gauge-invariant PDF displacement to a fermion."""
        current_bz = W_index[1]
        previous_bz = W_index_previous[1]

        if current_bz - previous_bz == 0:
            return fermion
        if current_bz - previous_bz == 1:
            return gauge.pure_gauge.covDev(fermion, 2)
        if current_bz - previous_bz == -1:
            return gauge.pure_gauge.covDev(fermion, 6)
        raise ValueError("Invalid shift for PDF Wilson line")

    def _contract_one_operator_list(self, latt_info, gauge, eta, xi, phases, W_index_list, operator_kind):
        xp = _get_xp_from_array(xi.data)
        phases = _asarray_on_queue(phases, xp, xi.data)
        gamma_ls = gamma_stack(xi.data)

        loops = []
        shifted_xi = xi.copy()
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract disconnected {operator_kind} {iW + 1}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list[iW - 1]

            if operator_kind in {"CG_qTMD", "CG_PDF"}:
                if operator_kind == "CG_PDF" and W_index[1] in {0, -1}:
                    shifted_xi = xi.copy()
                    W_index_previous = [0, 0, 0, 0]
                if operator_kind == "CG_qTMD" and W_index[3] != W_index_previous[3]:
                    shifted_xi = xi.copy()
                    W_index_previous = [0, 0, 0, W_index[3]]
                shifted_xi = self.create_fermion_TMD_CG(shifted_xi, W_index, W_index_previous)
            elif operator_kind == "GI_PDF":
                if W_index[1] in {0, -1}:
                    shifted_xi = xi.copy()
                    W_index_previous = [0, 0, 0, 0]
                shifted_xi = self.create_fermion_PDF_GI(gauge, shifted_xi, W_index, W_index_previous)
            else:
                raise ValueError(f"Unsupported operator_kind {operator_kind!r}")

            local_loop = xp.einsum("wtzyxia,gij,wtzyxja->gwtzyx", eta.data.conj(), gamma_ls, shifted_xi.data, optimize=True)
            loop = xp.einsum("qwtzyx,gwtzyx->gqt", phases, local_loop, optimize=True)
            loops.append(core.gatherLattice(array_to_numpy(loop), [2, -1, -1, -1]))
            del local_loop, loop

        return np.asarray(loops)

    def measure_1pt(
        self,
        gauge: LatticeGauge,
        invPara,
        randPara,
        tag: str,
        operator_kind: str = "GI_PDF",
    ):
        """Measure disconnected qTMD/PDF one-point loops."""
        if operator_kind not in _VALID_OPERATOR_KINDS:
            raise ValueError(f"operator_kind should be one of {_VALID_OPERATOR_KINDS}, got {operator_kind!r}")

        n_vec, n_zn, randseed = randPara
        mass, csw, tol, maxiter = invPara
        U = gauge
        latt_info = U.latt_info
        global_size = latt_info.global_size
        Ns3 = global_size[0] * global_size[1] * global_size[2]

        xp = _get_xp_from_array(LatticeFermion(latt_info).data)
        xp.random.seed(randseed)

        xi_0, nu = 1.0, 1.0
        multigrid = [[
            max(1, global_size[0] // 1),
            max(1, global_size[1] // 1),
            max(1, global_size[2] // 2),
            max(1, global_size[3] // 8),
        ]]
        dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw, csw, multigrid)
        dirac.loadGauge(U)
        mpi_print(latt_info, "Disconnected qTMD inverter ready.")

        qlist = self.qlist if operator_kind == "CG_qTMD" else self.qlist_PDF
        qext_xyz = [[q[0], q[1], q[2]] for q in qlist]
        phases_q = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
        if operator_kind == "CG_qTMD":
            W_index_list_dir0, W_index_list_dir1 = self.create_TMD_Wilsonline_index_list_CG()
            W_index_list = W_index_list_dir0 + W_index_list_dir1
        else:
            W_index_list = self.create_PDF_Wilsonline_index_list()

        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors)
        source_bookkeeping = source_bookkeeping_arrays(n_eff)
        loop_pervec = None

        for vec_picked, base_idx, hp_idx, xi in iter_noise_sources(latt_info, n_vec, n_zn, self.noise_scheme, self.hp_num_vectors, self.hp_ordering):
            mpi_print(latt_info, f"vec {vec_picked} base {base_idx} hp {hp_idx}")
            source_bookkeeping["base_noise_index"][vec_picked] = base_idx
            source_bookkeeping["hp_index"][vec_picked] = hp_idx
            dirac.loadGauge(U)
            eta = dirac.invert(xi)

            loops = self._contract_one_operator_list(latt_info, U, eta, xi, phases_q, W_index_list, operator_kind)
            loops = getMPIComm().bcast(loops, root=0)
            if loop_pervec is None:
                loop_pervec = np.zeros(
                    (n_eff, loops.shape[0], loops.shape[1], loops.shape[2], loops.shape[3]),
                    dtype=np.complex128,
                )
            loop_pervec[vec_picked] = loops
            del eta, loops

        mpi_print(latt_info, "disconnected qTMD random vectors done.")

        loop_avg = np.mean(loop_pervec, axis=0) / Ns3
        attrs = {
            "measurement": "disconnected_qTMD_1pt",
            "operator_kind": operator_kind,
            "qext": np.asarray(qlist, dtype=np.int32),
            "W_index_list": np.asarray(W_index_list, dtype=np.int32),
            "gamma_list": np.asarray(my_gammas, dtype="S"),
            "volume_norm": Ns3,
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
            "loop_convention": "eta_dagger_Gamma_O_b_xi",
        }
        save_disconnected_qTMD_1pt_hdf5(
            tag,
            loop_pervec,
            loop_avg,
            my_gammas,
            qlist,
            W_index_list,
            attrs=attrs,
            source_bookkeeping=source_bookkeeping,
        )

        return loop_avg
