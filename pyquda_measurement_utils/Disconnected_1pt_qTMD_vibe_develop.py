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

_VALID_NOISE_SCHEMES = {"zn", "hierarchical_probing"}
_VALID_HP_ORDERINGS = {
    "global_xyzt_gray_projected_to_evenodd",
    "spatial_xyz_then_t_gray_projected_to_evenodd",
}
_VALID_OPERATOR_KINDS = {"CG_qTMD", "CG_PDF", "GI_PDF"}


def _array_to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if type(arr).__module__.split(".")[0] == "cupy":
        return arr.get()
    if type(arr).__module__.split(".")[0] == "dpnp":
        import dpnp

        return dpnp.asnumpy(arr)
    return np.asarray(arr)


def _normalize_noise_scheme(noise_scheme: str) -> str:
    scheme = str(noise_scheme).strip().lower()
    if scheme not in _VALID_NOISE_SCHEMES:
        raise ValueError(f"noise_scheme should be one of {_VALID_NOISE_SCHEMES}, got {noise_scheme!r}")
    return scheme


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return int(value - 1).bit_length()


class DisconnectedQuarkqTMD1pt:
    """Hadron-independent stochastic disconnected qTMD/PDF loop measurement."""

    def __init__(self, parameters):
        self.eta = parameters.get("eta", [0])
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.qlist = parameters["qext"]
        self.qlist_PDF = parameters.get("qext_PDF", self.qlist)

        self.noise_scheme = _normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "global_xyzt_gray_projected_to_evenodd")
        if self.hp_ordering not in _VALID_HP_ORDERINGS:
            raise ValueError(f"hp_ordering should be one of {_VALID_HP_ORDERINGS}, got {self.hp_ordering!r}")
        if not _is_power_of_two(self.hp_num_vectors):
            raise ValueError(f"hp_num_vectors should be a positive power of two, got {self.hp_num_vectors}")

    @staticmethod
    def make_zn_noise_fermion(latt_info, n: int = 2) -> LatticeFermion:
        """Create one stochastic fermion source with Z_n phases."""
        xi = LatticeFermion(latt_info)
        xp = _get_xp_from_array(xi.data)
        r = xp.random.randint(0, n, size=xi.data.shape)
        xi.data[:] = xp.exp(2j * xp.pi * r / n).astype(xi.data.dtype)
        return xi

    @staticmethod
    def _hierarchical_gray_index(latt_info, hp_ordering: str):
        coords = latt_info.coordinate()
        x, y, z, t = [np.asarray(coords[mu], dtype=np.int64) for mu in range(4)]
        Gx, Gy, Gz, _Gt = latt_info.global_size

        if hp_ordering == "global_xyzt_gray_projected_to_evenodd":
            site_id = x + Gx * (y + Gy * (z + Gz * t))
            return site_id ^ (site_id >> 1)

        if hp_ordering == "spatial_xyz_then_t_gray_projected_to_evenodd":
            spatial_id = x + Gx * (y + Gy * z)
            spatial_gray = spatial_id ^ (spatial_id >> 1)
            time_gray = t ^ (t >> 1)
            spatial_bits = _ceil_log2(Gx * Gy * Gz)
            return spatial_gray | (time_gray << spatial_bits)

        raise ValueError(f"Unsupported hp_ordering {hp_ordering!r}")

    @classmethod
    def _hierarchical_probe_pattern(cls, latt_info, hp_idx: int, hp_ordering: str):
        """Build a site-only Rademacher probing vector in even-odd layout."""
        if hp_idx == 0:
            return np.ones_like(latt_info.coordinate(0), dtype=np.float64)

        gray_id = cls._hierarchical_gray_index(latt_info, hp_ordering)
        parity = np.zeros_like(gray_id, dtype=bool)
        mask = int(hp_idx)
        bit = 1
        while bit <= mask:
            if mask & bit:
                parity ^= (gray_id & bit) != 0
            bit <<= 1
        return np.where(parity, -1.0, 1.0)

    @classmethod
    def apply_hierarchical_probe(cls, xi: LatticeFermion, hp_idx: int, hp_ordering: str) -> LatticeFermion:
        """Multiply a base stochastic source by one hierarchical probing vector."""
        if hp_idx == 0:
            return xi.copy()

        probed = xi.copy()
        pattern = cls._hierarchical_probe_pattern(xi.latt_info, hp_idx, hp_ordering)
        pattern = _asarray_on_queue(pattern, _get_xp_from_array(xi.data), xi.data)
        probed.data[:] *= pattern[..., None, None]
        return probed

    def _iter_noise_sources(self, latt_info, n_vec: int, n_zn: int):
        """Yield effective stochastic sources with optional hierarchical probing."""
        if self.noise_scheme == "zn":
            for base_idx in range(n_vec):
                yield base_idx, base_idx, 0, self.make_zn_noise_fermion(latt_info, n=n_zn)
            return

        for base_idx in range(n_vec):
            base_noise = self.make_zn_noise_fermion(latt_info, n=n_zn)
            for hp_idx in range(self.hp_num_vectors):
                effective_idx = base_idx * self.hp_num_vectors + hp_idx
                yield effective_idx, base_idx, hp_idx, self.apply_hierarchical_probe(base_noise, hp_idx, self.hp_ordering)

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
            loops.append(core.gatherLattice(_array_to_numpy(loop), [2, -1, -1, -1]))
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

        effective_n_inversions = n_vec * self.hp_num_vectors if self.noise_scheme == "hierarchical_probing" else n_vec
        source_index = np.arange(effective_n_inversions, dtype=np.int32)
        base_noise_index = np.zeros(effective_n_inversions, dtype=np.int32)
        hp_index = np.zeros(effective_n_inversions, dtype=np.int32)
        loop_pervec = None

        for vec_picked, base_idx, hp_idx, xi in self._iter_noise_sources(latt_info, n_vec, n_zn):
            mpi_print(latt_info, f"vec {vec_picked} base {base_idx} hp {hp_idx}")
            base_noise_index[vec_picked] = base_idx
            hp_index[vec_picked] = hp_idx
            dirac.loadGauge(U)
            eta = dirac.invert(xi)

            loops = self._contract_one_operator_list(latt_info, U, eta, xi, phases_q, W_index_list, operator_kind)
            loops = getMPIComm().bcast(loops, root=0)
            if loop_pervec is None:
                loop_pervec = np.zeros(
                    (effective_n_inversions, loops.shape[0], loops.shape[1], loops.shape[2], loops.shape[3]),
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
            "effective_n_inversions": effective_n_inversions,
            "n_zn": n_zn,
            "rand_seed": randseed,
            "noise_scheme": self.noise_scheme,
            "hp_num_vectors": self.hp_num_vectors,
            "hp_ordering": self.hp_ordering,
            "loop_convention": "eta_dagger_Gamma_O_b_xi",
        }
        source_bookkeeping = {
            "source_index": source_index,
            "base_noise_index": base_noise_index,
            "hp_index": hp_index,
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
