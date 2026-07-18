r"""Disconnected qTMD/PDF one-point loop measurements.

This module measures hadron-independent stochastic quark loops for qTMD/PDF
operators.  The loops are intended to be combined with pion or proton two-point
functions in downstream analysis:

    C3_disc = < C2_H L_qTMD > - < C2_H > < L_qTMD >.

The operator choices mirror the connected qTMD definitions already used in
``pion_qTMD_vibe_develop.py``:

    CG_qTMD: coordinate-gauge style spatial displacement, no explicit links.
    CG_PDF:  straight-z displacement, no explicit links.
    GI_PDF:  straight-z covariant displacement through gauge links.
    GI_qTMD: fixed-length staple covariant displacement through gauge links.

The stochastic estimator uses eta = D^{-1} xi and the loop convention

    L_Gamma,b(q,tau) =
        sum_x exp(i q.x) xi^\dagger(x) Gamma O_b eta(x).

The nonlocal operator O_b acts on the solved propagator side.  The GI qTMD staple
uses the fixed-total-length convention

    x -> x + (eta + b_z / 2) zhat
      -> x + (eta + b_z / 2) zhat + b_T e_perp
      -> x + b_z zhat + b_T e_perp,

so the staple length is 2 * eta + b_T for every even b_z with
eta >= abs(b_z) / 2.
"""

import os
from pathlib import Path

import h5py
import numpy as np

from pyquda import getMPIComm
from pyquda.field import LatticeFermion, LatticeGauge, LatticeLink
from pyquda_utils import core, phase
from pyquda_utils.convert import fermionToLink, linkToFermion

from pyquda_measurement_utils.pion_utils_vibe_develop import gamma_stack, my_gammas
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    COUNTER_NOISE_ALGORITHM,
    append_completed_base,
    array_to_numpy,
    base_part_ranges,
    canonical_temp_path,
    discover_shard_layout,
    effective_n_inversions,
    hp_vectors_per_base,
    iter_noise_base_hp_interval,
    iter_validated_shard_parts,
    normalize_noise_scheme,
    part_source_bookkeeping,
    prepare_sample_log,
    selected_base_range,
    shard_part_attrs,
    shard_part_path,
    validate_hierarchical_probing_options,
    write_raw_part_hdf5,
)

_VALID_OPERATOR_KINDS = {"CG_qTMD", "CG_PDF", "GI_PDF", "GI_qTMD"}
QTMD_SCHEMA_VERSION = 3
QTMD_LOOP_CONVENTION = "xi_dagger_Gamma_O_b_eta"
QTMD_TRACE_TARGET = "Tr[P_qtau Gamma O_b Dinv]"


def create_gi_qtmd_wilsonline_index_lists(eta_list, max_b_z, max_b_T):
    """Create fixed-length GI qTMD Wilson-index lists for transverse x/y."""
    index_list_trans0 = []
    index_list_trans1 = []
    for eta in eta_list:
        eta = int(eta)
        for current_bz in range(0, int(max_b_z) + 1, 2):
            if eta < current_bz // 2:
                continue
            for current_b_T in range(0, int(max_b_T) + 1):
                index_list_trans0.append([current_b_T, current_bz, eta, 0])
                index_list_trans1.append([current_b_T, current_bz, eta, 1])
                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, eta, 0])
                    index_list_trans1.append([current_b_T, -current_bz, eta, 1])
    return index_list_trans0, index_list_trans1


def gi_qtmd_staple_segments(W_index):
    """Return signed nearest-neighbor path segments for a GI qTMD staple.

    Each segment is ``(direction, signed_steps)`` with direction ``0, 1, 2``
    corresponding to x, y, z.  The fixed-staple-length convention is
    z(eta + b_z / 2), transverse(b_T), z(b_z / 2 - eta).
    """
    b_T, b_z, eta, transverse_direction = [int(round(v)) for v in W_index]
    if b_T < 0:
        raise ValueError("GI_qTMD requires non-negative b_T")
    if b_z % 2 != 0:
        raise ValueError("GI_qTMD requires even b_z in the fixed-staple-length convention")
    if eta < 0:
        raise ValueError("GI_qTMD requires non-negative eta")
    if eta < abs(b_z) // 2:
        raise ValueError("GI_qTMD requires eta >= abs(b_z) / 2")
    if transverse_direction not in {0, 1}:
        raise ValueError("GI_qTMD transverse_direction should be 0 or 1")

    half_bz = b_z // 2
    return [
        (2, eta + half_bz),
        (transverse_direction, b_T),
        (2, half_bz - eta),
    ]


def _apply_signed_covariant_shift(gauge, fermion, direction, steps):
    """Transport a field while constructing a cached staple link."""
    shifted = fermion
    if steps > 0:
        for _ in range(steps):
            shifted = gauge.pure_gauge.covDev(shifted, direction)
    elif steps < 0:
        for _ in range(-steps):
            shifted = gauge.pure_gauge.covDev(shifted, direction + 4)
    return shifted


def _transport_staple_field(gauge, fermion, W_index):
    """Transport along the geometric staple returned by ``gi_qtmd_staple_segments``.

    Since ``D_mu psi(x) = U_mu(x) psi(x + mu)``, composed covariant
    shifts act on the endpoint field in the reverse order of the geometric
    Wilson path.
    """
    shifted = fermion.copy()
    for direction, steps in reversed(gi_qtmd_staple_segments(W_index)):
        shifted = _apply_signed_covariant_shift(gauge, shifted, direction, steps)
    return shifted


def build_gi_qtmd_staple_link(gauge: LatticeGauge, W_index):
    """Build a gauge-only staple transporter matching direct covDev convention."""
    link = LatticeLink(gauge.latt_info)
    link_as_fermion = linkToFermion(link)
    transported = _transport_staple_field(gauge, link_as_fermion, W_index)
    return fermionToLink(transported)


def build_gi_qtmd_staple_links(gauge: LatticeGauge, W_index_list):
    """Build reusable gauge-only staple transporters for a Wilson-index list."""
    return {tuple(W_index): build_gi_qtmd_staple_link(gauge, W_index) for W_index in W_index_list}


def create_fermion_TMD_GI_from_link(staple_link: LatticeLink, fermion: LatticeFermion, W_index):
    """Apply a cached GI qTMD staple transporter to the endpoint fermion."""
    b_T, b_z, _eta, transverse_direction = [int(round(v)) for v in W_index]
    endpoint = fermion.shift(b_T, transverse_direction).shift(b_z, 2)
    shifted = LatticeFermion(fermion.latt_info)
    xp = _get_xp_from_array(fermion.data)
    shifted.data[:] = xp.einsum("wtzyxab,wtzyxib->wtzyxia", staple_link.data, endpoint.data, optimize=True)
    return shifted


def _contract_xi_dagger_gamma_shifted_eta(xi_data, gamma_ls, shifted_eta_data):
    """Contract xi^dagger Gamma shifted_eta at every lattice site."""
    xp = _get_xp_from_array(xi_data)
    return xp.einsum(
        "wtzyxia,gij,wtzyxja->gwtzyx",
        xi_data.conj(),
        gamma_ls,
        shifted_eta_data,
        optimize=True,
    )


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
        if parameters.get("config_num") is None:
            raise ValueError("config_num is required for counter-based disconnected noise")
        self.config_num = int(parameters["config_num"])
        self.gauge_preprocessing = parameters.get("gauge_preprocessing", "unspecified")
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

    def create_TMD_Wilsonline_index_list_GI(self):
        """Create the fixed-length gauge-invariant qTMD staple list."""
        return create_gi_qtmd_wilsonline_index_lists(self.eta, self.b_z, self.b_T)

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

    def _contract_one_operator_list(self, latt_info, gauge, eta, xi, phases, W_index_list, operator_kind, staple_links=None):
        xp = _get_xp_from_array(xi.data)
        phases = _asarray_on_queue(phases, xp, xi.data)
        gamma_ls = gamma_stack(xi.data)

        loops = []
        shifted_eta = eta.copy()
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract disconnected {operator_kind} {iW + 1}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list[iW - 1]

            if operator_kind in {"CG_qTMD", "CG_PDF"}:
                if operator_kind == "CG_PDF" and W_index[1] in {0, -1}:
                    shifted_eta = eta.copy()
                    W_index_previous = [0, 0, 0, 0]
                if operator_kind == "CG_qTMD" and W_index[3] != W_index_previous[3]:
                    shifted_eta = eta.copy()
                    W_index_previous = [0, 0, 0, W_index[3]]
                shifted_eta = self.create_fermion_TMD_CG(shifted_eta, W_index, W_index_previous)
            elif operator_kind == "GI_PDF":
                if W_index[1] in {0, -1}:
                    shifted_eta = eta.copy()
                    W_index_previous = [0, 0, 0, 0]
                shifted_eta = self.create_fermion_PDF_GI(gauge, shifted_eta, W_index, W_index_previous)
            elif operator_kind == "GI_qTMD":
                if staple_links is None:
                    raise ValueError("GI_qTMD production requires the staple-link cache")
                shifted_eta = create_fermion_TMD_GI_from_link(staple_links[tuple(W_index)], eta, W_index)
            else:
                raise ValueError(f"Unsupported operator_kind {operator_kind!r}")

            local_loop = _contract_xi_dagger_gamma_shifted_eta(
                xi.data, gamma_ls, shifted_eta.data
            )
            loop = xp.einsum("qwtzyx,gwtzyx->gqt", phases, local_loop, optimize=True)
            loops.append(core.gatherLattice(array_to_numpy(loop), [2, -1, -1, -1]))
            del local_loop, loop

        return np.asarray(loops)

    def _measure_base_shards(
        self, U, dirac, randPara, tag, operator_kind, phases_q, W_index_list,
        staple_links, attrs, shard_dir, base_start, base_stop,
        block_interval_solves, sample_log_file, completed_bases,
    ):
        n_vec, n_zn, _ = randPara
        hp_count = hp_vectors_per_base(self.noise_scheme, self.hp_num_vectors)
        shard_dir = Path(shard_dir) if shard_dir else Path(tag).parent / "shards"
        common_attrs = {
            key: value for key, value in attrs.items()
            if key not in {"n_vec", "n_base_noise", "effective_n_inversions"}
        }
        common_attrs["output_kind"] = "disconnected_qTMD_1pt"
        common_attrs["block_interval_solves"] = int(block_interval_solves)
        metadata = {
            "gamma_list": np.asarray(my_gammas, dtype="S"),
            "momentum_list": np.asarray(attrs["qext"], dtype=np.int32),
            "W_index_list": np.asarray(W_index_list, dtype=np.int32),
        }
        loop_shape = (
            len(W_index_list), len(my_gammas), len(attrs["qext"]),
            U.latt_info.global_size[3],
        )
        comm = getMPIComm()
        for base_idx in selected_base_range(n_vec, base_start, base_stop):
            if base_idx in completed_bases:
                mpi_print(U.latt_info, f"qTMD base SKIP from sample log: base{base_idx:06d}")
                continue
            for part_idx, hp_start, hp_stop in base_part_ranges(hp_count, block_interval_solves):
                count = hp_stop - hp_start
                path = shard_part_path(shard_dir, tag, base_idx, part_idx, hp_start, hp_stop)
                write_attrs = shard_part_attrs(
                    common_attrs, base_idx, part_idx, hp_start, hp_stop, hp_count
                )
                bookkeeping = part_source_bookkeeping(
                    base_idx, hp_start, hp_stop, hp_count
                )
                loop_part = np.zeros((count,) + loop_shape, dtype=np.complex128)
                for _, _, hp_idx, xi in iter_noise_base_hp_interval(
                    U.latt_info, base_idx, hp_start, hp_stop, n_zn,
                    self.noise_scheme, self.hp_num_vectors, self.hp_ordering,
                    config_num=int(attrs["config_num"]),
                    noise_stream=int(attrs["noise_stream"]),
                ):
                    eta = dirac.invert(xi)
                    loops = self._contract_one_operator_list(
                        U.latt_info, U, eta, xi, phases_q, W_index_list,
                        operator_kind, staple_links=staple_links,
                    )
                    loop_part[hp_idx - hp_start] = comm.bcast(loops, root=0)
                if U.latt_info.mpi_rank == 0:
                    write_raw_part_hdf5(
                        path, {"loop_pervec": loop_part}, write_attrs,
                        bookkeeping, metadata_datasets=metadata,
                    )
                comm.Barrier()

            if U.latt_info.mpi_rank == 0:
                append_completed_base(
                    sample_log_file, tag, common_attrs, base_idx, metadata
                )
            comm.Barrier()
        return None

    def measure_1pt(
        self,
        gauge: LatticeGauge,
        invPara,
        randPara,
        tag: str,
        operator_kind: str = "GI_PDF",
        shard_dir=None,
        base_start=0,
        base_stop=None,
        block_interval_solves=64,
        sample_log_file=None,
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
        counter_config, counter_stream = int(self.config_num), int(randseed)

        qlist = self.qlist if operator_kind in {"CG_qTMD", "GI_qTMD"} else self.qlist_PDF
        if operator_kind == "CG_qTMD":
            W_index_list_dir0, W_index_list_dir1 = self.create_TMD_Wilsonline_index_list_CG()
            W_index_list = W_index_list_dir0 + W_index_list_dir1
        elif operator_kind == "GI_qTMD":
            W_index_list_dir0, W_index_list_dir1 = self.create_TMD_Wilsonline_index_list_GI()
            W_index_list = W_index_list_dir0 + W_index_list_dir1
        else:
            W_index_list = self.create_PDF_Wilsonline_index_list()
        if len(W_index_list) == 0:
            raise ValueError(f"No Wilson-line indices were generated for operator_kind {operator_kind!r}")
        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors)
        attrs = {
            "measurement": "disconnected_qTMD_1pt",
            "operator_kind": operator_kind,
            "qext": np.asarray(qlist, dtype=np.int32),
            "W_index_list": np.asarray(W_index_list, dtype=np.int32),
            "gamma_list": np.asarray(my_gammas, dtype="S"),
            "volume_norm": Ns3,
            "mass": mass,
            "csw": csw,
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "n_vec": n_vec,
            "n_base_noise": n_vec,
            "effective_n_inversions": n_eff,
            "n_zn": n_zn,
            "config_num": counter_config,
            "noise_stream": counter_stream,
            "noise_generator": COUNTER_NOISE_ALGORITHM,
            "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
            "noise_scheme": self.noise_scheme,
            "hp_num_vectors": self.hp_num_vectors,
            "hp_ordering": self.hp_ordering,
            "gi_qtmd_staple_mode": "link_cache",
            "schema_version": QTMD_SCHEMA_VERSION,
            "loop_convention": QTMD_LOOP_CONVENTION,
            "trace_target": QTMD_TRACE_TARGET,
        }
        if sample_log_file is None:
            raise ValueError("sample_log_file is required for base-level resume")
        common_attrs = {
            key: value for key, value in attrs.items()
            if key not in {"n_vec", "n_base_noise", "effective_n_inversions"}
        }
        common_attrs["output_kind"] = "disconnected_qTMD_1pt"
        common_attrs["block_interval_solves"] = int(block_interval_solves)
        metadata = {
            "gamma_list": np.asarray(my_gammas, dtype="S"),
            "momentum_list": np.asarray(attrs["qext"], dtype=np.int32),
            "W_index_list": np.asarray(W_index_list, dtype=np.int32),
        }
        comm = getMPIComm()
        if latt_info.mpi_rank == 0:
            completed_bases = prepare_sample_log(
                sample_log_file, tag, common_attrs, metadata
            )
        else:
            completed_bases = None
        completed_bases = set(comm.bcast(completed_bases, root=0))
        selected_bases = list(selected_base_range(n_vec, base_start, base_stop))
        if all(base_idx in completed_bases for base_idx in selected_bases):
            mpi_print(latt_info, "All selected qTMD bases are complete in the sample log.")
            return None

        xi_0 = 1.0
        multigrid = [[
            max(1, global_size[0] // 1),
            max(1, global_size[1] // 1),
            max(1, global_size[2] // 2),
            max(1, global_size[3] // 8),
        ]]
        dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw, csw, multigrid)
        dirac.loadGauge(U)
        mpi_print(latt_info, "Disconnected qTMD inverter ready.")
        qext_xyz = [[q[0], q[1], q[2]] for q in qlist]
        phases_q = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
        staple_links = None
        if operator_kind == "GI_qTMD":
            mpi_print(latt_info, f"Build {len(W_index_list)} GI_qTMD staple transporters.")
            staple_links = build_gi_qtmd_staple_links(U, W_index_list)
        return self._measure_base_shards(
            U, dirac, randPara, tag, operator_kind, phases_q, W_index_list,
            staple_links, attrs, shard_dir, base_start, base_stop,
            block_interval_solves, sample_log_file, completed_bases,
        )


def finalize_disconnected_qtmd_1pt_shards(shard_dir, canonical_tag, n_base_noise):
    """Validate qTMD base shards and atomically build the canonical loop file."""
    n_base_noise = int(n_base_noise)
    manifest = discover_shard_layout(
        shard_dir, canonical_tag, n_base_noise,
        raw_dataset_names=("loop_pervec",),
        metadata_dataset_names=("gamma_list", "momentum_list", "W_index_list"),
    )
    reference_attrs = manifest["reference_attrs"]
    if int(reference_attrs.get("schema_version", -1)) != QTMD_SCHEMA_VERSION:
        raise ValueError(
            f"qTMD shards require schema_version={QTMD_SCHEMA_VERSION}; "
            "old disconnected qTMD data must be discarded and regenerated"
        )
    if reference_attrs.get("loop_convention") != QTMD_LOOP_CONVENTION:
        raise ValueError(
            f"qTMD shards require loop_convention={QTMD_LOOP_CONVENTION}; "
            "old disconnected qTMD data must be discarded and regenerated"
        )
    if reference_attrs.get("trace_target") != QTMD_TRACE_TARGET:
        raise ValueError(
            f"qTMD shards require trace_target={QTMD_TRACE_TARGET}; "
            "old disconnected qTMD data must be discarded and regenerated"
        )
    reference_metadata = manifest["metadata"]
    loop_shape = manifest["raw_tails"]["loop_pervec"]
    all_parts = manifest["parts"]
    total_sources = manifest["total_sources"]
    canonical_attrs = {
        key: value for key, value in reference_attrs.items()
        if key not in {"shard_schema", "output_kind", "block_interval_solves", "hp_vectors_per_base"}
    }
    canonical_attrs.update({
        "measurement": "disconnected_qTMD_1pt",
        "n_vec": n_base_noise,
        "n_base_noise": n_base_noise,
        "effective_n_inversions": total_sources,
    })
    final_path, temp_path = canonical_temp_path(canonical_tag)
    with h5py.File(temp_path, "w") as out:
        for key, value in canonical_attrs.items():
            out.attrs[key] = value
        for name, values in reference_metadata.items():
            out.create_dataset(name, data=values)
        raw = out.require_group("raw")
        raw_loop = raw.create_dataset("loop_pervec", shape=(total_sources,) + loop_shape, dtype=np.complex128)
        bookkeeping_datasets = {
            name: raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in ("base_noise_index", "hp_index")
        }
        loop_sum = np.zeros(loop_shape, dtype=np.complex128)
        for part_info, part in iter_validated_shard_parts(manifest):
            start = part_info["output_start"]
            stop = part_info["output_stop"]
            values = part["raw/loop_pervec"][()]
            raw_loop[start:stop] = values
            loop_sum += np.sum(values, axis=0)
            for name, dataset in bookkeeping_datasets.items():
                dataset[start:stop] = part[f"raw/{name}"][()]

        loop_avg = loop_sum / total_sources / int(canonical_attrs["volume_norm"])
        gamma_names = [value.decode() if isinstance(value, bytes) else str(value) for value in reference_metadata["gamma_list"]]
        momenta = reference_metadata["momentum_list"]
        w_indices = reference_metadata["W_index_list"]
        sm = out.require_group("avg").require_group("SS")
        for ig, gamma_name in enumerate(gamma_names):
            g_gamma = sm.require_group(gamma_name)
            for ip, momentum in enumerate(momenta):
                p_tag = "PX" + str(momentum[0]) + "PY" + str(momentum[1]) + "PZ" + str(momentum[2])
                g_p = g_gamma.require_group(p_tag)
                for iw, index in enumerate(w_indices):
                    path = "b_X" if int(index[3]) == 0 else "b_Y"
                    g_data = g_p.require_group(path + "/eta" + str(index[2]) + "/bT" + str(index[0]))
                    g_data.create_dataset("bz" + str(index[1]), data=loop_avg[iw, ig, ip])
        out.flush()
    os.replace(temp_path, final_path)
    return str(final_path)
