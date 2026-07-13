"""Shared flowed EMT one-point loop measurements.

This module contains the hadron-independent quark and gluon 1pt pieces used by
both pion and proton EMT workflows.  These loops are the building blocks for
disconnected diagrams in analysis:

    C3_disc = < C2 L > - < C2 > < L >.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from opt_einsum import contract

from pyquda import getMPIComm
from pyquda.field import LatticeGauge, LatticePropagator, LatticeFermion, MultiLatticeFermion
from pyquda_utils import core, gamma, phase, convert
from pyquda_comm.array import arrayIdentity, arrayZeros

from pyquda_measurement_utils.io_corr import (
    save_emt_gluon_1pt_hdf5,
)
from pyquda_measurement_utils.disconnected_shards import (
    append_completed_base,
    base_part_ranges,
    canonical_temp_path,
    discover_shard_layout,
    hp_vectors_per_base,
    iter_validated_shard_parts,
    prepare_sample_log,
    selected_base_range,
    shard_part_attrs,
    shard_part_path,
    write_raw_part_hdf5,
)
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    COUNTER_NOISE_ALGORITHM,
    effective_n_inversions,
    iter_noise_base_hp_interval,
    normalize_noise_scheme,
    part_source_bookkeeping,
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


def _unique_zero_momentum_index(momentum_list):
    """Return the unique zero-momentum index or raise a clear error."""
    zero_indices = [
        idx
        for idx, momentum in enumerate(momentum_list)
        if np.asarray(momentum).size > 0 and np.all(np.asarray(momentum) == 0)
    ]
    if len(zero_indices) != 1:
        raise ValueError(
            "ringed kinetic output requires qext to contain exactly one zero momentum; "
            f"found {len(zero_indices)}"
        )
    return zero_indices[0]


def ringed_kinetic_pervec_from_emt(Tmunu_pervec, zero_momentum_index, spatial_volume):
    """Extract the ringed kinetic timeslices from raw EMT diagonal components."""
    tensor = np.asarray(Tmunu_pervec)
    if tensor.ndim != 6 or tensor.shape[1:3] != (4, 4):
        raise ValueError(
            "Tmunu_pervec should have shape [N_eff,4,4,Nq,Nflow,Nt], "
            f"got {tensor.shape}"
        )
    q0_index = int(zero_momentum_index)
    if not 0 <= q0_index < tensor.shape[3]:
        raise ValueError(f"zero_momentum_index {q0_index} outside Nq={tensor.shape[3]}")
    spatial_volume = int(spatial_volume)
    if spatial_volume <= 0:
        raise ValueError(f"spatial_volume should be positive, got {spatial_volume}")

    diagonal_sum = sum(tensor[:, mu, mu, q0_index, :, :] for mu in range(4))
    return (-2.0 / spatial_volume) * diagonal_sum


def validate_quark_gluon_loop_axes(
    quark_qext, gluon_qext, quark_flow_times, gluon_flow_times
):
    """Require matched momentum and flow axes before quark/gluon analysis."""
    if not np.array_equal(np.asarray(quark_qext), np.asarray(gluon_qext)):
        raise ValueError("Quark and gluon 1pt files must use matching qext")
    if not np.array_equal(
        np.asarray(quark_flow_times), np.asarray(gluon_flow_times)
    ):
        raise ValueError("Quark and gluon 1pt files must use matching flow_times")


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
        self.config_num = parameters.get("config_num")
        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "interleaved_xyz_binary_projected_to_evenodd")
        self.nc = int(parameters.get("Nc", 3))
        self.gauge_preprocessing = parameters.get("gauge_preprocessing", "unspecified")
        self.flavor_convention = parameters.get(
            "flavor_convention",
            "single_flavor_trace_for_this_dirac_operator",
        )
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
    def _impose_P_Breit_slice(complex_field, phases_3pt):
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
        phases_3pt,
    ):
        """Build flowed quark 1pt EMT and scalar diagnostics."""
        Nt = U_f.latt_info.global_size[3]
        Nq = len(phases_3pt)

        CHI = np.zeros([2, Nq, Nt], dtype=np.complex128)
        dot_xi_eta = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), eta.data)
        CHI[0] = self._impose_P_Breit_slice(dot_xi_eta, phases_3pt)
        dot_xi_xi = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), xi.data)
        CHI[1] = self._impose_P_Breit_slice(dot_xi_xi, phases_3pt)

        Tmunu = np.zeros([4, 4, Nq, Nt], dtype=np.complex128)
        U_f.gauge_dirac.loadGauge(U_f)
        D_gammas_local = self._dirac_gammas_for(eta.data)
        for mu in range(4):
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            for nu in range(4):
                Y = contract("ab,...bc->...ac", D_gammas_local[nu], tmp.data)
                complex_field = contract("...sc,...sc->...", xi.data.conj(), Y)
                Tmunu[nu, mu] += -0.5 * self._impose_P_Breit_slice(complex_field, phases_3pt)

        for mu in range(4):
            for nu in range(mu + 1, 4):
                Tmunu[mu, nu] = (Tmunu[mu, nu] + Tmunu[nu, mu]) / 2
                Tmunu[nu, mu] = Tmunu[mu, nu]

        return Tmunu, CHI

    def _measure_flowed_source(self, U, xi, eta, phases_3pt):
        n_flow = self.flow_steps + 1
        nt = U.latt_info.global_size[3]
        tmunu = np.zeros((4, 4, len(self.qlist), n_flow, nt), dtype=np.complex128)
        chi = np.zeros((2, len(self.qlist), n_flow, nt), dtype=np.complex128)
        U_f = U.copy()
        U_f.setAntiPeriodicT()
        for step in range(n_flow):
            mpi_print(U_f.latt_info, f"calc Tmunu, step = {step}")
            tmunu[:, :, :, step], chi[:, :, step] = self._get_Tmunu_symmetrized_P_Breit_slice(
                U_f, xi, eta, phases_3pt
            )
            if step < self.flow_steps:
                if step == 0:
                    n_steps, step_size = 10, self.flow_epsilon / 10
                else:
                    n_steps, step_size = 1, self.flow_epsilon
                flowed = U_f.gradientFlow(convert.multiField([xi, eta]), self.flow_type, n_steps, step_size)
                xi, eta = flowed[0], flowed[1]
        return tmunu, chi

    def _measurement_attrs(self, latt_info, invPara, randPara, counter_config, counter_stream, n_eff, spatial_volume):
        n_vec, n_zn, _ = randPara
        mass, csw, tol, maxiter = invPara
        return {
            "measurement": "quark_1pt",
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "pf": np.asarray(self.pf, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "volume_norm": spatial_volume,
            "upper_triangle_only": True,
            "operator_normalization": "unrenormalized_flowed_quark_bilinear",
            "renormalization_applied": False,
            "renormalization_stage": "analysis_stage",
            "flavor_convention": self.flavor_convention,
            "derivative_convention": "Tmunu[nu,mu]=-0.5*xi_dag*gamma_nu*(Dplus_mu-Dminus_mu)*eta; symmetrized",
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
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
        }

    def _measure_base_shards(
        self, U, dirac, invPara, randPara, tag, phases_3pt, attrs,
        shard_dir, sample_log_file, base_start, base_stop, block_interval_solves,
        completed_bases,
    ):
        n_vec, n_zn, _ = randPara
        hp_count = hp_vectors_per_base(self.noise_scheme, self.hp_num_vectors)
        shard_dir = Path(shard_dir) if shard_dir else Path(tag).parent / "shards"
        common_attrs = {
            key: value for key, value in attrs.items()
            if key not in {"n_vec", "n_base_noise", "effective_n_inversions"}
        }
        common_attrs["output_kind"] = "emt_quark_1pt"
        common_attrs["block_interval_solves"] = int(block_interval_solves)
        nt = U.latt_info.global_size[3]
        n_flow = self.flow_steps + 1
        comm = getMPIComm()

        for base_idx in selected_base_range(n_vec, base_start, base_stop):
            if base_idx in completed_bases:
                mpi_print(U.latt_info, f"EMT base SKIP from sample log: base{base_idx:06d}")
                continue
            for part_idx, hp_start, hp_stop in base_part_ranges(hp_count, block_interval_solves):
                count = hp_stop - hp_start
                path = shard_part_path(shard_dir, tag, base_idx, part_idx, hp_start, hp_stop)
                write_attrs = shard_part_attrs(
                    common_attrs, base_idx, part_idx, hp_start, hp_stop, hp_count
                )
                raw_shapes = {
                    "Tmunu_pervec": (count, 4, 4, len(self.qlist), n_flow, nt),
                    "CHI_pervec": (count, 2, len(self.qlist), n_flow, nt),
                    "source_index": (count,),
                    "base_noise_index": (count,),
                    "hp_index": (count,),
                }
                bookkeeping = part_source_bookkeeping(
                    base_idx, hp_start, hp_stop, hp_count
                )
                tmunu_part = np.zeros(raw_shapes["Tmunu_pervec"], dtype=np.complex128)
                chi_part = np.zeros(raw_shapes["CHI_pervec"], dtype=np.complex128)
                for _, _, hp_idx, xi in iter_noise_base_hp_interval(
                    U.latt_info, base_idx, hp_start, hp_stop, n_zn,
                    self.noise_scheme, self.hp_num_vectors, self.hp_ordering,
                    config_num=int(attrs["config_num"]),
                    noise_stream=int(attrs["noise_stream"]),
                ):
                    local_idx = hp_idx - hp_start
                    # The previous source leaves a flowed gauge resident in QUDA.
                    dirac.loadGauge(U)
                    eta = dirac.invert(xi)
                    tmunu_part[local_idx], chi_part[local_idx] = self._measure_flowed_source(
                        U, xi, eta, phases_3pt
                    )
                if U.latt_info.mpi_rank == 0:
                    write_raw_part_hdf5(
                        path,
                        {"Tmunu_pervec": tmunu_part, "CHI_pervec": chi_part},
                        write_attrs,
                        bookkeeping,
                    )
                comm.Barrier()

            if U.latt_info.mpi_rank == 0:
                append_completed_base(
                    sample_log_file, tag, common_attrs, base_idx
                )
            comm.Barrier()
        return None, None

    def flowed_fermionic_1pt(
        self,
        gauge: LatticeGauge,
        invPara,
        randPara,
        tag: str = "",
        ringed_tag=None,
        shard_dir=None,
        sample_log_file=None,
        base_start=0,
        base_stop=None,
        block_interval_solves=64,
    ):
        """Compute quark flowed 1pt observables with stochastic sources."""
        n_vec, n_zn, randseed = randPara
        mass, csw, tol, maxiter = invPara
        U = gauge
        latt_info = U.latt_info

        global_size = latt_info.global_size
        Ns3 = global_size[0] * global_size[1] * global_size[2]

        if ringed_tag is not None:
            if not str(ringed_tag).strip():
                raise ValueError("ringed_tag should be non-empty when ringed kinetic output is enabled")
            q0_index = _unique_zero_momentum_index(self.qlist)
        else:
            q0_index = None

        if self.config_num is None:
            raise ValueError("config_num is required for counter-based disconnected noise")
        counter_config = int(self.config_num)
        counter_stream = int(randseed)

        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors)
        attrs = self._measurement_attrs(latt_info, invPara, randPara, counter_config, counter_stream, n_eff, Ns3)
        if sample_log_file is None:
            raise ValueError("sample_log_file is required for base-level resume")
        common_attrs = {
            key: value for key, value in attrs.items()
            if key not in {"n_vec", "n_base_noise", "effective_n_inversions"}
        }
        common_attrs["output_kind"] = "emt_quark_1pt"
        common_attrs["block_interval_solves"] = int(block_interval_solves)
        comm = getMPIComm()
        if latt_info.mpi_rank == 0:
            completed_bases = prepare_sample_log(sample_log_file, tag, common_attrs)
        else:
            completed_bases = None
        completed_bases = set(comm.bcast(completed_bases, root=0))
        selected_bases = list(selected_base_range(n_vec, base_start, base_stop))
        if all(base_idx in completed_bases for base_idx in selected_bases):
            mpi_print(latt_info, "All selected EMT bases are complete in the sample log.")
            return None, None

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

        qext_xyz = [[q[0], q[1], q[2]] for q in self.qlist]
        phases_3pt = phase.MomentumPhase(latt_info).getPhases(qext_xyz, [0, 0, 0, 0])
        return self._measure_base_shards(
            U, dirac, invPara, randPara, tag, phases_3pt, attrs,
            shard_dir, sample_log_file, base_start, base_stop, block_interval_solves,
            completed_bases,
        )

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


def _copy_h5_attrs(obj, attrs):
    for key, value in attrs.items():
        if value is not None:
            obj.attrs[key] = value


def finalize_emt_quark_1pt_shards(shard_dir, canonical_tag, ringed_tag, n_base_noise):
    """Validate complete EMT base shards and atomically build canonical outputs."""
    n_base_noise = int(n_base_noise)
    manifest = discover_shard_layout(
        shard_dir, canonical_tag, n_base_noise,
        raw_dataset_names=("Tmunu_pervec", "CHI_pervec"),
    )
    reference_attrs = manifest["reference_attrs"]
    source_shape = manifest["raw_tails"]["Tmunu_pervec"]
    chi_shape = manifest["raw_tails"]["CHI_pervec"]
    all_parts = manifest["parts"]
    total_sources = manifest["total_sources"]
    canonical_attrs = {
        key: value for key, value in reference_attrs.items()
        if key not in {"shard_schema", "output_kind", "block_interval_solves", "hp_vectors_per_base"}
    }
    canonical_attrs.update({
        "measurement": "quark_1pt",
        "n_vec": n_base_noise,
        "n_base_noise": n_base_noise,
        "effective_n_inversions": total_sources,
    })
    q0_index = _unique_zero_momentum_index(canonical_attrs["qext"])
    spatial_volume = int(canonical_attrs["volume_norm"])
    n_flow = source_shape[-2]
    nt = source_shape[-1]

    final_path, temp_path = canonical_temp_path(canonical_tag)
    ringed_path, ringed_temp = canonical_temp_path(ringed_tag)
    with h5py.File(temp_path, "w") as out, h5py.File(ringed_temp, "w") as ringed:
        _copy_h5_attrs(out, canonical_attrs)
        raw = out.require_group("raw")
        raw_t = raw.create_dataset("Tmunu_pervec", shape=(total_sources,) + source_shape, dtype=np.complex128)
        raw_chi = raw.create_dataset("CHI_pervec", shape=(total_sources,) + chi_shape, dtype=np.complex128)
        source_datasets = {
            name: raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in ("source_index", "base_noise_index", "hp_index")
        }
        t_sum = np.zeros(source_shape, dtype=np.complex128)
        chi_sum = np.zeros(chi_shape, dtype=np.complex128)

        ringed_attrs = dict(canonical_attrs)
        ringed_attrs.update({
            "measurement": "flowed_quark_ringed_norm",
            "producer": "emt_quark_1pt",
            "content": "kinetic_only",
            "normalization_scope": "all_flowed_quark_fields",
            "operator": "bar_chi_overleftrightarrow_Dslash_chi",
            "Nc": 3,
            "spin_color_dilution": "none",
            "spin_color_dilution_factor": 1,
            "spin_color_trace_factor": 1,
            "site_noise_scope": "site_spin_color",
            "kinetic_source": "raw_Tmunu_pervec_zero_momentum_diagonal_trace",
            "kinetic_relation": "K_pervec=-2*sum_mu(Tmunu_pervec[mu,mu,q0])/spatial_volume",
            "zero_momentum_index": q0_index,
            "ringed_factors_stored": False,
            "ringed_factor_stage": "ensemble_analysis_from_configuration_averaged_kinetic",
        })
        _copy_h5_attrs(ringed, ringed_attrs)
        ringed.create_dataset("flow_times", data=np.asarray(canonical_attrs["flow_times"], dtype=np.float64))
        ring_raw = ringed.require_group("raw")
        ring_k = ring_raw.create_dataset("kinetic_pervec", shape=(total_sources, n_flow, nt), dtype=np.complex128)
        ring_sources = {
            name: ring_raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in ("source_index", "base_noise_index", "hp_index", "spin_index", "color_index")
        }
        kinetic_sum = np.zeros(n_flow, dtype=np.complex128)

        for part_info, part in iter_validated_shard_parts(manifest):
            start = part_info["output_start"]
            stop = part_info["output_stop"]
            t_data = part["raw/Tmunu_pervec"][()]
            chi_data = part["raw/CHI_pervec"][()]
            raw_t[start:stop] = t_data
            raw_chi[start:stop] = chi_data
            t_sum += np.sum(t_data, axis=0)
            chi_sum += np.sum(chi_data, axis=0)
            for name, dataset in source_datasets.items():
                dataset[start:stop] = part[f"raw/{name}"][()]
            kinetic = ringed_kinetic_pervec_from_emt(t_data, q0_index, spatial_volume)
            ring_k[start:stop] = kinetic
            kinetic_sum += np.sum(kinetic, axis=(0, -1))
            for name in ("source_index", "base_noise_index", "hp_index"):
                ring_sources[name][start:stop] = part[f"raw/{name}"][()]
            ring_sources["spin_index"][start:stop] = -1
            ring_sources["color_index"][start:stop] = -1

        avg = out.require_group("avg")
        avg.create_dataset("CHI", data=chi_sum / total_sources / spatial_volume)
        avg_t = avg.require_group("Tmunu")
        avg_t.attrs["upper_triangle_only"] = True
        t_avg = t_sum / total_sources / spatial_volume
        for mu in range(4):
            for nu in range(mu, 4):
                avg_t.create_dataset(f"T{mu+1}{nu+1}", data=t_avg[mu, nu])
        ringed.require_group("avg").create_dataset(
            "kinetic_spacetime", data=kinetic_sum / total_sources / nt
        )
        out.flush()
        ringed.flush()

    os.replace(ringed_temp, ringed_path)
    os.replace(temp_path, final_path)
    return str(final_path), str(ringed_path)


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
        self.config_num = parameters.get("config_num")

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
            "config_num": self.config_num,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "pf": np.asarray(self.pf, dtype=np.int32),
            "p_2pt": np.asarray(self.pilist, dtype=np.int32),
            "volume_norm": Ns3,
            "upper_triangle_only": True,
            "operator_normalization": "flowed_gluon_bilinear",
            "renormalization_applied": False,
            "renormalization_stage": "analysis_stage",
        }
        if latt_info.mpi_rank == 0:
            save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=attrs)

        return Tmunu_t


__all__ = [
    "EMTDisconnectedQuark1pt",
    "EMTDisconnectedGluon1pt",
    "my_gammas",
    "validate_quark_gluon_loop_axes",
]
