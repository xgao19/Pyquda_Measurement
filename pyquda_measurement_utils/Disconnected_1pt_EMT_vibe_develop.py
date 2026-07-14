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
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
    VECTOR_GAMMA_POSITIONS,
    basis_attrs,
    basis_metadata,
    gamma_stack,
    symmetric_vector_emt,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    COUNTER_NOISE_ALGORITHM,
    array_to_numpy,
    effective_n_inversions,
    iter_noise_base_hp_interval,
    normalize_noise_scheme,
    part_source_bookkeeping,
    validate_hierarchical_probing_options,
)

_VALID_FLOW_TYPES = {"wilson", "symanzik"}
EMT_OPERATOR_SCHEMA_VERSION = 3
my_gammas = list(GAMMA_LABELS)
pyquda_gammas_order = list(PYQUDA_GAMMA_IDS)
my_pyquda_gammas = [gamma.gamma(idx) for idx in pyquda_gammas_order]
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


def _normalize_flow_type(flow_type: str) -> str:
    flow = str(flow_type).strip().lower()
    if flow not in _VALID_FLOW_TYPES:
        raise ValueError(f"flow_type should be one of {_VALID_FLOW_TYPES}, got {flow_type!r}")
    return flow


def _flow_times(flow_epsilon, flow_steps):
    return np.arange(flow_steps + 1, dtype=np.float64) * float(flow_epsilon)


def parse_multigrid_blocks(value):
    """Parse dot-separated QUDA blocks and semicolon-separated MG levels."""
    if isinstance(value, str):
        level_text = [item.strip() for item in value.split(";") if item.strip()]
        if not level_text:
            raise ValueError("multigrid block specification is empty")
        blocks = []
        for item in level_text:
            try:
                block = [int(entry) for entry in item.split(".")]
            except ValueError as error:
                raise ValueError(
                    f"invalid multigrid block {item!r}; expected X.Y.Z.T"
                ) from error
            blocks.append(block)
    else:
        blocks = [[int(entry) for entry in block] for block in value]
    if not blocks or any(len(block) != 4 for block in blocks):
        raise ValueError("each multigrid level must contain four integers")
    if any(entry <= 0 for block in blocks for entry in block):
        raise ValueError("multigrid block entries must be positive")
    return blocks


def _unique_zero_momentum_index(momentum_list):
    """Return the unique zero-momentum index or raise a clear error."""
    zero_indices = [
        idx
        for idx, momentum in enumerate(momentum_list)
        if np.asarray(momentum).size > 0 and np.all(np.asarray(momentum) == 0)
    ]
    if len(zero_indices) != 1:
        raise ValueError(
            "EMT quark 1pt requires qext to contain exactly one zero momentum; "
            f"found {len(zero_indices)}"
        )
    return zero_indices[0]


def emt_tensor_from_derivative_bilinear(derivative_bilinear):
    """Derive the symmetric vector EMT with shape ``[...,4,4,Nq,Nflow,Nt]``."""
    return symmetric_vector_emt(derivative_bilinear, gamma_axis=1, derivative_axis=2)


def ringed_kinetic_pervec_from_derivative(
    derivative_bilinear_pervec, zero_momentum_index, spatial_volume
):
    """Extract ringed kinetic timeslices from vector derivative diagonals."""
    derivative = np.asarray(derivative_bilinear_pervec)
    if derivative.ndim != 6 or derivative.shape[1:3] != (16, 4):
        raise ValueError(
            "derivative_bilinear_pervec should have shape "
            "[N_eff,16,4,Nq,Nflow,Nt], "
            f"got {derivative.shape}"
        )
    q0_index = int(zero_momentum_index)
    if not 0 <= q0_index < derivative.shape[3]:
        raise ValueError(f"zero_momentum_index {q0_index} outside Nq={derivative.shape[3]}")
    spatial_volume = int(spatial_volume)
    if spatial_volume <= 0:
        raise ValueError(f"spatial_volume should be positive, got {spatial_volume}")

    diagonal_sum = sum(
        derivative[:, gamma_pos, mu, q0_index, :, :]
        for mu, gamma_pos in enumerate(VECTOR_GAMMA_POSITIONS)
    )
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

        self.flow_type = _normalize_flow_type(parameters["flow_type"])
        self.flow_epsilon = parameters["flow_epsilon"]
        self.flow_steps = parameters["flow_steps"]
        self.config_num = parameters.get("config_num")
        self.gauge_preprocessing = parameters.get(
            "gauge_preprocessing", "unspecified"
        )
        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get(
            "hp_ordering", "interleaved_xyzt_binary_projected_to_evenodd"
        )
        self.flavor_convention = parameters.get(
            "flavor_convention",
            "single_flavor_trace_for_this_dirac_operator",
        )
        self.multigrid_blocks = parse_multigrid_blocks(
            parameters.get("multigrid", [[8, 8, 4, 4]])
        )
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)

    @staticmethod
    def _gamma5_for(ref_arr):
        return _array_on_backend(_gamma_matrix(G5), ref_arr)

    @staticmethod
    def _gamma_stack_for(ref_arr):
        return gamma_stack(ref_arr)

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

    @staticmethod
    def _project_gamma_fields(gamma_fields, phases_3pt):
        """Project ``[gamma,site]`` scalar fields and preserve absolute time."""
        projected = contract("qwtzyx,gwtzyx->gqt", phases_3pt, gamma_fields)
        slice_t = core.gatherLattice(array_to_numpy(projected), [2, -1, -1, -1])
        return getMPIComm().bcast(slice_t, root=0)

    def _get_primitive_bilinears_P_Breit_slice(
        self,
        U_f: LatticeGauge,
        xi: LatticeFermion,
        eta: LatticeFermion,
        phases_3pt,
    ):
        """Build all local/one-derivative bilinears and the flowed-noise norm."""
        Nt = U_f.latt_info.global_size[3]
        Nq = len(phases_3pt)

        gamma_ls = self._gamma_stack_for(eta.data)
        local_fields = contract(
            "wtzyxia,gij,wtzyxja->gwtzyx",
            xi.data.conj(), gamma_ls, eta.data,
        )
        local = np.asarray(self._project_gamma_fields(local_fields, phases_3pt))
        del local_fields

        dot_xi_xi = contract("etzyxbc,etzyxbc->etzyx", xi.data.conj(), xi.data)
        flowed_noise_norm = np.asarray(
            self._impose_P_Breit_slice(dot_xi_xi, phases_3pt)
        )
        del dot_xi_xi

        derivative = np.zeros([16, 4, Nq, Nt], dtype=np.complex128)
        U_f.gauge_dirac.loadGauge(U_f)
        for mu in range(4):
            derivative_right = U_f.pure_gauge.covDev(eta, mu)
            derivative_left = U_f.pure_gauge.covDev(eta, mu + 4)
            tmp = derivative_right - derivative_left
            derivative_fields = contract(
                "wtzyxia,gij,wtzyxja->gwtzyx",
                xi.data.conj(), gamma_ls, tmp.data,
            )
            derivative[:, mu] = -0.5 * np.asarray(
                self._project_gamma_fields(derivative_fields, phases_3pt)
            )
            del derivative_fields, tmp, derivative_right, derivative_left

        return local, derivative, flowed_noise_norm

    def _measure_flowed_source(self, U, xi, eta, phases_3pt):
        n_flow = self.flow_steps + 1
        nt = U.latt_info.global_size[3]
        local = np.zeros((16, len(self.qlist), n_flow, nt), dtype=np.complex128)
        derivative = np.zeros((16, 4, len(self.qlist), n_flow, nt), dtype=np.complex128)
        flowed_noise_norm = np.zeros(
            (len(self.qlist), n_flow, nt), dtype=np.complex128
        )
        U_f = U.copy()
        U_f.setAntiPeriodicT()
        for step in range(n_flow):
            mpi_print(U_f.latt_info, f"calc primitive bilinears, step = {step}")
            (
                local[:, :, step],
                derivative[:, :, :, step],
                flowed_noise_norm[:, step],
            ) = (
                self._get_primitive_bilinears_P_Breit_slice(
                U_f, xi, eta, phases_3pt
                )
            )
            if step < self.flow_steps:
                if step == 0:
                    n_steps, step_size = 10, self.flow_epsilon / 10
                else:
                    n_steps, step_size = 1, self.flow_epsilon
                flowed = U_f.gradientFlow(convert.multiField([xi, eta]), self.flow_type, n_steps, step_size)
                xi, eta = flowed[0], flowed[1]
        return local, derivative, flowed_noise_norm

    def _measurement_attrs(self, latt_info, invPara, randPara, counter_config, counter_stream, n_eff, spatial_volume):
        n_vec, n_zn, _ = randPara
        mass, csw, tol, maxiter = invPara
        attrs = {
            "measurement": "quark_1pt",
            "emt_operator_schema_version": EMT_OPERATOR_SCHEMA_VERSION,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "volume_norm": spatial_volume,
            "primitive_local_axes": "source,gamma,q,flow,t",
            "primitive_derivative_axes": "source,gamma,derivative,q,flow,t",
            "flowed_noise_norm_axes": "source,q,flow,t",
            "primitive_derivative_unsymmetrized": True,
            "derived_emt_axes": "source,mu,nu,q,flow,t",
            "derived_emt_upper_triangle_only": True,
            "operator_normalization": "unrenormalized_flowed_quark_bilinear",
            "renormalization_applied": False,
            "renormalization_stage": "analysis_stage",
            "flavor_convention": self.flavor_convention,
            "local_bilinear_convention": "xi_dag*Gamma_A*eta",
            "derivative_convention": "L_D[A,mu]=-0.5*xi_dag*Gamma_A*(Dplus_mu-Dminus_mu)*eta; unsymmetrized",
            "emt_derivation": "B[nu,mu]=L_D[gamma_nu,mu]; T=0.5*(B+B_transpose)",
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
            "multigrid_blocks": np.asarray(self.multigrid_blocks, dtype=np.int32),
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
        attrs.update(basis_attrs())
        return attrs

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
        operator_metadata = basis_metadata()
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
                    "local_bilinear_pervec": (count, 16, len(self.qlist), n_flow, nt),
                    "derivative_bilinear_pervec": (count, 16, 4, len(self.qlist), n_flow, nt),
                    "flowed_noise_norm_pervec": (
                        count, len(self.qlist), n_flow, nt
                    ),
                    "source_index": (count,),
                    "base_noise_index": (count,),
                    "hp_index": (count,),
                }
                bookkeeping = part_source_bookkeeping(
                    base_idx, hp_start, hp_stop, hp_count
                )
                local_part = np.zeros(raw_shapes["local_bilinear_pervec"], dtype=np.complex128)
                derivative_part = np.zeros(raw_shapes["derivative_bilinear_pervec"], dtype=np.complex128)
                norm_part = np.zeros(
                    raw_shapes["flowed_noise_norm_pervec"], dtype=np.complex128
                )
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
                    (
                        local_part[local_idx],
                        derivative_part[local_idx],
                        norm_part[local_idx],
                    ) = self._measure_flowed_source(U, xi, eta, phases_3pt)
                if U.latt_info.mpi_rank == 0:
                    write_raw_part_hdf5(
                        path,
                        {
                            "local_bilinear_pervec": local_part,
                            "derivative_bilinear_pervec": derivative_part,
                            "flowed_noise_norm_pervec": norm_part,
                        },
                        write_attrs,
                        bookkeeping,
                        metadata_datasets=operator_metadata,
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

        _unique_zero_momentum_index(self.qlist)

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
            self.multigrid_blocks,
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


def finalize_emt_quark_1pt_shards(shard_dir, canonical_tag, n_base_noise):
    """Validate complete EMT base shards and atomically build one EMTc output."""
    n_base_noise = int(n_base_noise)
    manifest = discover_shard_layout(
        shard_dir, canonical_tag, n_base_noise,
        raw_dataset_names=(
            "local_bilinear_pervec",
            "derivative_bilinear_pervec",
            "flowed_noise_norm_pervec",
        ),
        metadata_dataset_names=tuple(basis_metadata()),
    )
    reference_attrs = manifest["reference_attrs"]
    schema_version = int(reference_attrs.get("emt_operator_schema_version", -1))
    if schema_version != EMT_OPERATOR_SCHEMA_VERSION:
        raise ValueError(
            "EMT shards require emt_operator_schema_version="
            f"{EMT_OPERATOR_SCHEMA_VERSION}; found {schema_version}"
        )
    local_shape = manifest["raw_tails"]["local_bilinear_pervec"]
    derivative_shape = manifest["raw_tails"]["derivative_bilinear_pervec"]
    norm_shape = manifest["raw_tails"]["flowed_noise_norm_pervec"]
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
    if local_shape[:1] != (16,) or derivative_shape[:2] != (16, 4):
        raise ValueError(
            "EMT primitive shard axes should begin with local[16] and derivative[16,4]"
        )
    n_flow = derivative_shape[-2]
    nt = derivative_shape[-1]

    final_path, temp_path = canonical_temp_path(canonical_tag)
    with h5py.File(temp_path, "w") as out:
        _copy_h5_attrs(out, canonical_attrs)
        for name, values in manifest["metadata"].items():
            out.create_dataset(name, data=values)
        raw = out.require_group("raw")
        raw_local = raw.create_dataset(
            "local_bilinear_pervec",
            shape=(total_sources,) + local_shape,
            dtype=np.complex128,
        )
        raw_derivative = raw.create_dataset(
            "derivative_bilinear_pervec",
            shape=(total_sources,) + derivative_shape,
            dtype=np.complex128,
        )
        raw_norm = raw.create_dataset(
            "flowed_noise_norm_pervec",
            shape=(total_sources,) + norm_shape,
            dtype=np.complex128,
        )
        source_datasets = {
            name: raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in ("source_index", "base_noise_index", "hp_index")
        }
        local_sum = np.zeros(local_shape, dtype=np.complex128)
        derivative_sum = np.zeros(derivative_shape, dtype=np.complex128)
        t_sum = np.zeros((4, 4) + derivative_shape[2:], dtype=np.complex128)
        norm_sum = np.zeros(norm_shape, dtype=np.complex128)

        ringed = out.require_group("derived/ringed")
        ringed_attrs = {
            "kinetic_pervec_axes": "source,flow,t",
            "kinetic_spacetime_axes": "flow",
            "kinetic_source": "raw_derivative_bilinear_pervec_vector_diagonal_at_zero_momentum",
            "kinetic_relation": "K_pervec=-2*sum_mu(L_D[gamma_mu,mu,q0])/spatial_volume",
            "zero_momentum_index": q0_index,
            "ringed_factors_stored": False,
            "ringed_factor_stage": "ensemble_analysis_from_configuration_averaged_kinetic",
        }
        _copy_h5_attrs(ringed, ringed_attrs)
        ring_k = ringed.create_dataset(
            "kinetic_pervec",
            shape=(total_sources, n_flow, nt),
            dtype=np.complex128,
        )
        kinetic_sum = np.zeros(n_flow, dtype=np.complex128)

        for part_info, part in iter_validated_shard_parts(manifest):
            start = part_info["output_start"]
            stop = part_info["output_stop"]
            local_data = part["raw/local_bilinear_pervec"][()]
            derivative_data = part["raw/derivative_bilinear_pervec"][()]
            norm_data = part["raw/flowed_noise_norm_pervec"][()]
            raw_local[start:stop] = local_data
            raw_derivative[start:stop] = derivative_data
            raw_norm[start:stop] = norm_data
            local_sum += np.sum(local_data, axis=0)
            derivative_sum += np.sum(derivative_data, axis=0)
            t_data = emt_tensor_from_derivative_bilinear(derivative_data)
            t_sum += np.sum(t_data, axis=0)
            norm_sum += np.sum(norm_data, axis=0)
            for name, dataset in source_datasets.items():
                dataset[start:stop] = part[f"raw/{name}"][()]
            kinetic = ringed_kinetic_pervec_from_derivative(
                derivative_data, q0_index, spatial_volume
            )
            ring_k[start:stop] = kinetic
            kinetic_sum += np.sum(kinetic, axis=(0, -1))

        avg = out.require_group("avg")
        avg.create_dataset(
            "local_bilinear", data=local_sum / total_sources / spatial_volume
        )
        avg.create_dataset(
            "derivative_bilinear",
            data=derivative_sum / total_sources / spatial_volume,
        )
        avg.create_dataset(
            "flowed_noise_norm",
            data=norm_sum / total_sources / spatial_volume,
        )
        avg_t = avg.require_group("Tmunu")
        avg_t.attrs["upper_triangle_only"] = True
        t_avg = t_sum / total_sources / spatial_volume
        for mu in range(4):
            for nu in range(mu, 4):
                avg_t.create_dataset(f"T{mu+1}{nu+1}", data=t_avg[mu, nu])
        ringed.create_dataset(
            "kinetic_spacetime", data=kinetic_sum / total_sources / nt
        )
        out.flush()

    os.replace(temp_path, final_path)
    return str(final_path)


class EMTDisconnectedGluon1pt:
    """Hadron-independent flowed gluon EMT loop measurement."""

    def __init__(self, parameters):
        self.qlist = parameters["qext"]

        self.flow_type = _normalize_flow_type(parameters["flow_type"])
        self.flow_epsilon = parameters["flow_epsilon"]
        self.flow_steps = parameters["flow_steps"]
        self.config_num = parameters.get("config_num")
        self.gauge_preprocessing = parameters.get(
            "gauge_preprocessing", "unspecified"
        )

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
            "volume_norm": Ns3,
            "upper_triangle_only": True,
            "operator_normalization": "flowed_gluon_bilinear",
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "renormalization_applied": False,
            "renormalization_stage": "analysis_stage",
        }
        if latt_info.mpi_rank == 0:
            save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=attrs)

        return Tmunu_t


__all__ = [
    "EMTDisconnectedQuark1pt",
    "EMTDisconnectedGluon1pt",
    "EMT_OPERATOR_SCHEMA_VERSION",
    "parse_multigrid_blocks",
    "my_gammas",
    "validate_quark_gluon_loop_axes",
]
