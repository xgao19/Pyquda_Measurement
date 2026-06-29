"""Standalone flowed-quark ringed-field normalization.

This module computes the kinetic expectation value used to normalize ringed
flowed quark fields.  It is intentionally independent of EMT contractions:
the resulting factors can be consumed by any flowed-quark operator measured with
the same Dirac operator, gauge preprocessing, and flow schedule.
"""

import numpy as np
from opt_einsum import contract

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    array_to_numpy,
    effective_n_inversions,
    iter_noise_sources,
    normalize_noise_scheme,
    normalize_spin_color_dilution,
    source_bookkeeping_arrays,
    spin_color_dilution_factor,
    validate_hierarchical_probing_options,
)
from pyquda_measurement_utils.io_corr import save_flowed_quark_ringed_norm_hdf5
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print

_VALID_FLOW_TYPES = {"wilson", "symanzik"}
_D_GAMMA_IDS = [1, 2, 4, 8]


def normalize_flow_type(flow_type: str) -> str:
    """Normalize and validate the gauge/fermion flow type."""
    flow = str(flow_type).strip().lower()
    if flow not in _VALID_FLOW_TYPES:
        raise ValueError(f"flow_type should be one of {_VALID_FLOW_TYPES}, got {flow_type!r}")
    return flow


def flow_times(flow_epsilon, flow_steps):
    """Return measure-before-flow output times in lattice units."""
    return np.arange(int(flow_steps) + 1, dtype=np.float64) * float(flow_epsilon)


def compute_ringed_factors(kinetic_spacetime, flow_time_values, nc=3):
    """Compute field and bilinear ringed factors from the kinetic expectation.

    ``kinetic_spacetime`` is the single-flavor expectation value
    ``<bar chi overleftrightarrow{Dslash} chi>``.  The bilinear factor is
    ``-2*Nc / ((4*pi)^2*t^2*K)``.  The unflowed ``t=0`` entry is returned as NaN.
    """
    kinetic = np.asarray(kinetic_spacetime, dtype=np.complex128)
    times = np.asarray(flow_time_values, dtype=np.float64)
    if kinetic.shape != times.shape:
        raise ValueError(f"kinetic and flow_times should have the same shape, got {kinetic.shape} and {times.shape}")

    z_bilinear = np.full(kinetic.shape, np.nan + 0j, dtype=np.complex128)
    positive_flow = times > 0
    z_bilinear[positive_flow] = -2.0 * float(nc) / (((4.0 * np.pi) ** 2) * times[positive_flow] ** 2 * kinetic[positive_flow])
    z_field_sqrt = np.sqrt(z_bilinear)
    return z_field_sqrt, z_bilinear


def kinetic_spacetime_from_raw(kinetic_pervec, spin_color_trace_factor=1):
    """Return the spin-color traced spacetime average from raw per-source data."""
    return float(spin_color_trace_factor) * np.mean(kinetic_pervec, axis=(0, -1))


def natural_estimator_block_size(noise_scheme, hp_num_vectors, spin_color_dilution="none"):
    """Return the smallest complete estimator unit in solves."""
    scheme = normalize_noise_scheme(noise_scheme)
    hp_factor = int(hp_num_vectors) if scheme == "hierarchical_probing" else 1
    return hp_factor * spin_color_dilution_factor(spin_color_dilution)


def complete_block_size_ge_min(natural_block_size, min_solves=256):
    """Return the smallest multiple of natural_block_size no smaller than min_solves."""
    natural = int(natural_block_size)
    minimum = int(min_solves)
    if natural <= 0:
        raise ValueError(f"natural_block_size should be positive, got {natural_block_size}")
    if minimum <= 0:
        raise ValueError(f"min_solves should be positive, got {min_solves}")
    return ((minimum + natural - 1) // natural) * natural


def flowed_quark_ringed_norm_block_tag(tag, block_index, block_start, block_stop_exclusive):
    """Return the output tag for a complete block HDF5 file."""
    return (
        f"{tag}.block{int(block_index):04d}"
        f".src{int(block_start):06d}-{int(block_stop_exclusive) - 1:06d}"
    )


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def _gamma_stack_on_backend(ref_arr):
    from pyquda_utils import gamma

    return _get_xp_from_array(ref_arr).stack([
        _array_on_backend(_gamma_matrix(gamma.gamma(gamma_id)), ref_arr)
        for gamma_id in _D_GAMMA_IDS
    ])


class FlowedQuarkRingedNorm:
    """Compute ringed-field normalization for flowed quark fields."""

    def __init__(self, parameters):
        self.flow_type = normalize_flow_type(parameters["flow_type"])
        self.flow_epsilon = float(parameters["flow_epsilon"])
        self.flow_steps = int(parameters["flow_steps"])
        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "global_xyzt_gray_projected_to_evenodd")
        self.spin_color_dilution = normalize_spin_color_dilution(parameters.get("spin_color_dilution", "none"))
        self.spin_color_dilution_factor = spin_color_dilution_factor(self.spin_color_dilution)
        self.nc = int(parameters.get("Nc", 3))
        self.multigrid = parameters.get("multigrid", [[8, 8, 4, 4]])
        self.gauge_preprocessing = parameters.get("gauge_preprocessing", "unspecified")
        self.flavor_convention = parameters.get("flavor_convention", "single_flavor_trace_for_this_dirac_operator")
        self.block_write = _as_bool(parameters.get("block_write", False))
        self.block_min_solves = int(parameters.get("block_min_solves", 256))
        self.save_full = _as_bool(parameters.get("save_full", True))
        self.natural_block_size = natural_estimator_block_size(
            self.noise_scheme,
            self.hp_num_vectors,
            self.spin_color_dilution,
        )
        self.block_size = complete_block_size_ge_min(self.natural_block_size, self.block_min_solves)
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)

    def _metadata_attrs(self, latt_info, invPara, randPara, n_eff, spatial_volume):
        n_vec, n_zn, randseed = randPara
        mass, csw, tol, maxiter = invPara
        flow_time_values = flow_times(self.flow_epsilon, self.flow_steps)
        return {
            "measurement": "flowed_quark_ringed_norm",
            "normalization_scope": "all_flowed_quark_fields",
            "operator": "bar_chi_overleftrightarrow_Dslash_chi",
            "Nc": self.nc,
            "flavor_convention": self.flavor_convention,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": flow_time_values,
            "mass": mass,
            "csw": csw,
            "tol": tol,
            "maxiter": maxiter,
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "noise_scheme": self.noise_scheme,
            "n_vec": n_vec,
            "n_zn": n_zn,
            "rand_seed": randseed,
            "hp_num_vectors": self.hp_num_vectors,
            "hp_ordering": self.hp_ordering,
            "spin_color_dilution": self.spin_color_dilution,
            "spin_color_dilution_factor": self.spin_color_dilution_factor,
            "spin_color_trace_factor": self.spin_color_dilution_factor,
            "site_noise_scope": "site_spin_color" if self.spin_color_dilution == "none" else "site_only",
            "effective_n_inversions": n_eff,
            "effective_n_inversions_total": n_eff,
            "volume_norm": spatial_volume,
            "volume_average": "spin_color_trace_factor_times_spacetime_average_from_raw_kinetic_pervec",
            "flow0_factor": np.nan,
            "derivative_convention": "gamma_mu*(Dplus_mu-Dminus_mu)",
            "field_factor_dataset": "avg/Z_ring_field_sqrt",
            "bilinear_factor_dataset": "avg/Z_ring_bilinear",
            "block_output": False,
            "block_complete_policy": "smallest_complete_estimator_block_ge_min_solves",
            "natural_block_size": self.natural_block_size,
            "block_min_solves": self.block_min_solves,
            "configured_block_size": self.block_size,
            "monolithic_full_output": self.save_full,
        }

    def _write_block_file(
        self,
        tag,
        kinetic_pervec,
        flow_time_values,
        base_attrs,
        source_bookkeeping,
        block_index,
        block_start,
        block_stop,
    ):
        block_raw = kinetic_pervec[block_start:block_stop]
        block_kinetic = kinetic_spacetime_from_raw(block_raw, self.spin_color_dilution_factor)
        block_z_field, block_z_bilinear = compute_ringed_factors(block_kinetic, flow_time_values, nc=self.nc)
        block_attrs = dict(base_attrs)
        block_attrs.update(
            {
                "block_output": True,
                "block_index": int(block_index),
                "block_start": int(block_start),
                "block_stop_exclusive": int(block_stop),
                "block_size": int(block_stop - block_start),
                "block_source_count": int(block_stop - block_start),
                "monolithic_full_output": self.save_full,
            }
        )
        block_bookkeeping = {
            name: np.asarray(values[block_start:block_stop], dtype=np.int32)
            for name, values in source_bookkeeping.items()
        }
        save_flowed_quark_ringed_norm_hdf5(
            flowed_quark_ringed_norm_block_tag(tag, block_index, block_start, block_stop),
            block_raw,
            block_kinetic,
            block_z_field,
            block_z_bilinear,
            flow_time_values,
            attrs=block_attrs,
            source_bookkeeping=block_bookkeeping,
        )

    @staticmethod
    def _project_zero_momentum_per_time(latt_info, local_field, q0_phase):
        from pyquda import getMPIComm
        from pyquda_utils import core

        slice_t = core.gatherLattice(
            array_to_numpy(contract("qwtzyx,wtzyx->qt", q0_phase, local_field)),
            [1, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.asarray(slice_t[0], dtype=np.complex128)

    def _kinetic_per_time_for_source(self, U_f, xi, eta, q0_phase, spatial_volume):
        U_f.gauge_dirac.loadGauge(U_f)
        gammas = _gamma_stack_on_backend(eta.data)
        local_kinetic = None
        for mu in range(4):
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            gamma_tmp = contract("ab,...bc->...ac", gammas[mu], tmp.data)
            term = contract("...sc,...sc->...", xi.data.conj(), gamma_tmp)
            local_kinetic = term if local_kinetic is None else local_kinetic + term
            del tmp, gamma_tmp, term

        per_time = self._project_zero_momentum_per_time(U_f.latt_info, local_kinetic, q0_phase)
        return per_time / spatial_volume

    def _advance_flowed_pair(self, U_f, xi, eta, step):
        if self.flow_steps <= 0 or step >= self.flow_steps:
            return xi, eta

        from pyquda_utils import convert

        if step == 0:
            n_steps = 10
            stepsize = self.flow_epsilon / 10.0
        else:
            n_steps = 1
            stepsize = self.flow_epsilon

        packed = convert.multiField([xi, eta])
        flowed = U_f.gradientFlow(packed, self.flow_type, n_steps, stepsize)
        return flowed[0], flowed[1]

    def flowed_kinetic_norm(self, gauge, invPara, randPara, tag: str = ""):
        """Compute and save the standalone flowed-quark ringed normalization."""
        from pyquda_utils import core, phase

        n_vec, n_zn, randseed = randPara
        mass, csw, tol, maxiter = invPara
        U = gauge
        latt_info = U.latt_info
        global_size = latt_info.global_size
        spatial_volume = global_size[0] * global_size[1] * global_size[2]
        nt = global_size[3]
        n_flow = self.flow_steps + 1
        flow_time_values = flow_times(self.flow_epsilon, self.flow_steps)

        dirac = core.getDirac(
            latt_info,
            mass,
            tol,
            maxiter,
            1.0,
            csw,
            csw,
            self.multigrid,
        )
        dirac.loadGauge(U)
        mpi_print(latt_info, "Flowed-quark ringed normalization inverter ready.")

        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors, self.spin_color_dilution)
        kinetic_pervec = np.zeros((n_eff, n_flow, nt), dtype=np.complex128)
        source_bookkeeping = source_bookkeeping_arrays(n_eff, include_spin_color=True)
        attrs = self._metadata_attrs(latt_info, invPara, randPara, n_eff, spatial_volume)
        if self.block_write:
            mpi_print(
                latt_info,
                (
                    "Flowed-quark ringed normalization block output enabled: "
                    f"natural_block_size={self.natural_block_size}, block_size={self.block_size}, "
                    f"save_full={self.save_full}"
                ),
            )

        rng_probe = None
        try:
            from pyquda.field import LatticeFermion

            rng_probe = LatticeFermion(latt_info)
            xp = _get_xp_from_array(rng_probe.data)
            xp.random.seed(randseed)
        finally:
            del rng_probe

        q0_phase = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0]], [0, 0, 0, 0])
        for vec_picked, base_idx, hp_idx, spin_idx, color_idx, xi in iter_noise_sources(
            latt_info,
            n_vec,
            n_zn,
            self.noise_scheme,
            self.hp_num_vectors,
            self.hp_ordering,
            spin_color_dilution=self.spin_color_dilution,
            include_spin_color=True,
        ):
            mpi_print(latt_info, f"ringed norm vec {vec_picked} base {base_idx} hp {hp_idx} spin {spin_idx} color {color_idx}")
            source_bookkeeping["base_noise_index"][vec_picked] = base_idx
            source_bookkeeping["hp_index"][vec_picked] = hp_idx
            source_bookkeeping["spin_index"][vec_picked] = spin_idx
            source_bookkeeping["color_index"][vec_picked] = color_idx
            dirac.loadGauge(U)
            eta = dirac.invert(xi)

            U_f = U.copy()
            U_f.setAntiPeriodicT()
            for step in range(n_flow):
                kinetic_pervec[vec_picked, step] = self._kinetic_per_time_for_source(
                    U_f,
                    xi,
                    eta,
                    q0_phase,
                    spatial_volume,
                )
                xi, eta = self._advance_flowed_pair(U_f, xi, eta, step)

            del U_f, xi, eta

            if self.block_write and (vec_picked + 1) % self.block_size == 0:
                block_stop = vec_picked + 1
                block_start = block_stop - self.block_size
                block_index = block_start // self.block_size
                mpi_print(
                    latt_info,
                    f"writing ringed norm block {block_index} sources {block_start}:{block_stop}",
                )
                if tag is not None and latt_info.mpi_rank == 0:
                    self._write_block_file(
                        tag,
                        kinetic_pervec,
                        flow_time_values,
                        attrs,
                        source_bookkeeping,
                        block_index,
                        block_start,
                        block_stop,
                    )

        kinetic_spacetime = kinetic_spacetime_from_raw(kinetic_pervec, self.spin_color_dilution_factor)
        z_field_sqrt, z_bilinear = compute_ringed_factors(kinetic_spacetime, flow_time_values, nc=self.nc)

        if tag is not None and latt_info.mpi_rank == 0 and self.save_full:
            save_flowed_quark_ringed_norm_hdf5(
                tag,
                kinetic_pervec,
                kinetic_spacetime,
                z_field_sqrt,
                z_bilinear,
                flow_time_values,
                attrs=attrs,
                source_bookkeeping=source_bookkeeping,
            )
        return kinetic_spacetime, z_field_sqrt, z_bilinear


__all__ = [
    "FlowedQuarkRingedNorm",
    "complete_block_size_ge_min",
    "compute_ringed_factors",
    "flowed_quark_ringed_norm_block_tag",
    "flow_times",
    "natural_estimator_block_size",
    "normalize_flow_type",
]
