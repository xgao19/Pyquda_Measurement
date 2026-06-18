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
    source_bookkeeping_arrays,
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
        self.nc = int(parameters.get("Nc", 3))
        self.multigrid = parameters.get("multigrid", [[8, 8, 4, 4]])
        self.gauge_preprocessing = parameters.get("gauge_preprocessing", "unspecified")
        self.flavor_convention = parameters.get("flavor_convention", "single_flavor_trace_for_this_dirac_operator")
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)

    @staticmethod
    def _project_zero_momentum_timeslice(latt_info, local_field, q0_phase):
        from pyquda import getMPIComm
        from pyquda_utils import core

        slice_t = core.gatherLattice(
            array_to_numpy(contract("qwtzyx,wtzyx->qt", q0_phase, local_field)),
            [1, -1, -1, -1],
        )
        slice_t = getMPIComm().bcast(slice_t, root=0)
        return np.asarray(slice_t[0], dtype=np.complex128)

    def _kinetic_timeslice_for_source(self, U_f, xi, eta, q0_phase, spatial_volume):
        U_f.gauge_dirac.loadGauge(U_f)
        gammas = _gamma_stack_on_backend(eta.data)
        local_kinetic = None
        for mu in range(4):
            tmp = U_f.pure_gauge.covDev(eta, mu) - U_f.pure_gauge.covDev(eta, mu + 4)
            gamma_tmp = contract("ab,...bc->...ac", gammas[mu], tmp.data)
            term = contract("...sc,...sc->...", xi.data.conj(), gamma_tmp)
            local_kinetic = term if local_kinetic is None else local_kinetic + term
            del tmp, gamma_tmp, term

        timeslice = self._project_zero_momentum_timeslice(U_f.latt_info, local_kinetic, q0_phase)
        return timeslice / spatial_volume

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

        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors)
        kinetic_pervec = np.zeros((n_eff, n_flow, nt), dtype=np.complex128)
        source_bookkeeping = source_bookkeeping_arrays(n_eff)

        rng_probe = None
        try:
            from pyquda.field import LatticeFermion

            rng_probe = LatticeFermion(latt_info)
            xp = _get_xp_from_array(rng_probe.data)
            xp.random.seed(randseed)
        finally:
            del rng_probe

        q0_phase = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0]], [0, 0, 0, 0])
        for vec_picked, base_idx, hp_idx, xi in iter_noise_sources(
            latt_info,
            n_vec,
            n_zn,
            self.noise_scheme,
            self.hp_num_vectors,
            self.hp_ordering,
        ):
            mpi_print(latt_info, f"ringed norm vec {vec_picked} base {base_idx} hp {hp_idx}")
            source_bookkeeping["base_noise_index"][vec_picked] = base_idx
            source_bookkeeping["hp_index"][vec_picked] = hp_idx
            dirac.loadGauge(U)
            eta = dirac.invert(xi)

            U_f = U.copy()
            U_f.setAntiPeriodicT()
            for step in range(n_flow):
                kinetic_pervec[vec_picked, step] = self._kinetic_timeslice_for_source(
                    U_f,
                    xi,
                    eta,
                    q0_phase,
                    spatial_volume,
                )
                xi, eta = self._advance_flowed_pair(U_f, xi, eta, step)

            del U_f, xi, eta

        kinetic_timeslice = np.mean(kinetic_pervec, axis=0)
        kinetic_spacetime = np.mean(kinetic_timeslice, axis=-1)
        z_field_sqrt, z_bilinear = compute_ringed_factors(kinetic_spacetime, flow_time_values, nc=self.nc)

        attrs = {
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
            "effective_n_inversions": n_eff,
            "volume_norm": spatial_volume,
            "volume_average": "spacetime_average_from_spatial_timeslice_average",
            "flow0_factor": np.nan,
            "derivative_convention": "gamma_mu*(Dplus_mu-Dminus_mu)",
            "field_factor_dataset": "avg/Z_ring_field_sqrt",
            "bilinear_factor_dataset": "avg/Z_ring_bilinear",
        }
        if tag is not None and latt_info.mpi_rank == 0:
            save_flowed_quark_ringed_norm_hdf5(
                tag,
                kinetic_pervec,
                kinetic_timeslice,
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
    "compute_ringed_factors",
    "flow_times",
    "normalize_flow_type",
]
