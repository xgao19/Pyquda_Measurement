"""Standalone flowed-quark ringed-field normalization.

This module computes the kinetic expectation value used to normalize ringed
flowed quark fields.  It is intentionally independent of EMT contractions:
the resulting factors can be consumed by any flowed-quark operator measured with
the same Dirac operator, gauge preprocessing, and flow schedule.
"""

from pathlib import Path
from time import perf_counter

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
from pyquda_measurement_utils.tools import (
    _asarray_on_queue,
    _get_xp_from_array,
    mpi_print,
    mpi_timer_print,
    timing_enabled,
)

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


def flowed_quark_ringed_norm_block_tag(tag, block_index, block_start, block_stop_exclusive):
    """Return the output tag for an interval block HDF5 file."""
    return (
        f"{tag}.block{int(block_index):04d}"
        f".src{int(block_start):06d}-{int(block_stop_exclusive) - 1:06d}"
    )


def flowed_quark_ringed_norm_sample_seed(randseed, base_idx):
    """Return a deterministic mixed seed for one base noise."""
    mask = (1 << 64) - 1
    value = ((int(randseed) & mask) << 32) ^ (int(base_idx) & mask) ^ 0x9E3779B97F4A7C15
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value = value ^ (value >> 31)
    seed = int(value % (2**31 - 1))
    return seed or 1


def flowed_quark_ringed_norm_hp256_sample_log_tag(base_idx, sample_seed):
    """Return the connected-style completion tag for one HP256 base noise."""
    return f"ringed_hp256_base{int(base_idx):03d}_seed{int(sample_seed)}"


def hp256_sample_source_range(base_idx, hp_num_vectors=256):
    """Return the absolute source range for one HP base-noise sample."""
    source_start = int(base_idx) * int(hp_num_vectors)
    return source_start, source_start + int(hp_num_vectors)


def hp256_sample_block_ranges(base_idx, block_interval_solves, hp_num_vectors=256):
    """Return interval block ranges covering one HP base-noise sample."""
    source_start, source_stop = hp256_sample_source_range(base_idx, hp_num_vectors)
    block_interval_solves = int(block_interval_solves)
    if int(hp_num_vectors) % block_interval_solves != 0:
        raise ValueError("hp_num_vectors should be divisible by block_interval_solves for HP256 sample logging")
    ranges = []
    for block_start in range(source_start, source_stop, block_interval_solves):
        block_stop = block_start + block_interval_solves
        block_index = block_start // block_interval_solves
        ranges.append((block_index, block_start, block_stop))
    return ranges


def _read_sample_log_tags(sample_log_file):
    path = Path(sample_log_file)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_sample_log_tag_once(sample_log_file, sample_log_tag):
    path = Path(sample_log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_tags = _read_sample_log_tags(path)
    if sample_log_tag in completed_tags:
        return False
    with path.open("a+") as f:
        f.write(sample_log_tag + "\n")
    return True


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


def _sync_backend_object(obj):
    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _sync_backend_object(item)
        return

    data = getattr(obj, "data", obj)
    queue = getattr(data, "sycl_queue", None)
    if queue is not None:
        queue.wait()
        return

    stream = getattr(data, "stream", None)
    if stream is not None and hasattr(stream, "synchronize"):
        stream.synchronize()
        return

    synchronize = getattr(data, "synchronize", None)
    if synchronize is not None:
        synchronize()
        return

    if type(data).__module__.split(".")[0] == "cupy":
        try:
            import cupy

            cupy.cuda.get_current_stream().synchronize()
        except Exception:
            pass


def _timer_start(*objects):
    if not timing_enabled():
        return None
    for obj in objects:
        _sync_backend_object(obj)
    from pyquda import getMPIComm

    getMPIComm().Barrier()
    return perf_counter()


def _timer_stop(start, *objects):
    if start is None:
        return 0.0
    for obj in objects:
        _sync_backend_object(obj)
    from pyquda import getMPIComm

    getMPIComm().Barrier()
    return perf_counter() - start


def _reset_block_timers(n_flow, flow_steps):
    return {
        "block_start": None,
        "invert": 0.0,
        "contract": np.zeros(n_flow, dtype=np.float64),
        "flow": np.zeros(max(int(flow_steps), 0), dtype=np.float64),
        "write": 0.0,
    }


def _print_block_timers(latt_info, block_index, block_start, block_stop, timers, total_seconds):
    source_count = int(block_stop - block_start)
    per_source = total_seconds / source_count if source_count else np.nan
    mpi_timer_print(
        latt_info,
        "ringed_norm_block",
        total_seconds,
        block=block_index,
        sources=source_count,
        source_start=block_start,
        source_stop_exclusive=block_stop,
        per_source=per_source,
    )
    mpi_timer_print(
        latt_info,
        "ringed_norm_invert",
        timers["invert"],
        block=block_index,
        sources=source_count,
        per_source=timers["invert"] / source_count if source_count else np.nan,
    )
    for step, seconds in enumerate(timers["contract"]):
        mpi_timer_print(
            latt_info,
            "ringed_norm_contract",
            float(seconds),
            block=block_index,
            step=step,
            sources=source_count,
            per_source=float(seconds) / source_count if source_count else np.nan,
        )
    for step, seconds in enumerate(timers["flow"]):
        mpi_timer_print(
            latt_info,
            "ringed_norm_flow",
            float(seconds),
            block=block_index,
            step=f"{step}_to_{step + 1}",
            sources=source_count,
            per_source=float(seconds) / source_count if source_count else np.nan,
        )
    mpi_timer_print(
        latt_info,
        "ringed_norm_block_write",
        timers["write"],
        block=block_index,
        sources=source_count,
    )


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
        self.sample_log_file = parameters.get("sample_log_file")
        self.sample_log_mode = parameters.get("sample_log_mode", "hp256_base_noise")
        self.base_start = parameters.get("base_start")
        self.base_stop = parameters.get("base_stop")
        self.block_size = int(parameters.get("block_interval_solves", 64))
        if self.block_size <= 0:
            raise ValueError(f"block_interval_solves should be positive, got {self.block_size}")
        self.natural_block_size = natural_estimator_block_size(
            self.noise_scheme,
            self.hp_num_vectors,
            self.spin_color_dilution,
        )
        validate_hierarchical_probing_options(self.hp_num_vectors, self.hp_ordering)
        if self.sample_log_file is not None:
            self._validate_sample_log_options()

    def _validate_sample_log_options(self):
        if self.sample_log_mode != "hp256_base_noise":
            raise ValueError(f"unsupported sample_log_mode {self.sample_log_mode!r}")
        if self.noise_scheme != "hierarchical_probing" or self.hp_num_vectors != 256 or self.spin_color_dilution != "none":
            raise ValueError("sample_log_file currently supports only HP256 hierarchical probing without spin-color dilution")
        if self.natural_block_size != 256:
            raise ValueError(f"HP256 sample log expects natural_block_size=256, got {self.natural_block_size}")
        if self.natural_block_size % self.block_size != 0:
            raise ValueError("HP256 sample log requires block_interval_solves to divide 256")

    def _sample_seed(self, randseed, base_idx):
        return flowed_quark_ringed_norm_sample_seed(randseed, base_idx)

    def _sample_log_tag(self, base_idx, randseed):
        return flowed_quark_ringed_norm_hp256_sample_log_tag(base_idx, self._sample_seed(randseed, base_idx))

    def _selected_base_indices(self, n_vec):
        n_vec = int(n_vec)
        base_start = 0 if self.base_start is None else int(self.base_start)
        base_stop = n_vec if self.base_stop is None else int(self.base_stop)
        if base_start < 0 or base_stop > n_vec or base_start >= base_stop:
            raise ValueError(f"invalid base range [{base_start}, {base_stop}) for n_vec={n_vec}")
        return set(range(base_start, base_stop))

    def _sample_block_files_exist(self, tag, base_idx):
        for block_index, block_start, block_stop in hp256_sample_block_ranges(base_idx, self.block_size, self.hp_num_vectors):
            block_tag = flowed_quark_ringed_norm_block_tag(tag, block_index, block_start, block_stop)
            if not Path(block_tag + ".h5").exists():
                return False
        return True

    def _completed_sample_log_tags(self, latt_info, tag, randseed, n_vec):
        if self.sample_log_file is None:
            return set()

        from pyquda import getMPIComm

        if latt_info.mpi_rank == 0:
            sample_log_path = Path(self.sample_log_file)
            sample_log_path.parent.mkdir(parents=True, exist_ok=True)
            sample_log_path.touch(exist_ok=True)
            logged_tags = _read_sample_log_tags(sample_log_path)
            completed_tags = {
                self._sample_log_tag(base_idx, randseed)
                for base_idx in range(int(n_vec))
                if self._sample_log_tag(base_idx, randseed) in logged_tags and self._sample_block_files_exist(tag, base_idx)
            }
        else:
            completed_tags = None
        completed_tags = getMPIComm().bcast(completed_tags, root=0)
        return set(completed_tags)

    def _log_sample_done(self, latt_info, sample_log_tag):
        if self.sample_log_file is None:
            return
        if latt_info.mpi_rank == 0 and _append_sample_log_tag_once(self.sample_log_file, sample_log_tag):
            print(f"RingedNorm LOGGED: {sample_log_tag}", flush=True)
        mpi_print(latt_info, f"RingedNorm DONE: {sample_log_tag}")

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
            "volume_norm": spatial_volume,
            "volume_average": "spin_color_trace_factor_times_spacetime_average_from_raw_kinetic_pervec",
            "flow0_factor": np.nan,
            "derivative_convention": "gamma_mu*(Dplus_mu-Dminus_mu)",
            "field_factor_dataset": "avg/Z_ring_field_sqrt",
            "bilinear_factor_dataset": "avg/Z_ring_bilinear",
            "natural_block_size": self.natural_block_size,
            "block_interval_solves": self.block_size,
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
        block_size = int(block_stop - block_start)
        estimator_remainder = block_size % self.natural_block_size
        block_attrs = dict(base_attrs)
        block_attrs.update(
            {
                "block_index": int(block_index),
                "block_start": int(block_start),
                "block_stop_exclusive": int(block_stop),
                "block_interval_solves": self.block_size,
                "estimator_complete": estimator_remainder == 0,
                "complete_estimator_units": block_size // self.natural_block_size,
                "estimator_remainder": estimator_remainder,
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

    def flowed_kinetic_norm(self, gauge, invPara, randPara, tag: str):
        """Compute and save the standalone flowed-quark ringed normalization."""
        if not tag:
            raise ValueError("flowed_kinetic_norm requires a non-empty output tag")

        n_vec, n_zn, randseed = randPara
        n_eff = effective_n_inversions(n_vec, self.noise_scheme, self.hp_num_vectors, self.spin_color_dilution)
        if n_eff % self.block_size != 0:
            raise ValueError(
                "effective_n_inversions must be divisible by block_interval_solves: "
                f"n_eff={n_eff}, block_interval_solves={self.block_size}"
            )

        from pyquda_utils import core, phase

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

        kinetic_pervec = np.zeros((n_eff, n_flow, nt), dtype=np.complex128)
        source_bookkeeping = source_bookkeeping_arrays(n_eff, include_spin_color=True)
        attrs = self._metadata_attrs(latt_info, invPara, randPara, n_eff, spatial_volume)
        mpi_print(
            latt_info,
            (
                "Flowed-quark ringed normalization interval block output enabled: "
                f"natural_block_size={self.natural_block_size}, block_interval_solves={self.block_size}"
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

        selected_base_indices = self._selected_base_indices(n_vec)
        completed_sample_tags = self._completed_sample_log_tags(latt_info, tag, randseed, n_vec)
        completed_base_indices = {
            base_idx
            for base_idx in range(int(n_vec))
            if self._sample_log_tag(base_idx, randseed) in completed_sample_tags
        } if self.sample_log_file is not None else set()
        out_of_range_base_indices = set(range(int(n_vec))) - selected_base_indices
        skipped_base_indices = completed_base_indices | out_of_range_base_indices
        for base_idx in sorted(completed_base_indices & selected_base_indices):
            mpi_print(latt_info, f"RingedNorm SKIP: {self._sample_log_tag(base_idx, randseed)}")
        if self.base_start is not None or self.base_stop is not None:
            mpi_print(latt_info, f"RingedNorm selected base range: {min(selected_base_indices)}:{max(selected_base_indices) + 1}")

        q0_phase = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0]], [0, 0, 0, 0])
        block_timers = _reset_block_timers(n_flow, self.flow_steps)
        computed_source_mask = np.zeros(n_eff, dtype=bool)
        for vec_picked, base_idx, hp_idx, spin_idx, color_idx, xi in iter_noise_sources(
            latt_info,
            n_vec,
            n_zn,
            self.noise_scheme,
            self.hp_num_vectors,
            self.hp_ordering,
            spin_color_dilution=self.spin_color_dilution,
            include_spin_color=True,
            skip_base_indices=skipped_base_indices,
            base_seed_fn=(lambda base_idx: self._sample_seed(randseed, base_idx)) if self.sample_log_file is not None else None,
        ):
            if self.sample_log_file is not None and hp_idx == 0:
                mpi_print(latt_info, f"RingedNorm START: {self._sample_log_tag(base_idx, randseed)}")
            if vec_picked % self.block_size == 0:
                block_timers = _reset_block_timers(n_flow, self.flow_steps)
                block_timers["block_start"] = _timer_start(xi)

            mpi_print(latt_info, f"ringed norm vec {vec_picked} base {base_idx} hp {hp_idx} spin {spin_idx} color {color_idx}")
            computed_source_mask[vec_picked] = True
            source_bookkeeping["base_noise_index"][vec_picked] = base_idx
            source_bookkeeping["hp_index"][vec_picked] = hp_idx
            source_bookkeeping["spin_index"][vec_picked] = spin_idx
            source_bookkeeping["color_index"][vec_picked] = color_idx

            t0 = _timer_start(xi)
            dirac.loadGauge(U)
            eta = dirac.invert(xi)
            block_timers["invert"] += _timer_stop(t0, eta)

            U_f = U.copy()
            U_f.setAntiPeriodicT()
            for step in range(n_flow):
                t0 = _timer_start(xi, eta)
                kinetic_pervec[vec_picked, step] = self._kinetic_per_time_for_source(
                    U_f,
                    xi,
                    eta,
                    q0_phase,
                    spatial_volume,
                )
                block_timers["contract"][step] += _timer_stop(t0, xi, eta)

                if step < self.flow_steps:
                    t0 = _timer_start(xi, eta)
                    xi, eta = self._advance_flowed_pair(U_f, xi, eta, step)
                    block_timers["flow"][step] += _timer_stop(t0, xi, eta)

            del U_f, xi, eta

            if (vec_picked + 1) % self.block_size == 0:
                block_stop = vec_picked + 1
                block_start = block_stop - self.block_size
                block_index = block_start // self.block_size
                mpi_print(
                    latt_info,
                    f"writing ringed norm block {block_index} sources {block_start}:{block_stop}",
                )
                t0 = _timer_start()
                if latt_info.mpi_rank == 0:
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
                block_timers["write"] = _timer_stop(t0)
                total_seconds = _timer_stop(block_timers["block_start"])
                _print_block_timers(latt_info, block_index, block_start, block_stop, block_timers, total_seconds)
                if self.sample_log_file is not None and block_stop % self.natural_block_size == 0:
                    self._log_sample_done(latt_info, self._sample_log_tag(base_idx, randseed))

        if np.any(computed_source_mask):
            kinetic_spacetime = kinetic_spacetime_from_raw(kinetic_pervec[computed_source_mask], self.spin_color_dilution_factor)
        else:
            kinetic_spacetime = np.full(n_flow, np.nan + 0j, dtype=np.complex128)
        z_field_sqrt, z_bilinear = compute_ringed_factors(kinetic_spacetime, flow_time_values, nc=self.nc)

        return kinetic_spacetime, z_field_sqrt, z_bilinear


__all__ = [
    "FlowedQuarkRingedNorm",
    "compute_ringed_factors",
    "flowed_quark_ringed_norm_block_tag",
    "flowed_quark_ringed_norm_hp256_sample_log_tag",
    "flowed_quark_ringed_norm_sample_seed",
    "flow_times",
    "hp256_sample_block_ranges",
    "hp256_sample_source_range",
    "natural_estimator_block_size",
    "normalize_flow_type",
]
