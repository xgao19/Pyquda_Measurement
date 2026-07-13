"""Standalone flowed-quark ringed-field normalization.

This module computes the kinetic expectation value used to normalize ringed
flowed quark fields.  It is intentionally independent of EMT contractions:
the resulting factors can be consumed by any flowed-quark operator measured with
the same Dirac operator, gauge preprocessing, and flow schedule.
"""

import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import h5py
import numpy as np
from opt_einsum import contract

from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    COUNTER_NOISE_ALGORITHM,
    array_to_numpy,
    effective_n_inversions,
    iter_noise_base_hp_interval,
    normalize_noise_scheme,
    normalize_spin_color_dilution,
    part_source_bookkeeping,
    spin_color_dilution_factor,
    validate_hierarchical_probing_options,
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
        if parameters.get("config_num") is None:
            raise ValueError("config_num is required for counter-based stochastic sources")
        self.config_num = int(parameters["config_num"])
        if self.config_num < 0:
            raise ValueError(f"config_num should be non-negative, got {self.config_num}")
        self.noise_scheme = normalize_noise_scheme(parameters.get("noise_scheme", "zn"))
        self.hp_num_vectors = int(parameters.get("hp_num_vectors", 1))
        self.hp_ordering = parameters.get("hp_ordering", "global_xyzt_gray_projected_to_evenodd")
        self.spin_color_dilution = normalize_spin_color_dilution(parameters.get("spin_color_dilution", "none"))
        self.spin_color_dilution_factor = spin_color_dilution_factor(self.spin_color_dilution)
        self.nc = int(parameters.get("Nc", 3))
        self.multigrid = parameters.get("multigrid", [[8, 8, 4, 4]])
        self.gauge_preprocessing = parameters.get("gauge_preprocessing", "unspecified")
        self.flavor_convention = parameters.get("flavor_convention", "single_flavor_trace_for_this_dirac_operator")
        self.shard_dir = parameters.get("shard_dir")
        self.sample_log_file = parameters.get("sample_log_file")
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

    def _metadata_attrs(
        self,
        latt_info,
        invPara,
        randPara,
        counter_config,
        counter_stream,
        n_eff,
        spatial_volume,
    ):
        n_vec, n_zn, noise_stream = randPara
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
            "config_num": int(counter_config),
            "noise_stream": int(counter_stream),
            "noise_generator": COUNTER_NOISE_ALGORITHM,
            "noise_counter_order": (
                "global_xyzt_spin_color_config_base_stream"
                if self.spin_color_dilution == "none"
                else "global_xyzt_config_base_stream"
            ),
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
            "natural_block_size": self.natural_block_size,
            "block_interval_solves": self.block_size,
            "content": "kinetic_only",
            "producer": "standalone_ringed",
            "ringed_factors_stored": False,
            "ringed_factor_stage": "ensemble_analysis_from_configuration_averaged_kinetic",
        }

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
        """Measure selected complete bases into resumable base/HP shard parts."""
        if not tag:
            raise ValueError("flowed_kinetic_norm requires a non-empty output tag")

        n_vec, n_zn, noise_stream = randPara
        if int(noise_stream) < 0:
            raise ValueError(f"noise stream should be non-negative, got {noise_stream}")
        n_eff = effective_n_inversions(
            n_vec, self.noise_scheme, self.hp_num_vectors,
            self.spin_color_dilution,
        )

        from pyquda_utils import core, phase

        mass, csw, tol, maxiter = invPara
        U = gauge
        latt_info = U.latt_info
        global_size = latt_info.global_size
        spatial_volume = global_size[0] * global_size[1] * global_size[2]
        nt = global_size[3]
        n_flow = self.flow_steps + 1

        counter_config, counter_stream = self.config_num, int(noise_stream)
        attrs = self._metadata_attrs(
            latt_info,
            invPara,
            randPara,
            counter_config,
            counter_stream,
            n_eff,
            spatial_volume,
        )
        hp_count = hp_vectors_per_base(self.noise_scheme, self.hp_num_vectors)
        solves_per_hp = self.spin_color_dilution_factor
        shard_dir = Path(self.shard_dir) if self.shard_dir else Path(tag).parent / "shards"
        common_attrs = {
            key: value for key, value in attrs.items()
            if key not in {"n_vec", "effective_n_inversions"}
        }
        common_attrs["output_kind"] = "flowed_quark_ringed_norm"
        from pyquda import getMPIComm
        comm = getMPIComm()
        if self.sample_log_file is None:
            raise ValueError("sample_log_file is required for base-level resume")
        if latt_info.mpi_rank == 0:
            completed_bases = prepare_sample_log(
                self.sample_log_file, tag, common_attrs
            )
        else:
            completed_bases = None
        completed_bases = set(comm.bcast(completed_bases, root=0))
        selected_bases = list(selected_base_range(n_vec, self.base_start or 0, self.base_stop))
        if all(base_idx in completed_bases for base_idx in selected_bases):
            mpi_print(latt_info, "All selected ringed bases are complete in the sample log.")
            return None

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
        q0_phase = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0]], [0, 0, 0, 0])

        for base_idx in selected_bases:
            if base_idx in completed_bases:
                mpi_print(latt_info, f"Ringed base SKIP from sample log: base{base_idx:06d}")
                continue
            for part_idx, hp_start, hp_stop in base_part_ranges(
                hp_count, self.block_size, solves_per_hp
            ):
                bookkeeping = part_source_bookkeeping(
                    base_idx, hp_start, hp_stop, hp_count,
                    self.spin_color_dilution, include_spin_color=True,
                )
                count = len(bookkeeping["source_index"])
                path = shard_part_path(
                    shard_dir, tag, base_idx, part_idx, hp_start, hp_stop
                )
                write_attrs = shard_part_attrs(
                    common_attrs, base_idx, part_idx, hp_start, hp_stop, hp_count,
                    solves_per_hp=solves_per_hp,
                    spin_color_dilution=self.spin_color_dilution,
                )

                kinetic_part = np.zeros((count, n_flow, nt), dtype=np.complex128)
                timers = _reset_block_timers(n_flow, self.flow_steps)
                timers["block_start"] = _timer_start()
                for local_idx, fields in enumerate(iter_noise_base_hp_interval(
                    latt_info, base_idx, hp_start, hp_stop, n_zn,
                    self.noise_scheme, self.hp_num_vectors, self.hp_ordering,
                    config_num=counter_config, noise_stream=counter_stream,
                    spin_color_dilution=self.spin_color_dilution,
                    include_spin_color=True,
                )):
                    _, _, hp_idx, spin_idx, color_idx, xi = fields
                    mpi_print(
                        latt_info,
                        f"ringed base {base_idx} hp {hp_idx} spin {spin_idx} color {color_idx}",
                    )
                    t0 = _timer_start(xi)
                    # Fermion flow leaves U_f resident; restore the inversion gauge.
                    dirac.loadGauge(U)
                    eta = dirac.invert(xi)
                    timers["invert"] += _timer_stop(t0, eta)
                    U_f = U.copy()
                    U_f.setAntiPeriodicT()
                    for step in range(n_flow):
                        t0 = _timer_start(xi, eta)
                        kinetic_part[local_idx, step] = self._kinetic_per_time_for_source(
                            U_f, xi, eta, q0_phase, spatial_volume
                        )
                        timers["contract"][step] += _timer_stop(t0, xi, eta)
                        if step < self.flow_steps:
                            t0 = _timer_start(xi, eta)
                            xi, eta = self._advance_flowed_pair(U_f, xi, eta, step)
                            timers["flow"][step] += _timer_stop(t0, xi, eta)
                    del U_f, xi, eta

                t0 = _timer_start()
                if latt_info.mpi_rank == 0:
                    write_raw_part_hdf5(
                        path, {"kinetic_pervec": kinetic_part}, write_attrs,
                        bookkeeping,
                    )
                timers["write"] = _timer_stop(t0)
                _print_block_timers(
                    latt_info, part_idx, int(bookkeeping["source_index"][0]),
                    int(bookkeeping["source_index"][-1]) + 1, timers,
                    _timer_stop(timers["block_start"]),
                )
                comm.Barrier()

            if latt_info.mpi_rank == 0:
                append_completed_base(
                    self.sample_log_file, tag, common_attrs, base_idx
                )
            comm.Barrier()
        return None


def finalize_flowed_quark_ringed_norm_shards(shard_dir, canonical_tag, n_base_noise):
    """Stream complete standalone ringed shards into one kinetic-only file."""
    manifest = discover_shard_layout(
        shard_dir, canonical_tag, n_base_noise,
        raw_dataset_names=("kinetic_pervec",),
        include_spin_color=True,
    )
    attrs = {
        key: value for key, value in manifest["reference_attrs"].items()
        if key not in {
            "shard_schema", "output_kind", "block_interval_solves",
            "hp_vectors_per_base",
        }
    }
    total_sources = manifest["total_sources"]
    n_base_noise = int(n_base_noise)
    attrs.update({
        "measurement": "flowed_quark_ringed_norm",
        "content": "kinetic_only",
        "n_vec": n_base_noise,
        "n_base_noise": n_base_noise,
        "effective_n_inversions": total_sources,
        "ringed_factors_stored": False,
    })
    kinetic_tail = manifest["raw_tails"]["kinetic_pervec"]
    nt = kinetic_tail[-1]
    trace_factor = int(attrs["spin_color_trace_factor"])
    final_path, temp_path = canonical_temp_path(canonical_tag)
    with h5py.File(temp_path, "w") as out:
        for key, value in attrs.items():
            out.attrs[key] = value
        out.create_dataset("flow_times", data=np.asarray(attrs["flow_times"], dtype=np.float64))
        raw = out.require_group("raw")
        kinetic = raw.create_dataset(
            "kinetic_pervec", shape=(total_sources,) + kinetic_tail,
            dtype=np.complex128,
        )
        book = {
            name: raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in (
                "source_index", "base_noise_index", "hp_index",
                "spin_index", "color_index",
            )
        }
        kinetic_sum = np.zeros(kinetic_tail[:-1], dtype=np.complex128)
        for info, part in iter_validated_shard_parts(manifest):
            start, stop = info["output_start"], info["output_stop"]
            values = part["raw/kinetic_pervec"][()]
            kinetic[start:stop] = values
            kinetic_sum += np.sum(values, axis=(0, -1))
            for name, dataset in book.items():
                dataset[start:stop] = part[f"raw/{name}"][()]
        out.require_group("avg").create_dataset(
            "kinetic_spacetime",
            data=trace_factor * kinetic_sum / total_sources / nt,
        )
        out.flush()
    os.replace(temp_path, final_path)
    return str(final_path)


def analyze_ringed_ensemble(input_files, output_file, nc=3):
    """Average per-configuration kinetic values, then compute ringed factors."""
    paths = [Path(path) for path in input_files]
    if not paths:
        raise ValueError("at least one explicit kinetic-only input file is required")
    configs = []
    kinetics = []
    reference = None
    flow_time_values = None
    match_keys = (
        "flow_type", "flow_epsilon", "flow_steps", "mass", "csw", "tol",
        "maxiter", "gauge_preprocessing", "t_boundary", "flavor_convention",
        "derivative_convention", "Nc",
    )
    for path in paths:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("content") != "kinetic_only":
                raise ValueError(f"{path} is not a kinetic-only ringed input")
            config_num = int(h5.attrs["config_num"])
            if config_num in configs:
                raise ValueError(f"duplicate configuration {config_num}")
            configs.append(config_num)
            current = {key: h5.attrs[key] for key in match_keys}
            times = h5["flow_times"][()]
            if reference is None:
                reference = current
                flow_time_values = times
            else:
                for key, expected in reference.items():
                    if not np.array_equal(np.asarray(current[key]), np.asarray(expected)):
                        raise ValueError(f"{path} has incompatible attribute {key}")
                if not np.array_equal(times, flow_time_values):
                    raise ValueError(f"{path} has incompatible flow_times")
            kinetics.append(h5["avg/kinetic_spacetime"][()])

    kinetic_ensemble = np.mean(np.asarray(kinetics), axis=0)
    z_field, z_bilinear = compute_ringed_factors(
        kinetic_ensemble, flow_time_values, nc=nc
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}.{uuid4().hex}"
    )
    with h5py.File(temp_path, "w") as out:
        out.attrs["measurement"] = "flowed_quark_ringed_norm_ensemble"
        out.attrs["ringed_factors_stored"] = True
        out.attrs["n_configurations"] = len(configs)
        out.attrs["configuration_numbers"] = np.asarray(configs, dtype=np.int64)
        for key, value in reference.items():
            out.attrs[key] = value
        out.create_dataset("flow_times", data=flow_time_values)
        avg = out.require_group("avg")
        avg.create_dataset("kinetic_ensemble", data=kinetic_ensemble)
        avg.create_dataset("Z_ring_field_sqrt", data=z_field)
        avg.create_dataset("Z_ring_bilinear", data=z_bilinear)
        out.flush()
    os.replace(temp_path, output_path)
    return str(output_path)


__all__ = [
    "FlowedQuarkRingedNorm",
    "analyze_ringed_ensemble",
    "compute_ringed_factors",
    "finalize_flowed_quark_ringed_norm_shards",
    "flow_times",
    "natural_estimator_block_size",
    "normalize_flow_type",
]
