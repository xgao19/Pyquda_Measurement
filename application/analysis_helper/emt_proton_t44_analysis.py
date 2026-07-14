"""Fixed-gauge proton connected/disconnected T44 ratio helpers.

The routines here operate only on already-produced HDF5 data.  ``T44`` is the
stored Euclidean temporal component, corresponding to the ``[3, 3]`` tensor
entry in the repository's ``X,Y,Z,T`` direction ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    VECTOR_GAMMA_POSITIONS,
)


T_GAMMA_POSITION = VECTOR_GAMMA_POSITIONS[3]
T_DERIVATIVE_POSITION = 3


@dataclass(frozen=True)
class T44BaseLoops:
    path: Path
    qext: np.ndarray
    flow_times: np.ndarray
    hp_count: int
    base_values: np.ndarray


def _splitmix64(values):
    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def deterministic_stratified_sources(
    lattice_size=(8, 8, 8, 8), config_num=1, stream=0, per_time=16
):
    """Return reproducible unique sites with exactly ``per_time`` per t slice."""
    lx, ly, lz, lt = (int(value) for value in lattice_size)
    per_time = int(per_time)
    if min(lx, ly, lz, lt, per_time) <= 0:
        raise ValueError("lattice extents and per_time should be positive")
    spatial_volume = lx * ly * lz
    if per_time > spatial_volume:
        raise ValueError("per_time exceeds the number of spatial sites")
    sites = []
    spatial = np.arange(spatial_volume, dtype=np.uint64)
    mask64 = (1 << 64) - 1
    config_key = np.uint64((int(config_num) * 0xD2B74407B1CE6E93) & mask64)
    stream_key = np.uint64((int(stream) * 0xCA5A826395121157) & mask64)
    for t in range(lt):
        time_key = np.uint64((t * 0x9E3779B185EBCA87) & mask64)
        key = (
            spatial
            ^ config_key
            ^ stream_key
            ^ time_key
        )
        chosen = np.argsort(_splitmix64(key), kind="stable")[:per_time]
        for linear in chosen:
            linear = int(linear)
            x = linear % lx
            y = (linear // lx) % ly
            z = linear // (lx * ly)
            sites.append((x, y, z, t))
    return np.asarray(sites, dtype=np.int32)


def pplus_unpolarized_c2(identity_channel, temporal_channel):
    """Apply the proton ``PpUnpol=(I+gamma_4)/4`` sink projector."""
    return 0.25 * (
        np.asarray(identity_channel, dtype=np.complex128)
        + np.asarray(temporal_channel, dtype=np.complex128)
    )


def read_pplus_c2(path, momenta):
    """Read source-relative proton C2 for requested momenta as ``[p,t]``."""
    path = Path(path)
    values = []
    with h5py.File(path, "r") as h5:
        for momentum in np.asarray(momenta, dtype=np.int32):
            label = f"PX{momentum[0]}PY{momentum[1]}PZ{momentum[2]}"
            values.append(pplus_unpolarized_c2(
                h5[f"SS/{GAMMA_LABELS[9]}/{label}"][...],
                h5[f"SS/{GAMMA_LABELS[1]}/{label}"][...],
            ))
    return np.asarray(values)


def load_t44_base_loops(path):
    """Load only T44 and group every complete randomized HP base."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if int(h5.attrs.get("emt_operator_schema_version", -1)) != 3:
            raise ValueError(f"{path} is not EMT operator schema 3")
        derivative = h5["raw/derivative_bilinear_pervec"]
        if derivative.ndim != 6 or derivative.shape[1:3] != (16, 4):
            raise ValueError(f"invalid derivative primitive shape {derivative.shape}")
        source_t44 = np.asarray(
            derivative[:, T_GAMMA_POSITION, T_DERIVATIVE_POSITION],
            dtype=np.complex128,
        ) / float(h5.attrs["volume_norm"])
        base_index = np.asarray(h5["raw/base_noise_index"], dtype=np.int64)
        hp_index = np.asarray(h5["raw/hp_index"], dtype=np.int64)
        scheme = h5.attrs["noise_scheme"]
        if isinstance(scheme, bytes):
            scheme = scheme.decode("utf-8")
        hp_count = (
            int(h5.attrs["hp_num_vectors"])
            if str(scheme) == "hierarchical_probing"
            else 1
        )
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)

    bases = np.unique(base_index)
    if not np.array_equal(bases, np.arange(len(bases), dtype=np.int64)):
        raise ValueError("base indices should be contiguous from zero")
    expected_hp = np.arange(hp_count, dtype=np.int64)
    base_values = np.empty((len(bases),) + source_t44.shape[1:], np.complex128)
    for base in bases:
        rows = np.flatnonzero(base_index == base)
        if not np.array_equal(np.sort(hp_index[rows]), expected_hp):
            raise ValueError(f"base {base} is not a complete HP estimator")
        base_values[base] = np.mean(source_t44[rows], axis=0)
    if len(source_t44) != len(bases) * hp_count:
        raise ValueError("source axis contains duplicated or incomplete HP vectors")
    return T44BaseLoops(path, qext, flow_times, hp_count, base_values)


def source_relative_loops(loop_qt, qext, source_positions, lattice_size):
    """Rephase and roll an origin-based absolute-time loop for every source."""
    loop_qt = np.asarray(loop_qt, dtype=np.complex128)
    qext = np.asarray(qext, dtype=np.int32)
    sources = np.asarray(source_positions, dtype=np.int32)
    if loop_qt.shape[-2:] != (len(qext), int(lattice_size[3])):
        raise ValueError("loop tail should be [q,t] and match qext/lattice time")
    spatial_size = np.asarray(lattice_size[:3], dtype=np.float64)
    phases = np.exp(
        -2j * np.pi * np.einsum(
            "qi,si->sq", qext[:, :3], sources[:, :3] / spatial_size
        )
    )
    time_indices = (
        sources[:, 3, None] + np.arange(int(lattice_size[3]))[None, :]
    ) % int(lattice_size[3])
    # np.take inserts the [source,t] index shape at the old time axis, giving
    # [...,q,source,t]; move source to the leading analysis axis.
    result = np.moveaxis(np.take(loop_qt, time_indices, axis=-1), -2, 0)
    phase_shape = (len(sources),) + (1,) * (loop_qt.ndim - 2) + (len(qext), 1)
    return result * phases.reshape(phase_shape)


def optimized_ratio(c3_qtau, c2_pf_t, c2_pi_qt, t_sep, taus=None):
    """Form the standard fixed-sink optimized ratio from averaged correlators."""
    c3 = np.asarray(c3_qtau, dtype=np.complex128)
    c2_pf = np.asarray(c2_pf_t, dtype=np.complex128)
    c2_pi = np.asarray(c2_pi_qt, dtype=np.complex128)
    t_sep = int(t_sep)
    taus = np.arange(c3.shape[-1], dtype=np.int32) if taus is None else np.asarray(taus, dtype=np.int32)
    if c3.shape[-2] != c2_pi.shape[-2] or c3.shape[-1] != len(taus):
        raise ValueError("C3 should end in [q,tau] matching C2 and taus")
    numerator = (
        c2_pi[..., t_sep - taus]
        * c2_pf[..., taus]
        * c2_pf[..., t_sep, None]
    )
    denominator = (
        c2_pf[..., t_sep - taus]
        * c2_pi[..., taus]
        * c2_pi[..., t_sep, None]
    )
    return c3 / c2_pf[..., t_sep, None] * np.sqrt(numerator / denominator)


def translation_covariance(c2_sink, relative_loops):
    """Unbiased source-translation covariance along the leading source axis."""
    c2_sink = np.asarray(c2_sink, dtype=np.complex128)
    loops = np.asarray(relative_loops, dtype=np.complex128)
    if c2_sink.ndim != 1 or loops.shape[0] != len(c2_sink) or len(c2_sink) < 2:
        raise ValueError("C2 and loop should share at least two source positions")
    shape = (len(c2_sink),) + (1,) * (loops.ndim - 1)
    covariance = np.mean(c2_sink.reshape(shape) * loops, axis=0)
    covariance -= np.mean(c2_sink) * np.mean(loops, axis=0)
    return len(c2_sink) / (len(c2_sink) - 1.0) * covariance


def disconnected_ratio(
    c2_pf_sources,
    c2_pi_sources,
    base_loops,
    qext,
    source_positions,
    lattice_size,
    t_sep,
    flow_index=1,
):
    """Fixed-gauge translation-covariance ratio using complete loop bases."""
    c2_pf_sources = np.asarray(c2_pf_sources, dtype=np.complex128)
    c2_pi_sources = np.asarray(c2_pi_sources, dtype=np.complex128)
    loop_mean = np.mean(np.asarray(base_loops, dtype=np.complex128), axis=0)
    loop_qt = loop_mean[:, int(flow_index), :]
    relative = source_relative_loops(loop_qt, qext, source_positions, lattice_size)
    taus = np.arange(1, int(t_sep), dtype=np.int32)
    c3 = translation_covariance(
        c2_pf_sources[:, int(t_sep)], relative[..., taus]
    )
    return optimized_ratio(
        c3,
        np.mean(c2_pf_sources, axis=0),
        np.mean(c2_pi_sources, axis=0),
        t_sep,
        taus,
    )


def delete_one_jackknife(estimator, *sample_arrays):
    """Evaluate ``estimator`` and its real/imaginary delete-one errors."""
    arrays = [np.asarray(values) for values in sample_arrays]
    n_sample = len(arrays[0])
    if n_sample < 2 or any(len(values) != n_sample for values in arrays):
        raise ValueError("jackknife arrays should share at least two samples")
    full = np.asarray(estimator(*arrays), dtype=np.complex128)
    replicas = np.stack([
        estimator(*(np.delete(values, idx, axis=0) for values in arrays))
        for idx in range(n_sample)
    ])
    center = np.mean(replicas, axis=0)
    factor = (n_sample - 1.0) / n_sample
    err_real = np.sqrt(factor * np.sum((replicas.real - center.real) ** 2, axis=0))
    err_imag = np.sqrt(factor * np.sum((replicas.imag - center.imag) ** 2, axis=0))
    return full, err_real, err_imag, replicas


def two_way_bootstrap_disconnected(
    c2_pf_sources,
    c2_pi_sources,
    base_loops,
    qext,
    source_positions,
    lattice_size,
    t_sep,
    flow_index=1,
    n_resamples=1000,
    seed=20260714,
):
    """Return full ratio plus source, base, and combined bootstrap errors."""
    c2_pf_sources = np.asarray(c2_pf_sources)
    c2_pi_sources = np.asarray(c2_pi_sources)
    base_loops = np.asarray(base_loops)
    source_positions = np.asarray(source_positions)
    rng = np.random.default_rng(int(seed))
    full = disconnected_ratio(
        c2_pf_sources, c2_pi_sources, base_loops, qext, source_positions,
        lattice_size, t_sep, flow_index,
    )
    n_resamples = int(n_resamples)
    source_draws = rng.integers(
        0, len(source_positions), (n_resamples, len(source_positions))
    )
    # Form all bootstrap base means with one BLAS contraction.  This avoids
    # allocating a full [Nbase,...] advanced-index copy for every resample.
    probabilities = np.full(len(base_loops), 1.0 / len(base_loops))
    base_counts = rng.multinomial(
        len(base_loops), probabilities, size=n_resamples
    )
    base_means = (
        base_counts @ base_loops.reshape(len(base_loops), -1)
        / len(base_loops)
    ).reshape((n_resamples,) + base_loops.shape[1:])
    source_samples = []
    base_samples = []
    combined_samples = []
    for resample_idx in range(n_resamples):
        src_idx = source_draws[resample_idx]
        sampled_base_mean = base_means[resample_idx : resample_idx + 1]
        source_samples.append(disconnected_ratio(
            c2_pf_sources[src_idx], c2_pi_sources[src_idx], base_loops,
            qext, source_positions[src_idx], lattice_size, t_sep, flow_index,
        ))
        base_samples.append(disconnected_ratio(
            c2_pf_sources, c2_pi_sources, sampled_base_mean, qext,
            source_positions, lattice_size, t_sep, flow_index,
        ))
        combined_samples.append(disconnected_ratio(
            c2_pf_sources[src_idx], c2_pi_sources[src_idx], sampled_base_mean,
            qext, source_positions[src_idx], lattice_size, t_sep, flow_index,
        ))
    errors = {}
    for name, samples in (
        ("source", source_samples), ("stochastic", base_samples),
        ("combined", combined_samples),
    ):
        samples = np.asarray(samples)
        errors[name] = {
            "real": np.std(samples.real, axis=0, ddof=1),
            "imag": np.std(samples.imag, axis=0, ddof=1),
        }
    return full, errors


__all__ = [
    "T44BaseLoops",
    "T_DERIVATIVE_POSITION",
    "T_GAMMA_POSITION",
    "delete_one_jackknife",
    "deterministic_stratified_sources",
    "disconnected_ratio",
    "load_t44_base_loops",
    "optimized_ratio",
    "pplus_unpolarized_c2",
    "read_pplus_c2",
    "source_relative_loops",
    "translation_covariance",
    "two_way_bootstrap_disconnected",
]
