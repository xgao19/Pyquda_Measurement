"""Shared noise, probing, shard, and resume helpers for disconnected 1pt."""

import errno
import fcntl
import hashlib
import json
import os
import operator
import re
from functools import lru_cache
from itertools import product
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


VALID_NOISE_SCHEMES = {"zn", "hierarchical_probing", "multiplier_probing"}
VALID_SPIN_COLOR_DILUTIONS = {"none", "point"}
SPIN_COLOR_POINT_FACTOR = 12
COUNTER_NOISE_ALGORITHM = "splitmix64_global_coordinate_v1"
SHARD_SCHEMA = "disconnected_1pt_base_parts_v4"
SAMPLE_LOG_SCHEMA = "disconnected_sample_log_v2"
SOURCE_BOOKKEEPING_SCHEMA = "base_probe_v2"
SOURCE_BOOKKEEPING_DILUTED_SCHEMA = "base_probe_spin_color_v3"
NESTED_HP_ORDERING = "nested_xyzt_checkerboard_v1"
COMPLETE_HP_LEVELS = {2: 1, 32: 2, 512: 4}
VALID_HP_ORDERINGS = {
    NESTED_HP_ORDERING,
}

_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
_CONFIG_SALT = np.uint64(0xD1B54A32D192ED03)
_BASE_NOISE_SALT = np.uint64(0x8CB92BA72F3D8DD7)
_STREAM_SALT = np.uint64(0xDB4F0B9175AE2165)
_BASE_LINE = re.compile(r"base([0-9]{6})")
_FINGERPRINT_EXCLUDED = {
    "n_vec",
    "n_base_noise",
    "effective_n_inversions",
    "base_start",
    "base_stop",
    "shard_dir",
    "sample_log_file",
}
_PART_ATTRS = {
    "base_noise_index",
    "part_index",
    "probe_start",
    "probe_stop_exclusive",
    "shard_mean_vector_count",
}


def normalize_noise_scheme(noise_scheme: str) -> str:
    """Normalize and validate the stochastic noise scheme name."""
    scheme = str(noise_scheme).strip().lower()
    if scheme not in VALID_NOISE_SCHEMES:
        raise ValueError(f"noise_scheme should be one of {VALID_NOISE_SCHEMES}, got {noise_scheme!r}")
    return scheme


def normalize_spin_color_dilution(spin_color_dilution: str) -> str:
    """Normalize and validate the spin-color dilution mode."""
    mode = str(spin_color_dilution).strip().lower()
    if mode not in VALID_SPIN_COLOR_DILUTIONS:
        raise ValueError(
            "spin_color_dilution should be one of "
            f"{VALID_SPIN_COLOR_DILUTIONS}, got {spin_color_dilution!r}"
        )
    return mode


def spin_color_dilution_factor(spin_color_dilution="none"):
    """Return the solve multiplier for a complete spin-color dilution basis."""
    mode = normalize_spin_color_dilution(spin_color_dilution)
    return SPIN_COLOR_POINT_FACTOR if mode == "point" else 1


def validate_hierarchical_probing_options(hp_num_vectors: int, hp_ordering: str):
    """Validate common hierarchical probing options."""
    if hp_ordering not in VALID_HP_ORDERINGS:
        raise ValueError(f"hp_ordering should be one of {VALID_HP_ORDERINGS}, got {hp_ordering!r}")
    count = _integer(hp_num_vectors, "hp_num_vectors")
    if count not in COMPLETE_HP_LEVELS:
        raise ValueError("complete HP requires hp_num_vectors in {2, 32, 512}")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def validate_nested_hp_shape(shape, count):
    """Validate the three complete four-dimensional checkerboard HP levels."""
    validate_hierarchical_probing_options(count, NESTED_HP_ORDERING)
    shape = tuple(_integer(L, "extent") for L in shape)
    if len(shape) != 4 or any(L < 2 for L in shape):
        raise ValueError("nested HP requires four extents >= 2")
    q = COMPLETE_HP_LEVELS[count]
    if any(L % (2 * q) for L in shape):
        raise ValueError(f"HP{count} requires each extent divisible by {2*q}")
    return shape


def hierarchical_nested_index(latt_info):
    """Encode the supported complete HP layers using at most nine label bits."""
    shape = validate_nested_hp_shape(latt_info.global_size, 2)
    coords = tuple(np.asarray(c, dtype=np.int64) for c in latt_info.coordinate())
    if len(coords) != 4 or any(c.shape != coords[0].shape for c in coords):
        raise ValueError("nested HP requires matching global XYZT coordinate arrays")
    label = np.bitwise_xor.reduce([c & 1 for c in coords])
    out_bit = 1
    for r in range(min(2, min(L.bit_length() - 1 for L in shape) - 1)):
        for c in coords[:3]:
            label |= ((c >> r) & 1) << out_bit
            out_bit += 1
        parity = np.bitwise_xor.reduce([(c >> (r + 1)) & 1 for c in coords])
        label |= parity << out_bit
        out_bit += 1
    return label


def effective_n_inversions(
    n_vec: int,
    noise_scheme: str,
    hp_num_vectors: int,
    multiplier_plan=None,
    spin_color_dilution="none",
) -> int:
    """Return the number of effective stochastic inversions."""
    hp_factor = vectors_per_base(noise_scheme, hp_num_vectors, multiplier_plan)
    return n_vec * hp_factor * spin_color_dilution_factor(spin_color_dilution)


@lru_cache(maxsize=64)
def _check_multiplier_torus(shape, distance, colors, sigma):
    """Check all realizable wrapped coordinate differences, not only origin pairs."""
    ranges = [range(-min(distance, L // 2), min(distance, L // 2) + 1) for L in shape]
    for delta in product(*ranges):
        if not 0 < sum(abs(dx) for dx in delta) <= distance:
            continue
        choices = [
            (0,) if dx == 0 else (dx, dx - L if dx > 0 else dx + L)
            for dx, L in zip(delta, shape)
        ]
        for actual in product(*choices):
            if sum(s * dx for s, dx in zip(sigma, actual)) % colors == 0:
                raise ValueError(f"invalid periodic multiplier plan: same-color displacement {actual}")


def validate_multiplier_plan(plan, shape=None):
    """Return a canonical, independently distance-checked linear coloring plan."""
    if not isinstance(plan, dict):
        raise ValueError("multiplier_plan must be a dictionary")
    required = {"shape", "distance", "colors", "sigma", "version"}
    if not required <= plan.keys():
        raise ValueError(f"multiplier_plan requires {sorted(required)}")
    extents = tuple(_integer(L, "extent") for L in plan["shape"])
    sigma = tuple(_integer(s, "sigma") for s in plan["sigma"])
    distance = _integer(plan["distance"], "distance")
    colors = _integer(plan["colors"], "colors")
    version = _integer(plan["version"], "version")
    if version != 1 or len(extents) != 4 or len(sigma) != 4:
        raise ValueError("multiplier requires version=1 and four XYZT extents/coefficients")
    if any(L < 2 for L in extents) or distance < 1 or not 1 <= colors <= np.iinfo(np.int32).max:
        raise ValueError("multiplier extents must be >=2; distance/colors must be positive")
    if shape is not None and tuple(shape) != extents:
        raise ValueError("multiplier plan shape does not match the lattice")
    sigma = tuple(s % colors for s in sigma)
    _check_multiplier_torus(extents, distance, colors, sigma)
    return dict(shape=list(extents), distance=distance, colors=colors, sigma=list(sigma), version=version)


def broadcast_multiplier_plan(plan_or_path, comm):
    """Read and validate on rank zero, broadcasting failures as well as the plan."""
    payload = None
    if comm.Get_rank() == 0:
        try:
            plan = plan_or_path
            if isinstance(plan, (str, Path)):
                with Path(plan).open(encoding="utf-8") as handle:
                    plan = json.load(handle)
            payload = (None, validate_multiplier_plan(plan))
        except (OSError, ValueError, TypeError) as error:
            payload = (str(error), None)
    error, plan = comm.bcast(payload, root=0)
    if error is not None:
        raise ValueError(f"multiplier plan initialization failed: {error}")
    return plan


def multiplier_probe_pattern(latt_info, probe_idx, plan):
    """Build one unit-modulus Fourier pattern from a verified periodic coloring."""
    plan = validate_multiplier_plan(plan, latt_info.global_size)
    probe_idx = _integer(probe_idx, "probe_idx")
    colors = plan["colors"]
    if not 0 <= probe_idx < colors:
        raise ValueError(f"probe_idx must be in [0, {colors})")
    coords = tuple(np.asarray(c, dtype=np.int64) for c in latt_info.coordinate())
    if len(coords) != 4 or any(c.shape != coords[0].shape for c in coords):
        raise ValueError("multiplier requires matching global XYZT coordinate arrays")
    site_colors = np.zeros_like(coords[0])
    for s, c in zip(plan["sigma"], coords):
        site_colors = (site_colors + (s * (c % colors)) % colors) % colors
    table = np.exp(2j * np.pi * ((probe_idx * np.arange(colors)) % colors) / colors)
    return table[site_colors]


def apply_multiplier_probe(xi, probe_idx, plan):
    """Apply a complete-basis Fourier character without changing the base noise."""
    pattern = multiplier_probe_pattern(xi.latt_info, probe_idx, plan)
    probed = xi.copy()
    pattern = _asarray_on_queue(pattern, _get_xp_from_array(xi.data), xi.data)
    probed.data[:] *= pattern[..., None, None]
    return probed


def apply_spin_color_point_dilution(xi, spin_idx, color_idx):
    """Project a fermion source onto one exact spin-color channel."""
    spin_idx = _integer(spin_idx, "spin_idx")
    color_idx = _integer(color_idx, "color_idx")
    diluted = xi.copy()
    if not 0 <= spin_idx < diluted.data.shape[-2]:
        raise ValueError(
            f"spin_idx {spin_idx} outside spin dimension {diluted.data.shape[-2]}"
        )
    if not 0 <= color_idx < diluted.data.shape[-1]:
        raise ValueError(
            f"color_idx {color_idx} outside color dimension {diluted.data.shape[-1]}"
        )
    selected = xi.data[..., spin_idx, color_idx].copy()
    diluted.data[:] = 0
    diluted.data[..., spin_idx, color_idx] = selected
    return diluted


def _splitmix64(values):
    """Apply the fixed SplitMix64 finalizer with unsigned wraparound."""
    z = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z + _SPLITMIX_GAMMA) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2) & _UINT64_MASK
    return z ^ (z >> np.uint64(31))


def counter_zn_phase_indices(
    latt_info,
    config_num: int,
    base_noise_index: int,
    stream_seed: int = 0,
    n: int = 4,
    spin_count: int = 4,
    color_count: int = 3,
):
    """Build decomposition-independent Z_n phase indices from global coordinates."""
    config_num = int(config_num)
    base_noise_index = int(base_noise_index)
    stream_seed = int(stream_seed)
    n = int(n)
    if config_num < 0 or base_noise_index < 0 or stream_seed < 0:
        raise ValueError("counter-noise configuration, base index, and stream seed must be non-negative")
    if n <= 0:
        raise ValueError(f"n should be positive, got {n}")

    coords = latt_info.coordinate()
    x, y, z, t = [np.asarray(coords[mu], dtype=np.uint64) for mu in range(4)]
    Gx, Gy, Gz, _ = [int(extent) for extent in latt_info.global_size]
    site_id = x + np.uint64(Gx) * (y + np.uint64(Gy) * (z + np.uint64(Gz) * t))
    config_key = _splitmix64(np.uint64(config_num) ^ _CONFIG_SALT)
    base_key = _splitmix64(np.uint64(base_noise_index) ^ _BASE_NOISE_SALT)
    stream_key = _splitmix64(np.uint64(stream_seed) ^ _STREAM_SALT)
    common_key = config_key ^ base_key ^ stream_key
    phase_dtype = np.min_scalar_type(n - 1)
    phase_indices = np.empty(site_id.shape + (spin_count, color_count), dtype=phase_dtype)
    for spin_idx in range(spin_count):
        for color_idx in range(color_count):
            spin_color_idx = spin_idx * color_count + color_idx
            counter = site_id * np.uint64(spin_count * color_count) + np.uint64(spin_color_idx)
            hashed = _splitmix64(counter ^ common_key)
            phase_indices[..., spin_idx, color_idx] = hashed % np.uint64(n)
    return phase_indices


def make_counter_zn_noise_fermion(
    latt_info,
    config_num: int,
    base_noise_index: int,
    stream_seed: int = 0,
    n: int = 4,
):
    """Create full-volume counter-based Z_n noise invariant under MPI decomposition."""
    from pyquda.field import LatticeFermion

    xi = LatticeFermion(latt_info)
    xp = _get_xp_from_array(xi.data)
    phase_indices = counter_zn_phase_indices(
        latt_info,
        config_num,
        base_noise_index,
        stream_seed=stream_seed,
        n=n,
        spin_count=xi.data.shape[-2],
        color_count=xi.data.shape[-1],
    )
    if int(n) == 4:
        phase_table = np.asarray([1.0, 1.0j, -1.0, -1.0j], dtype=np.dtype(xi.data.dtype))
    elif int(n) == 2:
        phase_table = np.asarray([1.0, -1.0], dtype=np.dtype(xi.data.dtype))
    else:
        phase_table = np.exp(2j * np.pi * np.arange(int(n)) / int(n)).astype(np.dtype(xi.data.dtype))
    phases = phase_table[phase_indices]
    xi.data[:] = _asarray_on_queue(phases, xp, xi.data)
    return xi


def hierarchical_probe_pattern(latt_info, probe_idx: int, hp_ordering: str):
    """Build a site-only Rademacher probing vector in even-odd layout."""
    validate_hierarchical_probing_options(2, hp_ordering)
    probe_idx = _integer(probe_idx, "probe_idx")
    shape = validate_nested_hp_shape(latt_info.global_size, 2)
    max_count = max(n for n, q in COMPLETE_HP_LEVELS.items()
                    if all(L % (2 * q) == 0 for L in shape))
    if not 0 <= probe_idx < max_count:
        raise ValueError(f"probe_idx must be in [0, {max_count})")
    if probe_idx == 0:
        return np.ones_like(latt_info.coordinate(0), dtype=np.float64)

    label = hierarchical_nested_index(latt_info)
    parity = np.zeros_like(label, dtype=bool)
    mask = int(probe_idx)
    bit = 1
    while bit <= mask:
        if mask & bit:
            parity ^= (label & bit) != 0
        bit <<= 1
    return np.where(parity, -1.0, 1.0)


def apply_hierarchical_probe(xi, probe_idx: int, hp_ordering: str):
    """Multiply a base stochastic source by one hierarchical probing vector."""
    probed = xi.copy()
    xp = _get_xp_from_array(xi.data)
    pattern = hierarchical_probe_pattern(xi.latt_info, probe_idx, hp_ordering)
    pattern = _asarray_on_queue(pattern, xp, xi.data)
    probed.data[:] *= pattern[..., None, None]
    return probed


def part_source_bookkeeping(
    base_idx: int,
    probe_start: int,
    probe_stop: int,
    probe_count: int,
    spin_color_dilution="none",
):
    """Return deterministic global indices for one base/HP shard part."""
    base_idx = int(base_idx)
    probe_start = int(probe_start)
    probe_stop = int(probe_stop)
    probe_count = int(probe_count)
    if base_idx < 0 or probe_count <= 0 or not 0 <= probe_start < probe_stop <= probe_count:
        raise ValueError(
            f"invalid base/HP interval: base={base_idx}, hp=[{probe_start}, {probe_stop}), "
            f"probe_count={probe_count}"
        )

    mode = normalize_spin_color_dilution(spin_color_dilution)
    base_indices, hp_indices, spin_indices, color_indices = [], [], [], []
    for probe_idx in range(probe_start, probe_stop):
        channels = product(range(4), range(3)) if mode == "point" else [(-1, -1)]
        for spin_idx, color_idx in channels:
            base_indices.append(base_idx)
            hp_indices.append(probe_idx)
            spin_indices.append(spin_idx)
            color_indices.append(color_idx)

    bookkeeping = {
        "base_noise_index": np.asarray(base_indices, dtype=np.int32),
        "probe_index": np.asarray(hp_indices, dtype=np.int32),
    }
    if mode == "point":
        bookkeeping["spin_index"] = np.asarray(spin_indices, dtype=np.int32)
        bookkeeping["color_index"] = np.asarray(color_indices, dtype=np.int32)
    return bookkeeping


def reconstruct_source_indices(base_noise_index, probe_index, probe_count):
    """Reconstruct the effective source index from base and HP indices."""
    base = np.asarray(base_noise_index, dtype=np.int64)
    hp = np.asarray(probe_index, dtype=np.int64)
    probe_count = int(probe_count)
    if base.shape != hp.shape:
        raise ValueError("base_noise_index and probe_index should have the same shape")
    if probe_count <= 0 or np.any(base < 0) or np.any(hp < 0) or np.any(hp >= probe_count):
        raise ValueError("invalid base/HP bookkeeping")
    return (base * probe_count + hp).astype(np.int64, copy=False)


def iter_noise_base_probe_interval(
    latt_info,
    base_idx: int,
    probe_start: int,
    probe_stop: int,
    n_zn: int,
    noise_scheme: str,
    hp_num_vectors: int,
    hp_ordering: str,
    config_num: int,
    noise_stream: int = 0,
    multiplier_plan=None,
    spin_color_dilution="none",
):
    """Yield only the requested HP interval of one deterministic base noise."""
    if config_num is None:
        raise ValueError("config_num is required for decomposition-independent stochastic sources")
    noise_scheme = normalize_noise_scheme(noise_scheme)
    probe_count = vectors_per_base(noise_scheme, hp_num_vectors, multiplier_plan)
    if noise_scheme == "hierarchical_probing" and hp_ordering == NESTED_HP_ORDERING:
        validate_nested_hp_shape(latt_info.global_size, hp_num_vectors)
    if noise_scheme == "multiplier_probing":
        multiplier_plan = validate_multiplier_plan(multiplier_plan, latt_info.global_size)
    dilution = normalize_spin_color_dilution(spin_color_dilution)
    factor = spin_color_dilution_factor(dilution)
    base_idx, probe_start, probe_stop = int(base_idx), int(probe_start), int(probe_stop)
    source_offset = base_idx * probe_count * factor
    channels = tuple(product(range(4), range(3))) if dilution == "point" else ((-1, -1),)
    part_source_bookkeeping(
        base_idx, probe_start, probe_stop, probe_count, dilution
    )
    base_noise = make_counter_zn_noise_fermion(
        latt_info, int(config_num), base_idx,
        stream_seed=int(noise_stream), n=n_zn,
    )

    for probe_idx in range(probe_start, probe_stop):
        if noise_scheme == "hierarchical_probing":
            source = apply_hierarchical_probe(base_noise, probe_idx, hp_ordering)
        elif noise_scheme == "multiplier_probing":
            source = apply_multiplier_probe(base_noise, probe_idx, multiplier_plan)
        else:
            source = base_noise.copy()
        for spin_idx, color_idx in channels:
            selected = (
                apply_spin_color_point_dilution(source, spin_idx, color_idx)
                if dilution == "point" else source
            )
            channel = spin_idx * 3 + color_idx if dilution == "point" else 0
            fields = (
                source_offset + probe_idx * factor + channel,
                base_idx,
                probe_idx,
                selected,
                spin_idx,
                color_idx,
            )
            yield fields


def _lock_sample_log(handle):
    """Lock a sample log on local filesystems or Lustre/DVS."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return "flock"
    except OSError as error:
        # Perlmutter DVS returns its internal errno 524 for unsupported flock.
        if error.errno not in {errno.ENOSYS, errno.EOPNOTSUPP, 524}:
            raise
        fcntl.lockf(handle.fileno(), fcntl.LOCK_EX)
        return "lockf"


def _unlock_sample_log(handle, lock_kind):
    if lock_kind == "flock":
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    else:
        fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)


def selected_base_range(n_base_noise, base_start=0, base_stop=None):
    n_base_noise = int(n_base_noise)
    start = int(base_start)
    stop = n_base_noise if base_stop is None else int(base_stop)
    if n_base_noise <= 0 or start < 0 or stop > n_base_noise or start >= stop:
        raise ValueError(
            f"invalid base range [{start}, {stop}) for n_base_noise={n_base_noise}"
        )
    return range(start, stop)


def vectors_per_base(noise_scheme, hp_num_vectors, multiplier_plan=None):
    scheme = normalize_noise_scheme(noise_scheme)
    if scheme == "multiplier_probing":
        return validate_multiplier_plan(multiplier_plan)["colors"]
    if scheme == "hierarchical_probing":
        validate_hierarchical_probing_options(hp_num_vectors, NESTED_HP_ORDERING)
        return int(hp_num_vectors)
    return 1


def base_part_ranges(probe_count, block_interval_solves, solve_multiplier=1):
    probe_count = int(probe_count)
    interval = int(block_interval_solves)
    solve_multiplier = int(solve_multiplier)
    if probe_count <= 0 or interval <= 0 or solve_multiplier <= 0:
        raise ValueError(
            "probe_count, block_interval_solves, and solve_multiplier should be positive, "
            f"got {probe_count}, {interval}, {solve_multiplier}"
        )
    if interval % solve_multiplier:
        raise ValueError(
            "block_interval_solves should contain complete dilution bases; "
            f"got {interval} solves for multiplier {solve_multiplier}"
        )
    vector_interval = interval // solve_multiplier
    return [
        (part_index, probe_start, min(probe_start + vector_interval, probe_count))
        for part_index, probe_start in enumerate(range(0, probe_count, vector_interval))
    ]


def shard_part_path(shard_dir, canonical_tag, base_idx, part_idx, probe_start, probe_stop):
    stem = Path(canonical_tag).name
    name = (
        f"{stem}.base{int(base_idx):06d}.part{int(part_idx):04d}"
        f".probe{int(probe_start):04d}-{int(probe_stop) - 1:04d}.h5"
    )
    return Path(shard_dir) / name


def disconnected_sample_log_path(data_dir, canonical_tag):
    return (
        Path(data_dir)
        / "sample_log_disconnected"
        / f"{Path(canonical_tag).name}.log"
    )


def _write_attrs(obj, attrs):
    for key, value in (attrs or {}).items():
        if value is not None:
            obj.attrs[key] = value


def _temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + f".tmp.{os.getpid()}.{uuid4().hex}")


def write_shard_part_hdf5(
    path,
    attrs,
    metadata_datasets=None,
    raw_datasets=None,
    source_bookkeeping=None,
    shard_mean_datasets=None,
):
    """Atomically replace one raw, mean, or combined shard part."""
    if raw_datasets is None and shard_mean_datasets is None:
        raise ValueError("a shard part should contain raw or shard-mean datasets")
    if raw_datasets is not None and source_bookkeeping is None:
        raise ValueError("raw shard datasets require source bookkeeping")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    with h5py.File(tmp, "w") as h5:
        _write_attrs(h5, attrs)
        h5.attrs["raw_per_vector_stored"] = raw_datasets is not None
        h5.attrs["shard_mean_stored"] = shard_mean_datasets is not None
        if raw_datasets is not None:
            raw = h5.require_group("raw")
            for name, values in raw_datasets.items():
                raw.create_dataset(name, data=values)
            for name, values in source_bookkeeping.items():
                raw.create_dataset(name, data=np.asarray(values, dtype=np.int32))
        if shard_mean_datasets is not None:
            shard_mean = h5.require_group("shard_mean")
            for name, values in shard_mean_datasets.items():
                shard_mean.create_dataset(name, data=values)
        for name, values in (metadata_datasets or {}).items():
            h5.create_dataset(name, data=values)
        h5.flush()
    os.replace(tmp, path)


def write_raw_part_hdf5(
    path, raw_datasets, attrs, source_bookkeeping, metadata_datasets=None
):
    """Atomically replace one raw shard part after its payload is closed."""
    write_shard_part_hdf5(
        path,
        attrs,
        metadata_datasets=metadata_datasets,
        raw_datasets=raw_datasets,
        source_bookkeeping=source_bookkeeping,
    )


def shard_part_attrs(
    base_attrs,
    base_idx,
    part_idx,
    probe_start,
    probe_stop,
    probe_count,
    spin_color_dilution="none",
):
    dilution = normalize_spin_color_dilution(spin_color_dilution)
    attrs = dict(base_attrs)
    attrs.update(
        {
            "shard_schema": SHARD_SCHEMA,
            "base_noise_index": int(base_idx),
            "part_index": int(part_idx),
            "probe_start": int(probe_start),
            "probe_stop_exclusive": int(probe_stop),
            "vectors_per_base": int(probe_count),
            "spin_color_dilution": dilution,
            "source_bookkeeping_schema": (
                SOURCE_BOOKKEEPING_DILUTED_SCHEMA
                if dilution == "point" else SOURCE_BOOKKEEPING_SCHEMA
            ),
        }
    )
    return attrs


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items())
            if str(key) not in _FINGERPRINT_EXCLUDED
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sample_log_fingerprint(attrs, metadata=None):
    attrs = dict(attrs or {})
    dilution = normalize_spin_color_dilution(
        attrs.get("spin_color_dilution", "none")
    )
    identity = {
        "source_bookkeeping_schema": (
            SOURCE_BOOKKEEPING_DILUTED_SCHEMA
            if dilution == "point" else SOURCE_BOOKKEEPING_SCHEMA
        ),
        "attrs": _jsonable(attrs),
        "metadata": _jsonable(dict(metadata or {})),
    }
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_log_header(canonical_tag, attrs, metadata=None):
    return (
        f"# {SAMPLE_LOG_SCHEMA} sha256={sample_log_fingerprint(attrs, metadata)} "
        f"canonical={Path(canonical_tag).name}"
    )


def _read_locked_sample_log(handle, expected_header, path):
    handle.seek(0)
    lines = [line.strip() for line in handle.read().splitlines() if line.strip()]
    if not lines:
        handle.seek(0)
        handle.write(expected_header + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return set()
    if lines[0] != expected_header:
        raise ValueError(
            f"sample log header mismatch for {path}; use a new output identity or clear the old log"
        )
    completed = set()
    for line in lines[1:]:
        match = _BASE_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid sample log entry {line!r} in {path}")
        completed.add(int(match.group(1)))
    return completed


def prepare_sample_log(path, canonical_tag, attrs, metadata=None):
    """Return completed bases using only the text log, never shard HDF5 state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_header = sample_log_header(canonical_tag, attrs, metadata)
    with path.open("a+", encoding="utf-8") as handle:
        lock_kind = _lock_sample_log(handle)
        try:
            return _read_locked_sample_log(handle, expected_header, path)
        finally:
            _unlock_sample_log(handle, lock_kind)


def append_completed_base(path, canonical_tag, attrs, base_idx, metadata=None):
    """Durably append one exact base-completion line, suppressing duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_header = sample_log_header(canonical_tag, attrs, metadata)
    base_idx = int(base_idx)
    line = f"base{base_idx:06d}"
    with path.open("a+", encoding="utf-8") as handle:
        lock_kind = _lock_sample_log(handle)
        try:
            completed = _read_locked_sample_log(handle, expected_header, path)
            if base_idx in completed:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True
        finally:
            _unlock_sample_log(handle, lock_kind)


def _attr_equal(actual, expected):
    left, right = np.asarray(actual), np.asarray(expected)
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        try:
            return np.array_equal(left, right, equal_nan=True)
        except TypeError:
            return bool(
                np.all((left == right) | (np.isnan(left) & np.isnan(right)))
            )
    return np.array_equal(left, right)


def discover_shard_layout(
    shard_dir,
    canonical_tag,
    n_base_noise,
    raw_dataset_names,
    metadata_dataset_names=(),
    base_means=False,
):
    """Inspect base-0 part-0 and construct the expected streaming layout."""
    shard_dir = Path(shard_dir)
    n_base_noise = int(n_base_noise)
    if n_base_noise <= 0:
        raise ValueError(f"n_base_noise should be positive, got {n_base_noise}")
    raw_dataset_names = tuple(raw_dataset_names)
    metadata_dataset_names = tuple(metadata_dataset_names)
    if not raw_dataset_names:
        raise ValueError("raw_dataset_names should not be empty")

    stem = Path(canonical_tag).name
    first_matches = sorted(shard_dir.glob(f"{stem}.base000000.part0000.probe*.h5"))
    if len(first_matches) != 1:
        raise ValueError(
            f"expected exactly one base-0 part-0 shard for {stem}, found {len(first_matches)}"
        )
    first_path = first_matches[0]
    try:
        with h5py.File(first_path, "r") as first:
            if first.attrs.get("shard_schema") != SHARD_SCHEMA:
                raise ValueError(f"shard {first_path} has incompatible schema")
            if not base_means and not bool(first.attrs["raw_per_vector_stored"]):
                raise ValueError(
                    "mean-only shards cannot be finalized as per-vector raw data"
                )
            probe_count = int(first.attrs["vectors_per_base"])
            interval = int(first.attrs["block_interval_solves"])
            dilution = normalize_spin_color_dilution(first.attrs["spin_color_dilution"])
            solve_multiplier = spin_color_dilution_factor(dilution)
            expected_bookkeeping_schema = (
                SOURCE_BOOKKEEPING_DILUTED_SCHEMA
                if dilution == "point" else SOURCE_BOOKKEEPING_SCHEMA
            )
            if first.attrs.get("source_bookkeeping_schema") != expected_bookkeeping_schema:
                raise ValueError(f"shard {first_path} has incompatible bookkeeping schema")
            if "raw/source_index" in first:
                raise ValueError(f"shard {first_path} contains obsolete raw/source_index")
            reference_attrs = {
                key: first.attrs[key]
                for key in first.attrs
                if key not in _PART_ATTRS
            }
            reference_metadata = {}
            for name in metadata_dataset_names:
                if name not in first:
                    raise ValueError(
                        f"shard {first_path} is missing metadata dataset {name}"
                    )
                reference_metadata[name] = first[name][()]
            raw_tails = {}
            for name in raw_dataset_names:
                dataset = f"raw/{name}"
                mean_dataset = f"shard_mean/{name.removesuffix('_pervec')}"
                if base_means:
                    if not first.attrs["shard_mean_stored"] or mean_dataset not in first:
                        raise ValueError(f"shard {first_path} requires {mean_dataset}")
                    raw_tails[name] = tuple(first[mean_dataset].shape)
                    continue
                if dataset not in first:
                    raise ValueError(f"shard {first_path} is missing {dataset}")
                raw_tails[name] = tuple(first[dataset].shape[1:])
    except OSError as err:
        raise ValueError(f"cannot open shard part {first_path}: {err}") from err

    expected_first = shard_part_path(
        shard_dir,
        canonical_tag,
        0,
        0,
        0,
        min(interval // solve_multiplier, probe_count),
    )
    if first_path != expected_first:
        raise ValueError(
            f"base-0 part-0 shard has unexpected HP interval: {first_path}"
        )

    parts = []
    output_offset = 0
    for base_idx in range(n_base_noise):
        for part_idx, probe_start, probe_stop in base_part_ranges(
            probe_count, interval, solve_multiplier
        ):
            bookkeeping = part_source_bookkeeping(
                base_idx, probe_start, probe_stop, probe_count, dilution
            )
            vector_count = probe_stop - probe_start
            source_count = vector_count * solve_multiplier
            parts.append(
                {
                    "base_idx": base_idx,
                    "part_idx": part_idx,
                    "probe_start": probe_start,
                    "probe_stop": probe_stop,
                    "path": shard_part_path(
                        shard_dir,
                        canonical_tag,
                        base_idx,
                        part_idx,
                        probe_start,
                        probe_stop,
                    ),
                    "output_start": output_offset,
                    "output_stop": output_offset + source_count,
                    "vector_count": vector_count,
                    "source_count": source_count,
                    "bookkeeping": bookkeeping,
                }
            )
            output_offset += source_count

    for base_idx in range(n_base_noise):
        expected = {info["path"] for info in parts if info["base_idx"] == base_idx}
        actual = set(
            shard_dir.glob(f"{stem}.base{base_idx:06d}.part*.probe*.h5")
        )
        if actual != expected:
            missing = sorted(str(path) for path in expected - actual)
            unexpected = sorted(str(path) for path in actual - expected)
            raise ValueError(
                f"shard filename coverage mismatch for base {base_idx}: "
                f"missing={missing}, unexpected={unexpected}"
            )

    return {
        "base_means": base_means,
        "reference_attrs": reference_attrs,
        "metadata": reference_metadata,
        "raw_tails": raw_tails,
        "parts": parts,
        "probe_count": probe_count,
        "spin_color_dilution": dilution,
        "solve_multiplier": solve_multiplier,
        "block_interval_solves": interval,
        "total_sources": output_offset,
        "raw_dataset_names": raw_dataset_names,
        "metadata_dataset_names": metadata_dataset_names,
    }


def iter_validated_shard_parts(layout):
    """Open, validate, and yield each expected part once during finalization."""
    reference_attrs = layout["reference_attrs"]
    reference_metadata = layout["metadata"]
    raw_tails = layout["raw_tails"]
    for info in layout["parts"]:
        path = info["path"]
        try:
            with h5py.File(path, "r") as part:
                part_fields = {
                    "shard_schema": SHARD_SCHEMA,
                    "base_noise_index": info["base_idx"],
                    "part_index": info["part_idx"],
                    "probe_start": info["probe_start"],
                    "probe_stop_exclusive": info["probe_stop"],
                    "vectors_per_base": layout["probe_count"],
                    "block_interval_solves": layout["block_interval_solves"],
                    "source_bookkeeping_schema": (
                        SOURCE_BOOKKEEPING_DILUTED_SCHEMA
                        if layout["spin_color_dilution"] == "point"
                        else SOURCE_BOOKKEEPING_SCHEMA
                    ),
                }
                base_means = layout["base_means"]
                has_raw = bool(part.attrs["raw_per_vector_stored"])
                for key, expected in part_fields.items():
                    if key not in part.attrs or not _attr_equal(
                        part.attrs[key], expected
                    ):
                        raise ValueError(
                            f"shard {path} has incompatible attribute {key}"
                        )
                common = {
                    key: part.attrs[key]
                    for key in part.attrs
                    if key not in _PART_ATTRS
                }
                if set(common) != set(reference_attrs):
                    raise ValueError(f"shard {path} has incompatible attribute set")
                for key, expected in reference_attrs.items():
                    if not _attr_equal(common[key], expected):
                        raise ValueError(
                            f"shard {path} has incompatible attribute {key}"
                        )
                for name, expected in reference_metadata.items():
                    if name not in part or not np.array_equal(part[name][()], expected):
                        raise ValueError(
                            f"shard {path} has incompatible metadata {name}"
                        )
                source_count = info["source_count"]
                vector_count = info["vector_count"]
                if base_means:
                    if part.attrs.get("shard_mean_vector_count") != vector_count:
                        raise ValueError(f"shard {path} has incompatible mean count")
                    for name, tail in raw_tails.items():
                        dataset = f"shard_mean/{name.removesuffix('_pervec')}"
                        if dataset not in part or part[dataset].shape != tail:
                            raise ValueError(f"shard {path} has incompatible {dataset} shape")
                        if not np.all(np.isfinite(part[dataset][()])):
                            raise ValueError(f"shard {path} contains non-finite means")
                for name, tail in raw_tails.items():
                    if base_means and not has_raw:
                        continue
                    dataset = f"raw/{name}"
                    if dataset not in part or tuple(part[dataset].shape) != (
                        source_count,
                    ) + tail:
                        raise ValueError(
                            f"shard {path} has incompatible {dataset} shape"
                        )
                for name, expected in info["bookkeeping"].items():
                    if base_means and not has_raw:
                        continue
                    dataset = f"raw/{name}"
                    if dataset not in part or not np.array_equal(
                        part[dataset][()], expected
                    ):
                        raise ValueError(f"shard {path} has incompatible {dataset}")
                yield info, part
        except OSError as err:
            raise ValueError(f"cannot open shard part {path}: {err}") from err


def canonical_temp_path(tag):
    path = Path(str(tag) + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, _temporary_path(path)


__all__ = [
    "NESTED_HP_ORDERING",
    "COMPLETE_HP_LEVELS",
    "hierarchical_nested_index",
    "validate_nested_hp_shape",
    "validate_multiplier_plan",
    "broadcast_multiplier_plan",
    "multiplier_probe_pattern",
    "apply_multiplier_probe",
    "COUNTER_NOISE_ALGORITHM",
    "SAMPLE_LOG_SCHEMA",
    "SHARD_SCHEMA",
    "SOURCE_BOOKKEEPING_SCHEMA",
    "SOURCE_BOOKKEEPING_DILUTED_SCHEMA",
    "SPIN_COLOR_POINT_FACTOR",
    "VALID_HP_ORDERINGS",
    "VALID_NOISE_SCHEMES",
    "append_completed_base",
    "apply_hierarchical_probe",
    "apply_spin_color_point_dilution",
    "base_part_ranges",
    "canonical_temp_path",
    "counter_zn_phase_indices",
    "disconnected_sample_log_path",
    "discover_shard_layout",
    "effective_n_inversions",
    "hierarchical_probe_pattern",
    "vectors_per_base",
    "iter_noise_base_probe_interval",
    "iter_validated_shard_parts",
    "make_counter_zn_noise_fermion",
    "normalize_noise_scheme",
    "normalize_spin_color_dilution",
    "part_source_bookkeeping",
    "prepare_sample_log",
    "reconstruct_source_indices",
    "sample_log_fingerprint",
    "sample_log_header",
    "selected_base_range",
    "shard_part_attrs",
    "shard_part_path",
    "spin_color_dilution_factor",
    "validate_hierarchical_probing_options",
    "write_raw_part_hdf5",
    "write_shard_part_hdf5",
]
