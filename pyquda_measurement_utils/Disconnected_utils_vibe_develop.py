"""Shared noise, probing, shard, and resume helpers for disconnected 1pt."""

import errno
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


VALID_NOISE_SCHEMES = {"zn", "hierarchical_probing"}
COUNTER_NOISE_ALGORITHM = "splitmix64_global_coordinate_v1"
SHARD_SCHEMA = "disconnected_1pt_base_parts_v3"
SAMPLE_LOG_SCHEMA = "disconnected_sample_log_v1"
SOURCE_BOOKKEEPING_SCHEMA = "base_hp_v1"
VALID_HP_ORDERINGS = {
    "global_xyzt_gray_projected_to_evenodd",
    "interleaved_xyzt_binary_projected_to_evenodd",
    "interleaved_xyz_binary_projected_to_evenodd",
    "spatial_xyz_then_t_gray_projected_to_evenodd",
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
    "hp_start",
    "hp_stop_exclusive",
    "shard_mean_vector_count",
}


def normalize_noise_scheme(noise_scheme: str) -> str:
    """Normalize and validate the stochastic noise scheme name."""
    scheme = str(noise_scheme).strip().lower()
    if scheme not in VALID_NOISE_SCHEMES:
        raise ValueError(f"noise_scheme should be one of {VALID_NOISE_SCHEMES}, got {noise_scheme!r}")
    return scheme


def is_power_of_two(value: int) -> bool:
    """Return whether value is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)) for positive integers, with ceil_log2(1)=0."""
    if value <= 1:
        return 0
    return int(value - 1).bit_length()


def validate_hierarchical_probing_options(hp_num_vectors: int, hp_ordering: str):
    """Validate common hierarchical probing options."""
    if hp_ordering not in VALID_HP_ORDERINGS:
        raise ValueError(f"hp_ordering should be one of {VALID_HP_ORDERINGS}, got {hp_ordering!r}")
    if not is_power_of_two(hp_num_vectors):
        raise ValueError(f"hp_num_vectors should be a positive power of two, got {hp_num_vectors}")


def effective_n_inversions(n_vec: int, noise_scheme: str, hp_num_vectors: int) -> int:
    """Return the number of effective stochastic inversions."""
    hp_factor = hp_num_vectors if noise_scheme == "hierarchical_probing" else 1
    return n_vec * hp_factor


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


def hierarchical_gray_index(latt_info, hp_ordering: str):
    """Return the Gray-code site index for a hierarchical probing ordering."""
    coords = latt_info.coordinate()
    x, y, z, t = [np.asarray(coords[mu], dtype=np.int64) for mu in range(4)]
    Gx, Gy, Gz, Gt = latt_info.global_size

    if hp_ordering == "global_xyzt_gray_projected_to_evenodd":
        site_id = x + Gx * (y + Gy * (z + Gz * t))
        return site_id ^ (site_id >> 1)

    if hp_ordering == "interleaved_xyzt_binary_projected_to_evenodd":
        site_id = np.zeros_like(x, dtype=np.int64)
        out_bit = 0
        for bit in range(max(ceil_log2(Gx), ceil_log2(Gy), ceil_log2(Gz), ceil_log2(Gt))):
            for coord, extent in ((x, Gx), (y, Gy), (z, Gz), (t, Gt)):
                if bit < ceil_log2(extent):
                    site_id |= ((coord >> bit) & 1) << out_bit
                    out_bit += 1
        return site_id

    if hp_ordering == "interleaved_xyz_binary_projected_to_evenodd":
        site_id = np.zeros_like(x, dtype=np.int64)
        out_bit = 0
        for bit in range(max(ceil_log2(Gx), ceil_log2(Gy), ceil_log2(Gz))):
            for coord, extent in ((x, Gx), (y, Gy), (z, Gz)):
                if bit < ceil_log2(extent):
                    site_id |= ((coord >> bit) & 1) << out_bit
                    out_bit += 1
        return site_id

    if hp_ordering == "spatial_xyz_then_t_gray_projected_to_evenodd":
        spatial_id = x + Gx * (y + Gy * z)
        spatial_gray = spatial_id ^ (spatial_id >> 1)
        time_gray = t ^ (t >> 1)
        spatial_bits = ceil_log2(Gx * Gy * Gz)
        return spatial_gray | (time_gray << spatial_bits)

    raise ValueError(f"Unsupported hp_ordering {hp_ordering!r}")


def hierarchical_probe_pattern(latt_info, hp_idx: int, hp_ordering: str):
    """Build a site-only Rademacher probing vector in even-odd layout."""
    if hp_idx == 0:
        return np.ones_like(latt_info.coordinate(0), dtype=np.float64)

    gray_id = hierarchical_gray_index(latt_info, hp_ordering)
    parity = np.zeros_like(gray_id, dtype=bool)
    mask = int(hp_idx)
    bit = 1
    while bit <= mask:
        if mask & bit:
            parity ^= (gray_id & bit) != 0
        bit <<= 1
    return np.where(parity, -1.0, 1.0)


def apply_hierarchical_probe(xi, hp_idx: int, hp_ordering: str):
    """Multiply a base stochastic source by one hierarchical probing vector."""
    if hp_idx == 0:
        return xi.copy()

    probed = xi.copy()
    xp = _get_xp_from_array(xi.data)
    pattern = hierarchical_probe_pattern(xi.latt_info, hp_idx, hp_ordering)
    pattern = _asarray_on_queue(pattern, xp, xi.data)
    probed.data[:] *= pattern[..., None, None]
    return probed


def part_source_bookkeeping(
    base_idx: int,
    hp_start: int,
    hp_stop: int,
    hp_count: int,
):
    """Return deterministic global indices for one base/HP shard part."""
    base_idx = int(base_idx)
    hp_start = int(hp_start)
    hp_stop = int(hp_stop)
    hp_count = int(hp_count)
    if base_idx < 0 or hp_count <= 0 or not 0 <= hp_start < hp_stop <= hp_count:
        raise ValueError(
            f"invalid base/HP interval: base={base_idx}, hp=[{hp_start}, {hp_stop}), "
            f"hp_count={hp_count}"
        )

    base_indices = []
    hp_indices = []
    for hp_idx in range(hp_start, hp_stop):
        base_indices.append(base_idx)
        hp_indices.append(hp_idx)

    bookkeeping = {
        "base_noise_index": np.asarray(base_indices, dtype=np.int32),
        "hp_index": np.asarray(hp_indices, dtype=np.int32),
    }
    return bookkeeping


def reconstruct_source_indices(base_noise_index, hp_index, hp_count):
    """Reconstruct the effective source index from base and HP indices."""
    base = np.asarray(base_noise_index, dtype=np.int64)
    hp = np.asarray(hp_index, dtype=np.int64)
    hp_count = int(hp_count)
    if base.shape != hp.shape:
        raise ValueError("base_noise_index and hp_index should have the same shape")
    if hp_count <= 0 or np.any(base < 0) or np.any(hp < 0) or np.any(hp >= hp_count):
        raise ValueError("invalid base/HP bookkeeping")
    return (base * hp_count + hp).astype(np.int64, copy=False)


def iter_noise_base_hp_interval(
    latt_info,
    base_idx: int,
    hp_start: int,
    hp_stop: int,
    n_zn: int,
    noise_scheme: str,
    hp_num_vectors: int,
    hp_ordering: str,
    config_num: int,
    noise_stream: int = 0,
):
    """Yield only the requested HP interval of one deterministic base noise."""
    if config_num is None:
        raise ValueError("config_num is required for decomposition-independent stochastic sources")
    noise_scheme = normalize_noise_scheme(noise_scheme)
    hp_count = int(hp_num_vectors) if noise_scheme == "hierarchical_probing" else 1
    base_noise = make_counter_zn_noise_fermion(
        latt_info, int(config_num), int(base_idx),
        stream_seed=int(noise_stream), n=n_zn,
    )

    for hp_idx in range(int(hp_start), int(hp_stop)):
        source = (
            apply_hierarchical_probe(base_noise, hp_idx, hp_ordering)
            if noise_scheme == "hierarchical_probing"
            else base_noise.copy()
        )
        fields = (
            int(base_idx) * hp_count + int(hp_idx),
            int(base_idx),
            int(hp_idx),
            source,
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


def hp_vectors_per_base(noise_scheme, hp_num_vectors):
    return (
        int(hp_num_vectors)
        if str(noise_scheme).strip().lower() == "hierarchical_probing"
        else 1
    )


def base_part_ranges(hp_count, block_interval_solves):
    hp_count = int(hp_count)
    interval = int(block_interval_solves)
    if hp_count <= 0 or interval <= 0:
        raise ValueError(
            "hp_count and block_interval_solves should be positive, "
            f"got {hp_count}, {interval}"
        )
    return [
        (part_index, hp_start, min(hp_start + interval, hp_count))
        for part_index, hp_start in enumerate(range(0, hp_count, interval))
    ]


def shard_part_path(shard_dir, canonical_tag, base_idx, part_idx, hp_start, hp_stop):
    stem = Path(canonical_tag).name
    name = (
        f"{stem}.base{int(base_idx):06d}.part{int(part_idx):04d}"
        f".hp{int(hp_start):04d}-{int(hp_stop) - 1:04d}.h5"
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
    hp_start,
    hp_stop,
    hp_count,
):
    attrs = dict(base_attrs)
    attrs.update(
        {
            "shard_schema": SHARD_SCHEMA,
            "base_noise_index": int(base_idx),
            "part_index": int(part_idx),
            "hp_start": int(hp_start),
            "hp_stop_exclusive": int(hp_stop),
            "hp_vectors_per_base": int(hp_count),
            "source_bookkeeping_schema": SOURCE_BOOKKEEPING_SCHEMA,
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
    identity = {
        "source_bookkeeping_schema": SOURCE_BOOKKEEPING_SCHEMA,
        "attrs": _jsonable(dict(attrs or {})),
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
    first_matches = sorted(shard_dir.glob(f"{stem}.base000000.part0000.hp*.h5"))
    if len(first_matches) != 1:
        raise ValueError(
            f"expected exactly one base-0 part-0 shard for {stem}, found {len(first_matches)}"
        )
    first_path = first_matches[0]
    try:
        with h5py.File(first_path, "r") as first:
            if first.attrs.get("shard_schema") != SHARD_SCHEMA:
                raise ValueError(f"shard {first_path} has incompatible schema")
            if not bool(first.attrs.get("raw_per_vector_stored", True)):
                raise ValueError(
                    "mean-only shards cannot be finalized as per-vector raw data"
                )
            hp_count = int(first.attrs["hp_vectors_per_base"])
            interval = int(first.attrs["block_interval_solves"])
            if first.attrs.get("source_bookkeeping_schema") != SOURCE_BOOKKEEPING_SCHEMA:
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
        min(interval, hp_count),
    )
    if first_path != expected_first:
        raise ValueError(
            f"base-0 part-0 shard has unexpected HP interval: {first_path}"
        )

    parts = []
    output_offset = 0
    for base_idx in range(n_base_noise):
        for part_idx, hp_start, hp_stop in base_part_ranges(hp_count, interval):
            bookkeeping = part_source_bookkeeping(
                base_idx, hp_start, hp_stop, hp_count
            )
            count = hp_stop - hp_start
            parts.append(
                {
                    "base_idx": base_idx,
                    "part_idx": part_idx,
                    "hp_start": hp_start,
                    "hp_stop": hp_stop,
                    "path": shard_part_path(
                        shard_dir,
                        canonical_tag,
                        base_idx,
                        part_idx,
                        hp_start,
                        hp_stop,
                    ),
                    "output_start": output_offset,
                    "output_stop": output_offset + count,
                    "bookkeeping": bookkeeping,
                }
            )
            output_offset += count

    for base_idx in range(n_base_noise):
        expected = {info["path"] for info in parts if info["base_idx"] == base_idx}
        actual = set(
            shard_dir.glob(f"{stem}.base{base_idx:06d}.part*.hp*.h5")
        )
        if actual != expected:
            missing = sorted(str(path) for path in expected - actual)
            unexpected = sorted(str(path) for path in actual - expected)
            raise ValueError(
                f"shard filename coverage mismatch for base {base_idx}: "
                f"missing={missing}, unexpected={unexpected}"
            )

    return {
        "reference_attrs": reference_attrs,
        "metadata": reference_metadata,
        "raw_tails": raw_tails,
        "parts": parts,
        "hp_count": hp_count,
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
                    "hp_start": info["hp_start"],
                    "hp_stop_exclusive": info["hp_stop"],
                    "hp_vectors_per_base": layout["hp_count"],
                    "block_interval_solves": layout["block_interval_solves"],
                    "source_bookkeeping_schema": SOURCE_BOOKKEEPING_SCHEMA,
                }
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
                count = info["output_stop"] - info["output_start"]
                for name, tail in raw_tails.items():
                    dataset = f"raw/{name}"
                    if dataset not in part or tuple(part[dataset].shape) != (
                        count,
                    ) + tail:
                        raise ValueError(
                            f"shard {path} has incompatible {dataset} shape"
                        )
                for name, expected in info["bookkeeping"].items():
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
    "COUNTER_NOISE_ALGORITHM",
    "SAMPLE_LOG_SCHEMA",
    "SHARD_SCHEMA",
    "SOURCE_BOOKKEEPING_SCHEMA",
    "VALID_HP_ORDERINGS",
    "VALID_NOISE_SCHEMES",
    "append_completed_base",
    "apply_hierarchical_probe",
    "base_part_ranges",
    "canonical_temp_path",
    "ceil_log2",
    "counter_zn_phase_indices",
    "disconnected_sample_log_path",
    "discover_shard_layout",
    "effective_n_inversions",
    "hierarchical_gray_index",
    "hierarchical_probe_pattern",
    "hp_vectors_per_base",
    "is_power_of_two",
    "iter_noise_base_hp_interval",
    "iter_validated_shard_parts",
    "make_counter_zn_noise_fermion",
    "normalize_noise_scheme",
    "part_source_bookkeeping",
    "prepare_sample_log",
    "reconstruct_source_indices",
    "sample_log_fingerprint",
    "sample_log_header",
    "selected_base_range",
    "shard_part_attrs",
    "shard_part_path",
    "validate_hierarchical_probing_options",
    "write_raw_part_hdf5",
    "write_shard_part_hdf5",
]
