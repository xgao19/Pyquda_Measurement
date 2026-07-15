"""Lightweight base-log and shard helpers for disconnected one-point measurements."""

import fcntl
import hashlib
import json
import os
import re
import errno
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np


SHARD_SCHEMA = "disconnected_1pt_base_parts_v2"
SAMPLE_LOG_SCHEMA = "disconnected_sample_log_v1"
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
}


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
        raise ValueError(f"invalid base range [{start}, {stop}) for n_base_noise={n_base_noise}")
    return range(start, stop)


def hp_vectors_per_base(noise_scheme, hp_num_vectors):
    return int(hp_num_vectors) if str(noise_scheme).strip().lower() == "hierarchical_probing" else 1


def base_part_ranges(hp_count, block_interval_solves, solves_per_hp=1):
    hp_count = int(hp_count)
    interval = int(block_interval_solves)
    solves_per_hp = int(solves_per_hp)
    if hp_count <= 0 or interval <= 0 or solves_per_hp <= 0:
        raise ValueError(
            "hp_count, block_interval_solves, and solves_per_hp should be positive, "
            f"got {hp_count}, {interval}, {solves_per_hp}"
        )
    if interval < solves_per_hp:
        raise ValueError(
            f"block_interval_solves={interval} cannot hold one complete HP vector "
            f"requiring {solves_per_hp} solves"
        )
    hp_per_part = interval // solves_per_hp
    return [
        (part_index, hp_start, min(hp_start + hp_per_part, hp_count))
        for part_index, hp_start in enumerate(range(0, hp_count, hp_per_part))
    ]


def shard_part_path(shard_dir, canonical_tag, base_idx, part_idx, hp_start, hp_stop):
    stem = Path(canonical_tag).name
    name = (
        f"{stem}.base{int(base_idx):06d}.part{int(part_idx):04d}"
        f".hp{int(hp_start):04d}-{int(hp_stop) - 1:04d}.h5"
    )
    return Path(shard_dir) / name


def disconnected_sample_log_path(data_dir, canonical_tag):
    return Path(data_dir) / "sample_log_disconnected" / f"{Path(canonical_tag).name}.log"


def _write_attrs(obj, attrs):
    for key, value in (attrs or {}).items():
        if value is not None:
            obj.attrs[key] = value


def _temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + f".tmp.{os.getpid()}.{uuid4().hex}")


def write_raw_part_hdf5(path, raw_datasets, attrs, source_bookkeeping, metadata_datasets=None):
    """Atomically replace one shard part after its complete HDF5 payload is closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    with h5py.File(tmp, "w") as h5:
        _write_attrs(h5, attrs)
        raw = h5.require_group("raw")
        for name, values in raw_datasets.items():
            raw.create_dataset(name, data=values)
        for name, values in source_bookkeeping.items():
            raw.create_dataset(name, data=np.asarray(values, dtype=np.int32))
        for name, values in (metadata_datasets or {}).items():
            h5.create_dataset(name, data=values)
        h5.flush()
    os.replace(tmp, path)


def shard_part_attrs(
    base_attrs, base_idx, part_idx, hp_start, hp_stop, hp_count,
    solves_per_hp=1, spin_color_dilution="none",
):
    attrs = dict(base_attrs)
    attrs.update({
        "shard_schema": SHARD_SCHEMA,
        "base_noise_index": int(base_idx),
        "part_index": int(part_idx),
        "hp_start": int(hp_start),
        "hp_stop_exclusive": int(hp_stop),
        "hp_vectors_per_base": int(hp_count),
        "solves_per_hp": int(solves_per_hp),
        "spin_color_dilution": str(spin_color_dilution),
    })
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
        "attrs": _jsonable(dict(attrs or {})),
        "metadata": _jsonable(dict(metadata or {})),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
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
            return bool(np.all((left == right) | (np.isnan(left) & np.isnan(right))))
    return np.array_equal(left, right)


def discover_shard_layout(
    shard_dir,
    canonical_tag,
    n_base_noise,
    raw_dataset_names,
    metadata_dataset_names=(),
    include_spin_color=False,
):
    """Inspect only base-0 part-0 and construct the expected streaming layout."""
    from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
        part_source_bookkeeping,
    )

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
            hp_count = int(first.attrs["hp_vectors_per_base"])
            interval = int(first.attrs["block_interval_solves"])
            solves_per_hp = int(first.attrs["solves_per_hp"])
            spin_color_dilution = str(first.attrs["spin_color_dilution"])
            reference_attrs = {
                key: first.attrs[key] for key in first.attrs if key not in _PART_ATTRS
            }
            reference_metadata = {}
            for name in metadata_dataset_names:
                if name not in first:
                    raise ValueError(f"shard {first_path} is missing metadata dataset {name}")
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
        shard_dir, canonical_tag, 0, 0, 0,
        min(interval // solves_per_hp, hp_count),
    )
    if first_path != expected_first:
        raise ValueError(f"base-0 part-0 shard has unexpected HP interval: {first_path}")

    parts = []
    output_offset = 0
    for base_idx in range(n_base_noise):
        for part_idx, hp_start, hp_stop in base_part_ranges(
            hp_count, interval, solves_per_hp
        ):
            bookkeeping = part_source_bookkeeping(
                base_idx, hp_start, hp_stop, hp_count, spin_color_dilution,
                include_spin_color=include_spin_color,
            )
            count = len(bookkeeping["source_index"])
            parts.append({
                "base_idx": base_idx,
                "part_idx": part_idx,
                "hp_start": hp_start,
                "hp_stop": hp_stop,
                "path": shard_part_path(
                    shard_dir, canonical_tag, base_idx, part_idx, hp_start, hp_stop
                ),
                "output_start": output_offset,
                "output_stop": output_offset + count,
                "bookkeeping": bookkeeping,
            })
            output_offset += count

    for base_idx in range(n_base_noise):
        expected = {
            info["path"] for info in parts if info["base_idx"] == base_idx
        }
        actual = set(shard_dir.glob(f"{stem}.base{base_idx:06d}.part*.hp*.h5"))
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
        "solves_per_hp": solves_per_hp,
        "spin_color_dilution": spin_color_dilution,
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
                    "solves_per_hp": layout["solves_per_hp"],
                    "spin_color_dilution": layout["spin_color_dilution"],
                }
                for key, expected in part_fields.items():
                    if key not in part.attrs or not _attr_equal(part.attrs[key], expected):
                        raise ValueError(f"shard {path} has incompatible attribute {key}")
                common = {
                    key: part.attrs[key] for key in part.attrs if key not in _PART_ATTRS
                }
                if set(common) != set(reference_attrs):
                    raise ValueError(f"shard {path} has incompatible attribute set")
                for key, expected in reference_attrs.items():
                    if not _attr_equal(common[key], expected):
                        raise ValueError(f"shard {path} has incompatible attribute {key}")
                for name, expected in reference_metadata.items():
                    if name not in part or not np.array_equal(part[name][()], expected):
                        raise ValueError(f"shard {path} has incompatible metadata {name}")
                count = info["output_stop"] - info["output_start"]
                for name, tail in raw_tails.items():
                    dataset = f"raw/{name}"
                    if dataset not in part or tuple(part[dataset].shape) != (count,) + tail:
                        raise ValueError(f"shard {path} has incompatible {dataset} shape")
                for name, expected in info["bookkeeping"].items():
                    dataset = f"raw/{name}"
                    if dataset not in part or not np.array_equal(part[dataset][()], expected):
                        raise ValueError(f"shard {path} has incompatible {dataset}")
                yield info, part
        except OSError as err:
            raise ValueError(f"cannot open shard part {path}: {err}") from err


def canonical_temp_path(tag):
    path = Path(str(tag) + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, _temporary_path(path)


__all__ = [
    "SAMPLE_LOG_SCHEMA",
    "SHARD_SCHEMA",
    "append_completed_base",
    "base_part_ranges",
    "canonical_temp_path",
    "disconnected_sample_log_path",
    "discover_shard_layout",
    "hp_vectors_per_base",
    "iter_validated_shard_parts",
    "prepare_sample_log",
    "sample_log_fingerprint",
    "sample_log_header",
    "selected_base_range",
    "shard_part_attrs",
    "shard_part_path",
    "write_raw_part_hdf5",
]
