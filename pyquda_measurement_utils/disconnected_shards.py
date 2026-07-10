"""Base-oriented shard helpers for disconnected one-point measurements."""

import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np


SHARD_SCHEMA = "disconnected_1pt_base_parts_v1"


def normalize_output_mode(output_mode):
    mode = str(output_mode).strip().lower()
    if mode not in {"monolithic", "base_shards"}:
        raise ValueError(f"output_mode should be monolithic or base_shards, got {output_mode!r}")
    return mode


def selected_base_range(n_base_noise, base_start=0, base_stop=None):
    n_base_noise = int(n_base_noise)
    start = int(base_start)
    stop = n_base_noise if base_stop is None else int(base_stop)
    if n_base_noise <= 0 or start < 0 or stop > n_base_noise or start >= stop:
        raise ValueError(f"invalid base range [{start}, {stop}) for n_base_noise={n_base_noise}")
    return range(start, stop)


def hp_vectors_per_base(noise_scheme, hp_num_vectors):
    return int(hp_num_vectors) if str(noise_scheme).strip().lower() == "hierarchical_probing" else 1


def base_part_ranges(hp_count, block_interval_solves):
    hp_count = int(hp_count)
    interval = int(block_interval_solves)
    if hp_count <= 0 or interval <= 0:
        raise ValueError(f"hp_count and block_interval_solves should be positive, got {hp_count}, {interval}")
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


def base_completion_path(shard_dir, canonical_tag, base_idx):
    return Path(shard_dir) / f"{Path(canonical_tag).name}.base{int(base_idx):06d}.complete.json"


def _write_attrs(obj, attrs):
    for key, value in (attrs or {}).items():
        if value is not None:
            obj.attrs[key] = value


def _temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + f".tmp.{os.getpid()}.{uuid4().hex}")


def write_raw_part_hdf5(path, raw_datasets, attrs, source_bookkeeping, metadata_datasets=None):
    """Atomically write one measurement part with raw and metadata datasets."""
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


def _attr_equal(actual, expected):
    return np.array_equal(np.asarray(actual), np.asarray(expected))


def validate_raw_part_hdf5(path, expected_attrs, raw_shapes, metadata=None):
    """Validate one completed part and return without modifying it."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as h5:
            for key, value in expected_attrs.items():
                if key not in h5.attrs or not _attr_equal(h5.attrs[key], value):
                    raise ValueError(f"{path} has incompatible attribute {key}")
            if not bool(h5.attrs.get("part_complete", False)):
                raise ValueError(f"{path} is not marked part_complete")
            for name, shape in raw_shapes.items():
                dataset = f"raw/{name}"
                if dataset not in h5 or tuple(h5[dataset].shape) != tuple(shape):
                    raise ValueError(f"{path} has incompatible {dataset} shape")
            for name, expected in (metadata or {}).items():
                if name not in h5 or not np.array_equal(h5[name][()], np.asarray(expected)):
                    raise ValueError(f"{path} has incompatible metadata dataset {name}")
    except OSError as err:
        raise ValueError(f"cannot open shard part {path}: {err}") from err
    return True


def write_base_completion_marker(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_base_completion_marker(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"invalid base completion marker {path}: {err}") from err


def expected_part_attrs(base_attrs, base_idx, part_idx, hp_start, hp_stop, hp_count):
    attrs = dict(base_attrs)
    attrs.update({
        "shard_schema": SHARD_SCHEMA,
        "base_noise_index": int(base_idx),
        "part_index": int(part_idx),
        "hp_start": int(hp_start),
        "hp_stop_exclusive": int(hp_stop),
        "hp_vectors_per_base": int(hp_count),
        "part_complete": True,
    })
    return attrs


def completion_payload(canonical_tag, base_idx, hp_count, block_interval_solves, part_paths):
    return {
        "shard_schema": SHARD_SCHEMA,
        "canonical_stem": Path(canonical_tag).name,
        "base_noise_index": int(base_idx),
        "hp_vectors_per_base": int(hp_count),
        "block_interval_solves": int(block_interval_solves),
        "parts": [Path(path).name for path in part_paths],
    }


def canonical_temp_path(tag):
    path = Path(str(tag) + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, _temporary_path(path)


__all__ = [
    "SHARD_SCHEMA",
    "base_completion_path",
    "base_part_ranges",
    "canonical_temp_path",
    "completion_payload",
    "expected_part_attrs",
    "hp_vectors_per_base",
    "normalize_output_mode",
    "read_base_completion_marker",
    "selected_base_range",
    "shard_part_path",
    "validate_raw_part_hdf5",
    "write_base_completion_marker",
    "write_raw_part_hdf5",
]
