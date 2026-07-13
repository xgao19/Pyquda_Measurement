"""Base-oriented shard helpers for disconnected one-point measurements."""

import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np


SHARD_SCHEMA = "disconnected_1pt_base_parts_v1"


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


def validate_raw_part_hdf5(
    path, expected_attrs, raw_shapes, metadata=None, source_bookkeeping=None
):
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
            for name, expected in (source_bookkeeping or {}).items():
                dataset = f"raw/{name}"
                if dataset not in h5 or not np.array_equal(
                    h5[dataset][()], np.asarray(expected, dtype=np.int32)
                ):
                    raise ValueError(f"{path} has incompatible source bookkeeping {name}")
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


def completion_payload(
    canonical_tag, base_idx, hp_count, block_interval_solves, part_paths,
    solves_per_hp=1, spin_color_dilution="none",
):
    return {
        "shard_schema": SHARD_SCHEMA,
        "canonical_stem": Path(canonical_tag).name,
        "base_noise_index": int(base_idx),
        "hp_vectors_per_base": int(hp_count),
        "block_interval_solves": int(block_interval_solves),
        "solves_per_hp": int(solves_per_hp),
        "spin_color_dilution": str(spin_color_dilution),
        "parts": [Path(path).name for path in part_paths],
    }


def validate_complete_shard_set(
    shard_dir,
    canonical_tag,
    n_base_noise,
    raw_dataset_names,
    metadata_dataset_names=(),
    spin_color_dilution="none",
    solves_per_hp=1,
    include_spin_color=False,
):
    """Validate all base parts and return a common streaming manifest."""
    from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
        part_source_bookkeeping,
    )

    shard_dir = Path(shard_dir)
    n_base_noise = int(n_base_noise)
    solves_per_hp = int(solves_per_hp)
    if n_base_noise <= 0:
        raise ValueError(f"n_base_noise should be positive, got {n_base_noise}")
    raw_dataset_names = tuple(raw_dataset_names)
    metadata_dataset_names = tuple(metadata_dataset_names)
    if not raw_dataset_names:
        raise ValueError("raw_dataset_names should not be empty")

    reference_attrs = None
    reference_metadata = None
    raw_tails = None
    parts = []
    hp_count = None
    interval = None
    output_offset = 0
    excluded = {
        "base_noise_index", "part_index", "hp_start", "hp_stop_exclusive",
        "part_complete", "configured_n_base_noise",
    }

    for base_idx in range(n_base_noise):
        marker_path = base_completion_path(shard_dir, canonical_tag, base_idx)
        marker = read_base_completion_marker(marker_path)
        if marker is None:
            raise ValueError(f"missing completion marker for base {base_idx}: {marker_path}")
        if marker.get("shard_schema") != SHARD_SCHEMA:
            raise ValueError(f"completion marker has incompatible schema: {marker_path}")
        if marker.get("canonical_stem") != Path(canonical_tag).name:
            raise ValueError(f"completion marker has incompatible canonical stem: {marker_path}")
        if int(marker.get("base_noise_index", -1)) != base_idx:
            raise ValueError(f"completion marker has wrong base index: {marker_path}")
        marker_hp_count = int(marker.get("hp_vectors_per_base", -1))
        marker_interval = int(marker.get("block_interval_solves", -1))
        if int(marker.get("solves_per_hp", -1)) != solves_per_hp:
            raise ValueError(f"completion marker has incompatible solves_per_hp: {marker_path}")
        if marker.get("spin_color_dilution") != str(spin_color_dilution):
            raise ValueError(f"completion marker has incompatible spin-color dilution: {marker_path}")
        if hp_count is None:
            hp_count, interval = marker_hp_count, marker_interval
        elif marker_hp_count != hp_count or marker_interval != interval:
            raise ValueError(f"base {base_idx} completion marker has incompatible part layout")

        expected_names = []
        for part_idx, hp_start, hp_stop in base_part_ranges(
            hp_count, interval, solves_per_hp
        ):
            path = shard_part_path(
                shard_dir, canonical_tag, base_idx, part_idx, hp_start, hp_stop
            )
            expected_names.append(path.name)
            if not path.exists():
                raise ValueError(f"missing shard part {path}")
            expected_bookkeeping = part_source_bookkeeping(
                base_idx, hp_start, hp_stop, hp_count, spin_color_dilution,
                include_spin_color=include_spin_color,
            )
            count = len(expected_bookkeeping["source_index"])
            try:
                with h5py.File(path, "r") as h5:
                    if not bool(h5.attrs.get("part_complete", False)):
                        raise ValueError(f"incomplete shard part {path}")
                    part_fields = {
                        "shard_schema": SHARD_SCHEMA,
                        "base_noise_index": base_idx,
                        "part_index": part_idx,
                        "hp_start": hp_start,
                        "hp_stop_exclusive": hp_stop,
                        "hp_vectors_per_base": hp_count,
                    }
                    for key, expected in part_fields.items():
                        if key not in h5.attrs or not _attr_equal(h5.attrs[key], expected):
                            raise ValueError(f"shard {path} has incompatible attribute {key}")
                    common = {
                        key: h5.attrs[key] for key in h5.attrs if key not in excluded
                    }
                    metadata = {}
                    for name in metadata_dataset_names:
                        if name not in h5:
                            raise ValueError(f"shard {path} is missing metadata dataset {name}")
                        metadata[name] = h5[name][()]
                    current_tails = {}
                    for name in raw_dataset_names:
                        dataset = f"raw/{name}"
                        if dataset not in h5 or h5[dataset].shape[0] != count:
                            raise ValueError(f"shard {path} has incompatible {dataset} shape")
                        current_tails[name] = tuple(h5[dataset].shape[1:])
                    for name, expected in expected_bookkeeping.items():
                        dataset = f"raw/{name}"
                        if dataset not in h5 or not np.array_equal(h5[dataset][()], expected):
                            raise ValueError(f"shard {path} has incompatible {dataset}")
                    if reference_attrs is None:
                        reference_attrs = common
                        reference_metadata = metadata
                        raw_tails = current_tails
                    else:
                        if set(common) != set(reference_attrs):
                            raise ValueError(f"shard {path} has incompatible attribute set")
                        for key, expected in reference_attrs.items():
                            if not _attr_equal(common[key], expected):
                                raise ValueError(f"shard {path} has incompatible attribute {key}")
                        for name, expected in reference_metadata.items():
                            if not np.array_equal(metadata[name], expected):
                                raise ValueError(f"shard {path} has incompatible metadata {name}")
                        if current_tails != raw_tails:
                            raise ValueError(f"shard {path} has incompatible raw dataset shapes")
            except OSError as err:
                raise ValueError(f"cannot open shard part {path}: {err}") from err
            parts.append({
                "base_idx": base_idx,
                "part_idx": part_idx,
                "hp_start": hp_start,
                "hp_stop": hp_stop,
                "path": path,
                "output_start": output_offset,
                "output_stop": output_offset + count,
            })
            output_offset += count
        if marker.get("parts") != expected_names:
            raise ValueError(f"base {base_idx} completion marker has incompatible part list")

    return {
        "reference_attrs": reference_attrs,
        "metadata": reference_metadata,
        "raw_tails": raw_tails,
        "parts": parts,
        "hp_count": hp_count,
        "block_interval_solves": interval,
        "solves_per_hp": solves_per_hp,
        "total_sources": output_offset,
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
    "read_base_completion_marker",
    "selected_base_range",
    "shard_part_path",
    "validate_raw_part_hdf5",
    "validate_complete_shard_set",
    "write_base_completion_marker",
    "write_raw_part_hdf5",
]
