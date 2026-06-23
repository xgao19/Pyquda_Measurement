#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


DEFAULT_MATCHED_SOLVES = [16, 32, 64, 128, 256, 512, 1024]
CASES = {
    "zn1024": {
        "label": "pure_stochastic",
        "h5": "data/zn1024/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.0.0.x0y0z0t0.S8T8_zn1024.h5",
        "stderr": "log/zn1024.e",
        "block_size": 16,
        "raw_shape": (1024, 2, 8),
        "matched_solves": DEFAULT_MATCHED_SOLVES,
    },
    "hp64x16": {
        "label": "stochastic_hp_16",
        "h5": "data/hp64x16/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.0.0.x0y0z0t0.S8T8_hp64x16.h5",
        "stderr": "log/hp64x16.e",
        "block_size": 16,
        "raw_shape": (1024, 2, 8),
        "matched_solves": DEFAULT_MATCHED_SOLVES,
    },
    "hp4x256": {
        "label": "stochastic_hp_256",
        "h5": "data/hp4x256/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.0.0.x0y0z0t0.S8T8_hp4x256.h5",
        "stderr": "log/hp4x256.e",
        "block_size": 256,
        "raw_shape": (1024, 2, 8),
        "matched_solves": DEFAULT_MATCHED_SOLVES,
    },
    "hp6x16sc12": {
        "label": "stochastic_hp_16_spin_color_dilution",
        "h5": "data/hp6x16sc12/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.0.0.x0y0z0t0.S8T8_hp6x16sc12.h5",
        "stderr": "log/hp6x16sc12.e",
        "block_size": 192,
        "raw_shape": (1152, 2, 8),
        "matched_solves": [192, 384, 576, 768, 960, 1152],
    },
}


def _parse_real_seconds(path):
    if not path.exists():
        return None
    match = re.search(r"^real\s+([0-9.]+)\s*$", path.read_text(errors="replace"), flags=re.MULTILINE)
    return None if match is None else float(match.group(1))


def _z_bilinear(kinetic, flow_time, nc):
    return -2.0 * float(nc) / (((4.0 * np.pi) ** 2) * flow_time**2 * kinetic)


def _sem_abs(block_values):
    if len(block_values) < 2:
        return float("nan")
    return float(np.std(block_values, ddof=1) / np.sqrt(len(block_values)))


def analyze_case(bench_root, case_name, spec):
    h5_path = bench_root / spec["h5"]
    stderr_path = bench_root / spec["stderr"]
    walltime = _parse_real_seconds(stderr_path)
    block_size = int(spec["block_size"])
    rows = []
    with h5py.File(h5_path, "r") as h5:
        raw = h5["raw/kinetic_pervec"][()]
        flow_times = h5["flow_times"][()]
        z_field = h5["avg/Z_ring_field_sqrt"][()]
        z_bilinear = h5["avg/Z_ring_bilinear"][()]
        nc = int(h5.attrs.get("Nc", 3))
        spin_color_trace_factor = float(h5.attrs.get("spin_color_trace_factor", h5.attrs.get("spin_color_dilution_factor", 1)))
        expected_shape = tuple(spec["raw_shape"])
        if raw.shape != expected_shape:
            raise ValueError(f"{case_name} raw/kinetic_pervec shape should be {expected_shape}, got {raw.shape}")
        if not np.isnan(z_field[0]) or not np.isnan(z_bilinear[0]):
            raise ValueError(f"{case_name} flow0 ringed factors should be NaN")
        if not np.all(np.isfinite(raw[:, 1, :])):
            raise ValueError(f"{case_name} positive-flow kinetic_pervec contains non-finite values")
        if not np.allclose(z_field[1:] ** 2, z_bilinear[1:]):
            raise ValueError(f"{case_name} Z_ring_field_sqrt**2 does not match Z_ring_bilinear")

        flow_time = float(flow_times[1])
        for solves in spec["matched_solves"]:
            if solves % block_size != 0:
                continue
            flow1 = raw[:solves, 1, :]
            cumulative_k = spin_color_trace_factor * np.mean(flow1)
            block_values = spin_color_trace_factor * flow1.reshape(solves // block_size, block_size, flow1.shape[-1]).mean(axis=(1, 2))
            sem_abs = _sem_abs(block_values)
            relative_sem = sem_abs / max(abs(cumulative_k), 1e-300)
            cumulative_z = _z_bilinear(cumulative_k, flow_time, nc)
            rows.append(
                {
                    "case": case_name,
                    "label": spec["label"],
                    "solves": solves,
                    "blocks": solves // block_size,
                    "block_size": block_size,
                    "K_real": float(np.real(cumulative_k)),
                    "K_imag": float(np.imag(cumulative_k)),
                    "K_abs": float(abs(cumulative_k)),
                    "K_block_sem_abs": sem_abs,
                    "K_block_relative_sem": float(relative_sem),
                    "Z_bilinear_real": float(np.real(cumulative_z)),
                    "Z_bilinear_imag": float(np.imag(cumulative_z)),
                    "Z_bilinear_abs": float(abs(cumulative_z)),
                    "total_walltime_sec": walltime,
                    "solves_per_sec_total": None if walltime is None else raw.shape[0] / walltime,
                    "spin_color_dilution": h5.attrs.get("spin_color_dilution", "none"),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", default=str(Path(__file__).resolve().parent / "benchmark/s8t8_hp_convergence"))
    args = parser.parse_args()

    bench_root = Path(args.bench_root).resolve()
    all_rows = []
    for case_name, spec in CASES.items():
        all_rows.extend(analyze_case(bench_root, case_name, spec))

    csv_path = bench_root / "summary.csv"
    json_path = bench_root / "summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps(all_rows, indent=2) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
