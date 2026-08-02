"""Complete-base convergence diagnostics for canonical EMT quark loops.

The CLI intentionally reads finalized EMTquarkLoop files, not production shards.  It
compares the two inexpensive diagnostics that are most useful when validating
a stochastic method on one gauge configuration:

* the embedded ringed-fermion kinetic estimator; and
* one selected symmetric EMT component reconstructed from the raw primitive.

Hierarchical-probing rows are averaged within a complete randomized base before
any variance or cumulative-mean estimate is formed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_mpl_cache = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class QuarkOnePointBaseSeries:
    label: str
    path: Path
    hp_count: int
    flow_index: int
    flow_time: float
    q_index: int
    qext: tuple[int, int, int, int]
    component: str
    ringed_base_values: np.ndarray
    emt_base_values: np.ndarray


@dataclass(frozen=True)
class CumulativeStatistics:
    base_count: np.ndarray
    solves: np.ndarray
    mean: np.ndarray
    sem_real: np.ndarray
    relative_sem_real: np.ndarray


def _decode_strings(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _component_indices(component):
    match = re.fullmatch(r"T([1-4])([1-4])", str(component).upper())
    if match is None:
        raise ValueError("component must be T11 ... T44")
    return int(match.group(1)) - 1, int(match.group(2)) - 1


def _unique_zero_momentum(qext):
    matches = np.flatnonzero(np.all(np.asarray(qext)[:, :3] == 0, axis=1))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one q=0 entry, found {len(matches)}")
    return int(matches[0])


def _base_average(values, base_index, hp_index, hp_count):
    bases = np.unique(base_index)
    if not np.array_equal(bases, np.arange(len(bases), dtype=bases.dtype)):
        raise ValueError("base_noise_index must be contiguous from zero")
    expected_hp = np.arange(hp_count, dtype=np.int64)
    result = np.empty(len(bases), dtype=np.complex128)
    for base in bases:
        rows = np.flatnonzero(base_index == base)
        if not np.array_equal(np.sort(hp_index[rows]), expected_hp):
            raise ValueError(
                f"base {int(base)} is incomplete; expected HP indices "
                f"0..{hp_count - 1}"
            )
        result[int(base)] = np.mean(values[rows])
    return result


def load_quark_1pt_base_series(
    path, label=None, flow_index=1, q_index=None, component="T44"
):
    """Load ringed and EMT scalar diagnostics grouped by complete HP base."""
    path = Path(path)
    label = path.stem if label is None else str(label)
    mu, nu = _component_indices(component)
    with h5py.File(path, "r") as h5:
        schema = int(h5.attrs.get("emt_operator_schema_version", -1))
        if schema != 5:
            raise ValueError(f"{path} has EMT operator schema {schema}, expected 5")
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        if q_index is None:
            q_index = _unique_zero_momentum(qext)
        q_index = int(q_index)
        if not 0 <= q_index < len(qext):
            raise ValueError(f"q index {q_index} outside {len(qext)} momenta")

        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)
        flow_index = int(flow_index)
        if not 0 <= flow_index < len(flow_times):
            raise ValueError(f"flow index {flow_index} outside {len(flow_times)} entries")

        scheme = h5.attrs["noise_scheme"]
        if isinstance(scheme, bytes):
            scheme = scheme.decode("utf-8")
        hp_count = (
            int(h5.attrs["hp_num_vectors"])
            if str(scheme) == "hierarchical_probing"
            else 1
        )
        base_index = np.asarray(h5["raw/base_noise_index"], dtype=np.int64)
        hp_index = np.asarray(h5["raw/hp_index"], dtype=np.int64)
        derivative = h5["raw/derivative_bilinear_pervec"]
        kinetic = h5["derived/ringed/kinetic_pervec"]
        if derivative.shape[0] != len(base_index) or kinetic.shape[0] != len(base_index):
            raise ValueError("raw source bookkeeping does not match the source axis")

        labels = _decode_strings(h5["gamma_list"][...])
        vector_position = [labels.index(name) for name in ("X", "Y", "Z", "T")]
        # T_{mu nu}=1/2 (D[gamma_mu,nu] + D[gamma_nu,mu]).  Read only
        # the one or two primitive channels required for the requested component.
        first = np.asarray(
            derivative[:, vector_position[mu], nu, q_index, flow_index, :],
            dtype=np.complex128,
        )
        if mu == nu:
            emt_per_source = np.mean(first, axis=-1)
        else:
            second = np.asarray(
                derivative[:, vector_position[nu], mu, q_index, flow_index, :],
                dtype=np.complex128,
            )
            emt_per_source = np.mean(0.5 * (first + second), axis=-1)
        emt_per_source /= float(h5.attrs["volume_norm"])
        ringed_per_source = np.mean(
            np.asarray(kinetic[:, flow_index, :], dtype=np.complex128), axis=-1
        )

    return QuarkOnePointBaseSeries(
        label=label,
        path=path,
        hp_count=hp_count,
        flow_index=flow_index,
        flow_time=float(flow_times[flow_index]),
        q_index=q_index,
        qext=tuple(int(value) for value in qext[q_index]),
        component=f"T{mu + 1}{nu + 1}",
        ringed_base_values=_base_average(
            ringed_per_source, base_index, hp_index, hp_count
        ),
        emt_base_values=_base_average(emt_per_source, base_index, hp_index, hp_count),
    )


def cumulative_statistics(base_values, hp_count):
    """Return cumulative mean and real-part SEM on complete-base boundaries."""
    values = np.asarray(base_values, dtype=np.complex128)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("base_values must be a non-empty one-dimensional array")
    count = np.arange(1, len(values) + 1, dtype=np.int64)
    mean = np.cumsum(values) / count
    sem = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) > 1:
        real = values.real
        sum_real = np.cumsum(real)
        sumsq_real = np.cumsum(real * real)
        n = count[1:].astype(np.float64)
        variance = np.maximum((sumsq_real[1:] - sum_real[1:] ** 2 / n) / (n - 1), 0)
        sem[1:] = np.sqrt(variance / n)
    relative = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(sem) & (np.abs(mean.real) > 0)
    relative[valid] = sem[valid] / np.abs(mean.real[valid])
    return CumulativeStatistics(
        base_count=count,
        solves=count * int(hp_count),
        mean=mean,
        sem_real=sem,
        relative_sem_real=relative,
    )


def _plot_observable(series_and_stats, observable, output_stem, ylabel):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    colors = plt.get_cmap("tab10")
    for index, (series, stats) in enumerate(series_and_stats):
        color = colors(index % 10)
        x = stats.solves
        top.plot(x, stats.mean.real, label=series.label, color=color)
        valid = np.isfinite(stats.sem_real)
        top.fill_between(
            x[valid],
            stats.mean.real[valid] - stats.sem_real[valid],
            stats.mean.real[valid] + stats.sem_real[valid],
            color=color,
            alpha=0.18,
        )
        rel_valid = np.isfinite(stats.relative_sem_real) & (stats.relative_sem_real > 0)
        bottom.plot(
            x[rel_valid], stats.relative_sem_real[rel_valid],
            marker="o", markersize=2.5, color=color,
        )
    top.set_ylabel(ylabel)
    top.set_title(observable)
    top.legend(frameon=False)
    bottom.set_xlabel("Dirac solves (complete-base boundaries)")
    bottom.set_ylabel(r"SEM$/|\mathrm{Re}\,\bar L|$")
    bottom.set_yscale("log")
    top.grid(alpha=0.25)
    bottom.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(Path(output_stem).with_suffix(".png"), dpi=220)
    fig.savefig(Path(output_stem).with_suffix(".pdf"))
    plt.close(fig)


def analyze_quark_1pt_convergence(
    inputs, output_dir, flow_index=1, q_index=None, component="T44"
):
    """Create CSV summaries and two convergence figures from EMTquarkLoop files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = [
        load_quark_1pt_base_series(
            path, label=label, flow_index=flow_index,
            q_index=q_index, component=component,
        )
        for label, path in inputs
    ]
    if len({item.label for item in series}) != len(series):
        raise ValueError("input labels must be unique")
    reference = (series[0].flow_time, series[0].qext, series[0].component)
    if any((item.flow_time, item.qext, item.component) != reference for item in series[1:]):
        raise ValueError("all inputs must use the same selected flow time, momentum, and component")

    ringed = [(item, cumulative_statistics(item.ringed_base_values, item.hp_count)) for item in series]
    emt = [(item, cumulative_statistics(item.emt_base_values, item.hp_count)) for item in series]

    fields = [
        "label", "path", "observable", "flow_index", "flow_time", "q_index",
        "qext", "component", "hp_count", "base_count", "solves", "mean_real",
        "mean_imag", "sem_real", "relative_sem_real",
    ]
    with (output_dir / "cumulative_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for observable, values in (("ringed_kinetic", ringed), (series[0].component, emt)):
            for item, stats in values:
                for index in range(len(stats.solves)):
                    writer.writerow({
                        "label": item.label,
                        "path": item.path,
                        "observable": observable,
                        "flow_index": item.flow_index,
                        "flow_time": item.flow_time,
                        "q_index": item.q_index,
                        "qext": ".".join(map(str, item.qext)),
                        "component": item.component,
                        "hp_count": item.hp_count,
                        "base_count": int(stats.base_count[index]),
                        "solves": int(stats.solves[index]),
                        "mean_real": stats.mean.real[index],
                        "mean_imag": stats.mean.imag[index],
                        "sem_real": stats.sem_real[index],
                        "relative_sem_real": stats.relative_sem_real[index],
                    })

    with (output_dir / "endpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "observable", "n_base", "solves", "mean_real", "mean_imag", "sem_real", "relative_sem_real"])
        for observable, values in (("ringed_kinetic", ringed), (series[0].component, emt)):
            for item, stats in values:
                writer.writerow([
                    item.label, observable, int(stats.base_count[-1]), int(stats.solves[-1]),
                    stats.mean.real[-1], stats.mean.imag[-1], stats.sem_real[-1],
                    stats.relative_sem_real[-1],
                ])

    _plot_observable(
        ringed,
        f"ringed kinetic at flow index {series[0].flow_index}",
        output_dir / "ringed_kinetic_convergence",
        r"cumulative $\mathrm{Re}\,K$",
    )
    _plot_observable(
        emt,
        f"{series[0].component}, q={series[0].qext}, flow index {series[0].flow_index}",
        output_dir / f"{series[0].component.lower()}_convergence",
        rf"cumulative $\mathrm{{Re}}\,{series[0].component}$",
    )
    return series


def _parse_input(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("--input must be LABEL=FILE.h5")
    label, path = text.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--input must be LABEL=FILE.h5")
    return label, path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare complete-base convergence in finalized EMTquarkLoop files."
    )
    parser.add_argument("--input", action="append", type=_parse_input, required=True, metavar="LABEL=FILE")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--flow_index", type=int, default=1)
    parser.add_argument("--q_index", type=int)
    parser.add_argument("--component", default="T44")
    args = parser.parse_args(argv)
    analyze_quark_1pt_convergence(
        args.input, args.output_dir, flow_index=args.flow_index,
        q_index=args.q_index, component=args.component,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CumulativeStatistics",
    "QuarkOnePointBaseSeries",
    "analyze_quark_1pt_convergence",
    "cumulative_statistics",
    "load_quark_1pt_base_series",
    "main",
]
