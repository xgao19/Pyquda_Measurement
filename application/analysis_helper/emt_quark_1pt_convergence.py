"""Complete-base convergence diagnostics from finalized EMT .bases.h5 files."""

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

from application.analysis_helper.emt_base_data import validate_base_file

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
    solves_per_base: int
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


def load_quark_1pt_base_series(path, label=None, flow_index=1, q_index=None, component="T44"):
    """Read normalized EMT and kinetic observables from complete-base output."""
    path = Path(path)
    mu, nu = sorted(_component_indices(component))
    with h5py.File(path, "r") as h5:
        bases, cost = validate_base_file(h5)
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        q_index = _unique_zero_momentum(qext) if q_index is None else int(q_index)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=float)
        flow_index = int(flow_index)
        if not 0 <= q_index < len(qext) or not 0 <= flow_index < len(flow_times):
            raise ValueError("momentum/flow index is outside stored axes")
        tensor = h5[f"base/Tmunu/T{mu+1}{nu+1}"]
        nt = int(h5.attrs["global_lattice_size"][3])
        if tensor.shape != (len(bases), len(qext), len(flow_times), nt):
            raise ValueError("EMT base axes do not match metadata")
        kinetic = h5["base/ringed/kinetic_spacetime"]
        if kinetic.shape != (len(bases), len(flow_times)):
            raise ValueError("kinetic base axes do not match metadata")
        emt_values = np.mean(tensor[:, q_index, flow_index, :], axis=-1)
        ringed_values = kinetic[:, flow_index]
        if not np.all(np.isfinite(emt_values)) or not np.all(np.isfinite(ringed_values)):
            raise ValueError("non-finite complete-base observables")
    return QuarkOnePointBaseSeries(
        label=path.stem if label is None else str(label), path=path, solves_per_base=cost,
        flow_index=flow_index, flow_time=float(flow_times[flow_index]), q_index=q_index,
        qext=tuple(int(v) for v in qext[q_index]), component=f"T{mu+1}{nu+1}",
        ringed_base_values=ringed_values, emt_base_values=emt_values,
    )


def cumulative_statistics(base_values, solves_per_base):
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
        solves=count * int(solves_per_base),
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

    ringed = [(item, cumulative_statistics(item.ringed_base_values, item.solves_per_base)) for item in series]
    emt = [(item, cumulative_statistics(item.emt_base_values, item.solves_per_base)) for item in series]

    fields = [
        "label", "path", "observable", "flow_index", "flow_time", "q_index",
        "qext", "component", "solves_per_base", "base_count", "solves", "mean_real",
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
                        "solves_per_base": item.solves_per_base,
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
