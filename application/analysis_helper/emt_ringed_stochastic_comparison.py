"""Fixed-cost stochastic comparisons for EMT-derived ringed kinetic data."""

from __future__ import annotations

import csv
import os
import tempfile
from itertools import combinations

from application.analysis_helper.emt_base_data import validate_base_file
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_mpl_cache = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
_xdg_cache = Path(tempfile.gettempdir()) / f"xdg-cache-{os.getuid()}"
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RingedBaseSeries:
    """One complete-base kinetic estimator series from a canonical EMTquarkLoop file."""

    mode: str
    path: Path
    n_zn: int
    noise_scheme: str
    probe_count: int
    solves_per_base: int
    flow_index: int
    flow_time: float
    base_values: np.ndarray


@dataclass(frozen=True)
class CumulativeRingedStatistics:
    """Cumulative statistics indexed by complete base count and solve cost."""

    mode: str
    n_zn: int
    noise_scheme: str
    probe_count: int
    solves_per_base: int
    flow_time: float
    base_count: np.ndarray
    solves: np.ndarray
    mean_real: np.ndarray
    mean_imag: np.ndarray
    sem_real: np.ndarray
    relative_sem: np.ndarray


def fixed_cost_job_manifest(mode_specs, total_solves=2048, chunks=4):
    """Build complete-base partitions from explicit method specifications."""
    total_solves, chunks = int(total_solves), int(chunks)
    if total_solves <= 0 or chunks <= 0 or not mode_specs:
        raise ValueError("positive total_solves/chunks and nonempty mode_specs are required")
    jobs = []
    for mode, spec in mode_specs.items():
        count = int(spec["vectors_per_base"])
        dilution = spec.get("spin_color_dilution", "none")
        if count <= 0 or dilution not in ("none", "point"):
            raise ValueError("invalid probe count or dilution")
        cost = count * (12 if dilution == "point" else 1)
        if total_solves % cost or (total_solves // cost) % chunks:
            raise ValueError(f"budget/chunks split a complete {mode} base")
        n_base = total_solves // cost
        stride = n_base // chunks
        for chunk in range(chunks):
            start, stop = chunk * stride, (chunk + 1) * stride
            jobs.append(dict(job_id=f"{mode}_{start}_{stop}", mode=mode,
                             base_start=start, base_stop=stop, n_base_total=n_base,
                             vectors_per_base=count, solves=(stop-start)*cost))
    return jobs


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_ringed_base_series(path, mode=None, flow_index=1):
    """Read independent kinetic estimates directly from a finalized .bases.h5."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        bases, cost = validate_base_file(h5)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=float)
        kinetic = h5["base/ringed/kinetic_spacetime"]
        if kinetic.shape != (len(bases), len(flow_times)):
            raise ValueError("kinetic base/flow axes do not match metadata")
        flow_index = int(flow_index)
        if not 0 <= flow_index < len(flow_times):
            raise ValueError("flow index is outside stored flow times")
        values = np.asarray(kinetic[:, flow_index], dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite base kinetic values")
        average = h5["avg/ringed/kinetic_spacetime"][flow_index]
        if not np.allclose(values.mean(), average, rtol=1e-12, atol=1e-12):
            raise ValueError("base kinetic mean disagrees with stored average")
        return RingedBaseSeries(
            mode=path.stem if mode is None else str(mode), path=path,
            n_zn=int(h5.attrs["n_zn"]), noise_scheme=_text(h5.attrs["noise_scheme"]),
            probe_count=int(h5.attrs["vectors_per_base"]), solves_per_base=cost,
            flow_index=flow_index, flow_time=float(flow_times[flow_index]), base_values=values,
        )


def cumulative_ringed_statistics(series):
    """Compute cumulative mean and real-part SEM using complete bases."""
    values = np.asarray(series.base_values, dtype=np.complex128)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("base_values should be a non-empty one-dimensional array")
    base_count = np.arange(1, len(values) + 1, dtype=np.int64)
    mean = np.cumsum(values) / base_count
    real = values.real
    sum_real = np.cumsum(real)
    sumsq_real = np.cumsum(real * real)
    sem = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) > 1:
        n = base_count[1:].astype(np.float64)
        centered_sum = sumsq_real[1:] - sum_real[1:] ** 2 / n
        sample_variance = np.maximum(centered_sum / (n - 1.0), 0.0)
        sem[1:] = np.sqrt(sample_variance / n)
    relative_sem = np.full(len(values), np.nan, dtype=np.float64)
    denominator = np.abs(mean.real)
    valid = np.isfinite(sem) & (denominator > 0)
    relative_sem[valid] = sem[valid] / denominator[valid]
    return CumulativeRingedStatistics(
        mode=series.mode,
        n_zn=series.n_zn,
        noise_scheme=series.noise_scheme,
        probe_count=series.probe_count,
        solves_per_base=series.solves_per_base,
        flow_time=series.flow_time,
        base_count=base_count,
        solves=base_count * series.solves_per_base,
        mean_real=mean.real,
        mean_imag=mean.imag,
        sem_real=sem,
        relative_sem=relative_sem,
    )


def statistics_at_solves(stats, solve_grid):
    """Select exact solve counts, rejecting costs that split a natural base."""
    solve_grid = np.asarray(solve_grid, dtype=np.int64)
    lookup = {int(value): idx for idx, value in enumerate(stats.solves)}
    missing = [int(value) for value in solve_grid if int(value) not in lookup]
    if missing:
        raise ValueError(f"{stats.mode} has no complete-base estimates at solves {missing}")
    return np.asarray([lookup[int(value)] for value in solve_grid], dtype=np.int64)


def _write_cumulative_csv(stats_by_mode, path):
    fields = [
        "mode", "n_zn", "noise_scheme", "probe_count", "flow_time", "base_count",
        "solves", "mean_real", "mean_imag", "sem_real", "relative_sem",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in stats_by_mode:
            stats = stats_by_mode[mode]
            for idx in range(len(stats.solves)):
                writer.writerow({field: getattr(stats, field)[idx] if field in {
                    "base_count", "solves", "mean_real", "mean_imag", "sem_real", "relative_sem"
                } else getattr(stats, field) for field in fields})


def _plot_curves(curves, output_stem, title, solve_grid=None, max_solves=None):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for label, stats, color, linestyle in curves:
        if solve_grid is None:
            idx = np.arange(len(stats.solves))
        else:
            idx = statistics_at_solves(stats, solve_grid)
        x = stats.solves[idx]
        mean = stats.mean_real[idx]
        sem = stats.sem_real[idx]
        top.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.7, label=label)
        valid_band = np.isfinite(sem)
        top.fill_between(
            x[valid_band], mean[valid_band] - sem[valid_band],
            mean[valid_band] + sem[valid_band], color=color, alpha=0.18,
        )
        relative = stats.relative_sem[idx]
        valid_relative = np.isfinite(relative) & (relative > 0)
        bottom.plot(
            x[valid_relative], relative[valid_relative], color=color,
            linestyle=linestyle, linewidth=1.7, marker="o", markersize=2.5,
            label=label,
        )
    top.set_title(title)
    top.set_ylabel(r"cumulative $\mathrm{Re}\,K$")
    top.legend(frameon=False)
    bottom.set_xlabel("Dirac solves")
    bottom.set_ylabel(r"SEM$/|\mathrm{Re}\,K|$")
    bottom.set_yscale("log")
    top.grid(alpha=0.25)
    bottom.grid(alpha=0.25, which="both")
    if max_solves is None:
        max_solves = max(int(stats.solves[-1]) for _, stats, _, _ in curves)
    bottom.set_xlim(0, int(max_solves))
    fig.tight_layout()
    output_stem = Path(output_stem)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _variance_ratio(numerator, denominator):
    num = float(numerator.sem_real[-1])
    den = float(denominator.sem_real[-1])
    return np.nan if not np.isfinite(num) or not np.isfinite(den) or den == 0 else (num / den) ** 2


def _variance_ratio_bootstrap(numerator, denominator, n_resamples=2000, seed=0):
    rng = np.random.default_rng(int(seed))
    num = np.asarray(numerator.base_values.real, dtype=np.float64)
    den = np.asarray(denominator.base_values.real, dtype=np.float64)
    if min(len(num), len(den)) < 2 or np.var(den) == 0:
        return np.full(4, np.nan)
    values = np.empty(int(n_resamples), dtype=np.float64)
    for idx in range(len(values)):
        num_sample = num[rng.integers(0, len(num), len(num))]
        den_sample = den[rng.integers(0, len(den), len(den))]
        num_sem2 = np.var(num_sample, ddof=1) / len(num_sample)
        den_sem2 = np.var(den_sample, ddof=1) / len(den_sample)
        values[idx] = num_sem2 / den_sem2 if den_sem2 > 0 else np.nan
    finite = values[np.isfinite(values)]
    return np.percentile(finite, [2.5, 16.0, 84.0, 97.5]) if len(finite) else np.full(4, np.nan)


def analyze_ringed_stochastic_comparison(
    mode_files, output_dir, flow_index=1, expected_total_solves=None
):
    """Compare metadata-defined methods at equal cost using complete bases."""
    if not mode_files:
        raise ValueError("mode_files must not be empty")
    series = {mode: load_ringed_base_series(path, mode, flow_index)
              for mode, path in mode_files.items()}
    stats = {mode: cumulative_ringed_statistics(item) for mode, item in series.items()}
    endings = {int(item.solves[-1]) for item in stats.values()}
    if len(endings) != 1:
        raise ValueError("all methods must have matching total solve counts")
    total_solves = endings.pop()
    if expected_total_solves is not None and total_solves != int(expected_total_solves):
        raise ValueError("total solve count does not match requested budget")
    if len({item.flow_time for item in series.values()}) != 1:
        raise ValueError("all methods must use the same flow time")
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_cumulative_csv(stats, output_dir / "cumulative_statistics.csv")
    with (output_dir / "endpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "noise_scheme", "vectors_per_base", "solves_per_base",
                         "n_base", "solves", "mean_real", "mean_imag", "sem_real"])
        for mode, item in stats.items():
            writer.writerow([mode, item.noise_scheme, item.probe_count, item.solves_per_base,
                             item.base_count[-1], item.solves[-1], item.mean_real[-1],
                             item.mean_imag[-1], item.sem_real[-1]])
    with (output_dir / f"variance_ratios_{total_solves}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["numerator", "denominator", "sem_squared_ratio", "bootstrap_95_low",
                         "bootstrap_68_low", "bootstrap_68_high", "bootstrap_95_high"])
        for index, (num, den) in enumerate(combinations(stats, 2)):
            writer.writerow([num, den, _variance_ratio(stats[num], stats[den]),
                             *_variance_ratio_bootstrap(series[num], series[den], seed=index)])
    colors = plt.get_cmap("tab10")
    curves = [(mode, item, colors(i % 10), "-") for i, (mode, item) in enumerate(stats.items())]
    _plot_curves(curves, figures / "kinetic_comparison", "Complete-base kinetic comparison",
                 max_solves=total_solves)
    lines = ["# Complete-base stochastic comparison", "",
             f"Total cost per method: {total_solves} Dirac solves.", "",
             "| method | bases | Re K | Im K | SEM(Re K) |", "|---|---:|---:|---:|---:|"]
    for mode, item in stats.items():
        lines.append(f"| {mode} | {item.base_count[-1]} | {item.mean_real[-1]:.10e} | "
                     f"{item.mean_imag[-1]:.10e} | {item.sem_real[-1]:.4e} |")
    lines.extend(["", "SEM uses independent complete bases; probe and dilution channels are not independent samples.",
                  "Cost includes all spin-color dilution solves. Single-base uncertainty is undefined.",
                  "Variance-ratio intervals use independent resampling of the two base series; they are not paired intervals.",
                  "These are fixed-gauge stochastic uncertainties, not ensemble gauge errors.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return stats


__all__ = [
    "CumulativeRingedStatistics",
    "RingedBaseSeries",
    "analyze_ringed_stochastic_comparison",
    "cumulative_ringed_statistics",
    "fixed_cost_job_manifest",
    "load_ringed_base_series",
    "statistics_at_solves",
]
