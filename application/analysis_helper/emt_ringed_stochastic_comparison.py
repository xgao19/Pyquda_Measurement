"""Fixed-cost stochastic comparisons for EMT-derived ringed kinetic data."""

from __future__ import annotations

import csv
import os
import tempfile
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
    """One complete-base kinetic estimator series from a canonical EMTc file."""

    mode: str
    path: Path
    n_zn: int
    noise_scheme: str
    hp_count: int
    flow_index: int
    flow_time: float
    base_values: np.ndarray


@dataclass(frozen=True)
class CumulativeRingedStatistics:
    """Cumulative statistics indexed by complete base count and solve cost."""

    mode: str
    n_zn: int
    noise_scheme: str
    hp_count: int
    flow_time: float
    base_count: np.ndarray
    solves: np.ndarray
    mean_real: np.ndarray
    mean_imag: np.ndarray
    sem_real: np.ndarray
    relative_sem: np.ndarray


MODE_SPECS = {
    "z2_pure": {"n_zn": 2, "noise_scheme": "zn", "hp_count": 1},
    "z4_pure": {"n_zn": 4, "noise_scheme": "zn", "hp_count": 1},
    "z2_hp16": {
        "n_zn": 2,
        "noise_scheme": "hierarchical_probing",
        "hp_count": 16,
    },
    "z4_hp16": {
        "n_zn": 4,
        "noise_scheme": "hierarchical_probing",
        "hp_count": 16,
    },
    "z2_hp256": {
        "n_zn": 2,
        "noise_scheme": "hierarchical_probing",
        "hp_count": 256,
    },
    "z4_hp256": {
        "n_zn": 4,
        "noise_scheme": "hierarchical_probing",
        "hp_count": 256,
    },
}


def fixed_cost_job_manifest(total_solves=2048, chunks=4):
    """Build disjoint complete-base jobs for every comparison mode."""
    total_solves = int(total_solves)
    chunks = int(chunks)
    if total_solves <= 0 or chunks <= 0:
        raise ValueError("total_solves and chunks should be positive")
    jobs = []
    for mode, spec in MODE_SPECS.items():
        hp_count = int(spec["hp_count"])
        if total_solves % hp_count:
            raise ValueError(
                f"{total_solves} solves split the natural {mode} base size {hp_count}"
            )
        n_base = total_solves // hp_count
        if n_base % chunks:
            raise ValueError(
                f"{mode} base count {n_base} cannot be split into {chunks} chunks"
            )
        stride = n_base // chunks
        for chunk in range(chunks):
            start = chunk * stride
            stop = start + stride
            jobs.append({
                "job_id": f"{mode}_{start}_{stop}",
                "mode": mode,
                "base_start": start,
                "base_stop": stop,
                "n_base_total": n_base,
                "hp_count": hp_count,
                "solves": (stop - start) * hp_count,
            })
    return jobs


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_ringed_base_series(path, mode, flow_index=1):
    """Load and average complete HP bases from one canonical EMTc file."""
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown comparison mode {mode!r}")
    spec = MODE_SPECS[mode]
    path = Path(path)
    with h5py.File(path, "r") as h5:
        schema = int(h5.attrs.get("emt_operator_schema_version", -1))
        if schema != 3:
            raise ValueError(f"{path} has EMT operator schema {schema}, expected 3")
        n_zn = int(h5.attrs["n_zn"])
        scheme = _text(h5.attrs["noise_scheme"])
        hp_count = int(h5.attrs["hp_num_vectors"]) if scheme == "hierarchical_probing" else 1
        observed = {"n_zn": n_zn, "noise_scheme": scheme, "hp_count": hp_count}
        if observed != spec:
            raise ValueError(f"{path} mode metadata {observed} does not match {mode}: {spec}")

        kinetic = h5["derived/ringed/kinetic_pervec"]
        if kinetic.ndim != 3:
            raise ValueError(f"{path} kinetic_pervec should have source,flow,t axes")
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)
        flow_index = int(flow_index)
        if not 0 <= flow_index < kinetic.shape[1]:
            raise ValueError(f"flow index {flow_index} outside {kinetic.shape[1]} entries")
        base_index = np.asarray(h5["raw/base_noise_index"], dtype=np.int64)
        hp_index = np.asarray(h5["raw/hp_index"], dtype=np.int64)
        if len(base_index) != kinetic.shape[0] or len(hp_index) != kinetic.shape[0]:
            raise ValueError(f"{path} source bookkeeping does not match kinetic source axis")

        bases = np.unique(base_index)
        if not np.array_equal(bases, np.arange(len(bases), dtype=np.int64)):
            raise ValueError(f"{path} base indices should be contiguous from zero")
        base_values = np.empty(len(bases), dtype=np.complex128)
        expected_hp = np.arange(hp_count, dtype=np.int64)
        for base in bases:
            source_rows = np.flatnonzero(base_index == base)
            observed_hp = np.sort(hp_index[source_rows])
            if not np.array_equal(observed_hp, expected_hp):
                raise ValueError(
                    f"{path} base {base} has incomplete HP indices: "
                    f"expected 0..{hp_count - 1}, found {observed_hp.tolist()}"
                )
            base_values[base] = np.mean(kinetic[source_rows, flow_index, :])

        expected_sources = len(bases) * hp_count
        if kinetic.shape[0] != expected_sources:
            raise ValueError(
                f"{path} has {kinetic.shape[0]} sources, expected {expected_sources}"
            )
        if int(h5.attrs["effective_n_inversions"]) != expected_sources:
            raise ValueError(f"{path} effective_n_inversions metadata is inconsistent")
        stored_average = np.asarray(
            h5["derived/ringed/kinetic_spacetime"], dtype=np.complex128
        )[flow_index]

    if not np.allclose(np.mean(base_values), stored_average, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"{path} recomputed base average {np.mean(base_values)} "
            f"does not match stored kinetic_spacetime {stored_average}"
        )
    return RingedBaseSeries(
        mode=mode,
        path=path,
        n_zn=n_zn,
        noise_scheme=scheme,
        hp_count=hp_count,
        flow_index=flow_index,
        flow_time=float(flow_times[flow_index]),
        base_values=base_values,
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
        hp_count=series.hp_count,
        flow_time=series.flow_time,
        base_count=base_count,
        solves=base_count * series.hp_count,
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
        "mode", "n_zn", "noise_scheme", "hp_count", "flow_time", "base_count",
        "solves", "mean_real", "mean_imag", "sem_real", "relative_sem",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODE_SPECS:
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
    values = np.empty(int(n_resamples), dtype=np.float64)
    for idx in range(len(values)):
        num_sample = num[rng.integers(0, len(num), len(num))]
        den_sample = den[rng.integers(0, len(den), len(den))]
        num_sem2 = np.var(num_sample, ddof=1) / len(num_sample)
        den_sem2 = np.var(den_sample, ddof=1) / len(den_sample)
        values[idx] = num_sem2 / den_sem2
    return np.percentile(values, [2.5, 16.0, 84.0, 97.5])


def analyze_ringed_stochastic_comparison(
    mode_files, output_dir, flow_index=1, expected_total_solves=None
):
    """Validate six canonical files and publish tables, five figures, and a report."""
    if set(mode_files) != set(MODE_SPECS):
        raise ValueError(f"mode_files should contain exactly {sorted(MODE_SPECS)}")
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    series = {
        mode: load_ringed_base_series(mode_files[mode], mode, flow_index=flow_index)
        for mode in MODE_SPECS
    }
    stats = {mode: cumulative_ringed_statistics(values) for mode, values in series.items()}
    endings = {mode: int(values.solves[-1]) for mode, values in stats.items()}
    if len(set(endings.values())) != 1:
        raise ValueError(f"all modes should end at one matching solve count: {endings}")
    total_solves = next(iter(endings.values()))
    if expected_total_solves is not None and total_solves != int(expected_total_solves):
        raise ValueError(
            f"expected {int(expected_total_solves)} solves per mode, found {endings}"
        )
    flow_times = {values.flow_time for values in stats.values()}
    if len(flow_times) != 1:
        raise ValueError(f"all modes should use one matching flow time, found {flow_times}")

    _write_cumulative_csv(stats, output_dir / "cumulative_statistics.csv")
    endpoint_fields = [
        "mode", "n_zn", "noise_scheme", "hp_count", "n_base", "solves",
        "flow_time", "mean_real", "mean_imag", "sem_real", "relative_sem",
    ]
    with (output_dir / "endpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=endpoint_fields)
        writer.writeheader()
        for mode in MODE_SPECS:
            values = stats[mode]
            writer.writerow({
                "mode": mode,
                "n_zn": values.n_zn,
                "noise_scheme": values.noise_scheme,
                "hp_count": values.hp_count,
                "n_base": int(values.base_count[-1]),
                "solves": int(values.solves[-1]),
                "flow_time": values.flow_time,
                "mean_real": values.mean_real[-1],
                "mean_imag": values.mean_imag[-1],
                "sem_real": values.sem_real[-1],
                "relative_sem": values.relative_sem[-1],
            })

    comparisons = []
    for method in ("pure", "hp16", "hp256"):
        comparisons.append((
            f"Z2_over_Z4_{method}", f"z2_{method}", f"z4_{method}",
            _variance_ratio(stats[f"z2_{method}"], stats[f"z4_{method}"]),
        ))
    for noise in ("z2", "z4"):
        for method in ("hp16", "hp256"):
            comparisons.append((
                f"{method}_over_pure_{noise}", f"{noise}_{method}", f"{noise}_pure",
                _variance_ratio(stats[f"{noise}_{method}"], stats[f"{noise}_pure"]),
            ))
    comparison_values = {name: value for name, _, _, value in comparisons}
    ratio_path = output_dir / f"variance_ratios_{total_solves}.csv"
    with ratio_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "comparison", "numerator", "denominator", "sem_squared_ratio",
            "bootstrap_95_low", "bootstrap_68_low", "bootstrap_68_high",
            "bootstrap_95_high",
        ])
        for comparison_idx, (name, numerator, denominator, ratio) in enumerate(comparisons):
            interval = _variance_ratio_bootstrap(
                series[numerator], series[denominator], seed=20260714 + comparison_idx
            )
            writer.writerow([name, numerator, denominator, ratio, *interval])

    if total_solves >= 2048:
        comparison_path = output_dir / "endpoint_comparison_2048_final.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "mode", "solves", "n_base", "mean_real", "sem_real",
                "sem_squared_relative_to_2048",
            ])
            for mode in MODE_SPECS:
                values = stats[mode]
                idx_2048 = statistics_at_solves(values, [2048])[0]
                sem2_2048 = values.sem_real[idx_2048] ** 2
                for idx in (idx_2048, len(values.solves) - 1):
                    writer.writerow([
                        mode, int(values.solves[idx]), int(values.base_count[idx]),
                        values.mean_real[idx], values.sem_real[idx],
                        values.sem_real[idx] ** 2 / sem2_2048,
                    ])

    pair_colors = {"Z2": "#d95f02", "Z4": "#1b70a6"}
    method_styles = {
        "pure": ("#222222", "-"),
        "HP16": ("#d95f02", "--"),
        "HP256": ("#1b9e77", "-."),
    }
    _plot_curves([
        ("Z2", stats["z2_pure"], pair_colors["Z2"], "-"),
        ("Z4", stats["z4_pure"], pair_colors["Z4"], "-"),
    ], figures / "01_pure_z2_vs_z4", "Pure stochastic: Z2 versus Z4", max_solves=total_solves)
    _plot_curves([
        ("Z2", stats["z2_hp16"], pair_colors["Z2"], "-"),
        ("Z4", stats["z4_hp16"], pair_colors["Z4"], "-"),
    ], figures / "02_hp16_z2_vs_z4", "HP16 + stochastic: Z2 versus Z4", max_solves=total_solves)
    _plot_curves([
        ("Z2", stats["z2_hp256"], pair_colors["Z2"], "-"),
        ("Z4", stats["z4_hp256"], pair_colors["Z4"], "-"),
    ], figures / "03_hp256_z2_vs_z4", "HP256 + stochastic: Z2 versus Z4", max_solves=total_solves)
    common_solves = np.arange(256, total_solves + 1, 256, dtype=np.int64)
    for noise, figure_index in (("z2", 4), ("z4", 5)):
        curves = []
        for method, label in (("pure", "pure"), ("hp16", "HP16"), ("hp256", "HP256")):
            color, linestyle = method_styles[label]
            curves.append((label, stats[f"{noise}_{method}"], color, linestyle))
        _plot_curves(
            curves,
            figures / f"{figure_index:02d}_{noise}_pure_hp16_hp256",
            f"{noise.upper()}: fixed-cost estimator comparison",
            solve_grid=common_solves,
            max_solves=total_solves,
        )

    report_lines = [
        "# S8T8 EMT disconnected stochastic comparison",
        "",
        f"Fixed gauge: `S8T8_wilson_b6.0`; flow time: `{next(iter(flow_times)):.6f}`.",
        "This is a stochastic-noise benchmark on one gauge, not a physical ensemble determination of Z_chi.",
        "",
        f"## {total_solves}-solve endpoints",
        "",
        "| mode | Re K | Im K | SEM | relative SEM |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODE_SPECS:
        values = stats[mode]
        report_lines.append(
            f"| {mode} | {values.mean_real[-1]:.10e} | {values.mean_imag[-1]:.10e} "
            f"| {values.sem_real[-1]:.4e} | {values.relative_sem[-1]:.4e} |"
        )
    report_lines.extend([
        "",
        f"## Findings at fixed {total_solves}-solve cost",
        "",
        f"- Pure-noise SEM-squared ratio Z2/Z4 = "
        f"{comparison_values['Z2_over_Z4_pure']:.3f}.",
        f"- HP16/pure SEM-squared ratios are "
        f"{comparison_values['hp16_over_pure_z2']:.3f} (Z2) and "
        f"{comparison_values['hp16_over_pure_z4']:.3f} (Z4).",
        f"- HP256/pure SEM-squared ratios are "
        f"{comparison_values['hp256_over_pure_z2']:.3f} (Z2) and "
        f"{comparison_values['hp256_over_pure_z4']:.3f} (Z4).",
        f"- HP256 has {len(series['z2_hp256'].base_values)} complete randomized bases; its variance ratio should be interpreted with that finite-base uncertainty.",
        "- This benchmark uses the spatial production HP ordering with a four-dimensional flowed observable; the result does not rank alternative 4D HP orderings.",
        "- All six EMT-derived kinetic estimates have positive Re K.  The current standalone formula Z_bilinear=-2*Nc/((4*pi)^2*t^2*K) would therefore be negative; its sign convention must be resolved before these data are used for a physical ringed factor.",
        "",
        "Lower relative SEM at the same solve count indicates better stochastic efficiency.",
        "HP uncertainties use complete randomized bases; partial HP prefixes are never treated as estimators.",
        f"See `{ratio_path.name}` for endpoint SEM-squared ratios and `figures/` for the five comparisons.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return stats


__all__ = [
    "CumulativeRingedStatistics",
    "MODE_SPECS",
    "RingedBaseSeries",
    "analyze_ringed_stochastic_comparison",
    "cumulative_ringed_statistics",
    "fixed_cost_job_manifest",
    "load_ringed_base_series",
    "statistics_at_solves",
]
