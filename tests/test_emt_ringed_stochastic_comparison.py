from pathlib import Path

import h5py
import numpy as np
import pytest

from application.analysis_helper.emt_ringed_stochastic_comparison import (
    RingedBaseSeries, analyze_ringed_stochastic_comparison,
    cumulative_ringed_statistics, fixed_cost_job_manifest,
    load_ringed_base_series, statistics_at_solves,
)


def _write_bases(path, count=2, n_base=4, dilution="none", scheme="hierarchical_probing"):
    values = -2.0 - 0.01 * np.arange(n_base) + 0.001j * np.arange(n_base)
    kinetic = np.stack((np.zeros(n_base), values), axis=1)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(dict(
            output_kind="emt_complete_base_statistics_v2",
            shard_schema="disconnected_1pt_base_parts_v4",
            emt_operator_schema_version=5, base_values_volume_normalized=True,
            n_zn=4, noise_scheme=scheme, vectors_per_base=count,
            spin_color_dilution=dilution, n_base_noise=n_base,
            flow_times=[0.0, 0.207936],
            effective_n_inversions=n_base*count*(12 if dilution == "point" else 1),
        ))
        h5.create_dataset("base/base_noise_index", data=np.arange(n_base))
        h5.create_dataset("base/ringed/kinetic_spacetime", data=kinetic)
        h5.create_dataset("avg/ringed/kinetic_spacetime", data=kinetic.mean(axis=0))
    return values


def test_cumulative_statistics_use_complete_base_cost():
    series = RingedBaseSeries(
        mode="hp2_sc12", path=Path("synthetic.bases.h5"), n_zn=4,
        noise_scheme="hierarchical_probing", probe_count=2, solves_per_base=24,
        flow_index=1, flow_time=0.207936, base_values=np.array([1., 3., 5.])+1j,
    )
    stats = cumulative_ringed_statistics(series)
    np.testing.assert_array_equal(stats.solves, [24, 48, 72])
    np.testing.assert_allclose(stats.mean_real, [1, 2, 3])
    assert np.isnan(stats.sem_real[0])
    np.testing.assert_allclose(stats.sem_real[1:], [1, 2/np.sqrt(3)])
    np.testing.assert_array_equal(statistics_at_solves(stats, [24, 72]), [0, 2])
    with pytest.raises(ValueError, match="no complete-base"):
        statistics_at_solves(stats, [25])


@pytest.mark.parametrize("count,dilution,scheme", [
    (1, "none", "zn"), (32, "none", "hierarchical_probing"),
    (16, "none", "multiplier_probing"), (2, "point", "hierarchical_probing"),
])
def test_loader_reads_base_values_and_method_metadata(tmp_path, count, dilution, scheme):
    path = tmp_path / "loop.bases.h5"
    expected = _write_bases(path, count=count, dilution=dilution, scheme=scheme)
    series = load_ringed_base_series(path, "user_label")
    np.testing.assert_array_equal(series.base_values, expected)
    assert series.probe_count == count and series.noise_scheme == scheme
    assert series.solves_per_base == count * (12 if dilution == "point" else 1)


@pytest.mark.parametrize("damage", ["schema", "count", "indices", "nan", "mean", "shape"])
def test_loader_rejects_invalid_bases(tmp_path, damage):
    path = tmp_path / "loop.bases.h5"
    _write_bases(path)
    with h5py.File(path, "r+") as h5:
        if damage == "schema":
            h5.attrs["output_kind"] = "legacy"
        elif damage == "count":
            h5.attrs["effective_n_inversions"] = 7
        elif damage == "indices":
            h5["base/base_noise_index"][1] = 0
        elif damage == "nan":
            h5["base/ringed/kinetic_spacetime"][0, 1] = np.nan
        elif damage == "mean":
            h5["avg/ringed/kinetic_spacetime"][1] = 9
        else:
            del h5["base/ringed/kinetic_spacetime"]
            h5.create_dataset("base/ringed/kinetic_spacetime", data=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        load_ringed_base_series(path)


def test_analysis_accepts_new_methods_and_writes_outputs(tmp_path):
    modes = {"hp32": (32, 12, "none", "hierarchical_probing"),
             "m16": (16, 24, "none", "multiplier_probing"),
             "hp2_sc12": (2, 16, "point", "hierarchical_probing")}
    files = {}
    for mode, (count, bases, dilution, scheme) in modes.items():
        path = tmp_path / (mode + ".bases.h5")
        _write_bases(path, count, bases, dilution, scheme)
        files[mode] = path
    output = tmp_path / "results"
    stats = analyze_ringed_stochastic_comparison(files, output, expected_total_solves=384)
    assert set(stats) == set(modes)
    for name in ("cumulative_statistics.csv", "endpoint_summary.csv", "variance_ratios_384.csv", "report.md"):
        assert (output/name).is_file()
    assert (output/"figures/kinetic_comparison.png").is_file()
    assert (output/"figures/kinetic_comparison.pdf").is_file()


def test_manifest_uses_explicit_probe_and_dilution_costs():
    specs = {"hp32": dict(vectors_per_base=32),
             "m16": dict(vectors_per_base=16),
             "hp2_sc12": dict(vectors_per_base=2, spin_color_dilution="point")}
    jobs = fixed_cost_job_manifest(specs, total_solves=384, chunks=4)
    assert len(jobs) == 12 and all(job["solves"] == 96 for job in jobs)
    for mode in specs:
        selected = [job for job in jobs if job["mode"] == mode]
        assert selected[0]["base_start"] == 0
        assert all(a["base_stop"] == b["base_start"] for a,b in zip(selected, selected[1:]))
    with pytest.raises(ValueError, match="split a complete"):
        fixed_cost_job_manifest(specs, total_solves=100, chunks=4)
