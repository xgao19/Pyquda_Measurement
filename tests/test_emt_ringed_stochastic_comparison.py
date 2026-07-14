from pathlib import Path

import h5py
import numpy as np

from application.analysis_helper.emt_ringed_stochastic_comparison import (
    MODE_SPECS,
    RingedBaseSeries,
    analyze_ringed_stochastic_comparison,
    cumulative_ringed_statistics,
    fixed_cost_job_manifest,
    load_ringed_base_series,
    statistics_at_solves,
)


def _write_canonical(path, mode, n_base=None, incomplete_last_base=False):
    spec = MODE_SPECS[mode]
    hp_count = spec["hp_count"]
    n_base = 2048 // hp_count if n_base is None else int(n_base)
    base_index = np.repeat(np.arange(n_base, dtype=np.int32), hp_count)
    hp_index = np.tile(np.arange(hp_count, dtype=np.int32), n_base)
    if incomplete_last_base:
        base_index = base_index[:-1]
        hp_index = hp_index[:-1]
    base_values = -2.0 - 0.01 * np.arange(n_base) + 0.001j * np.arange(n_base)
    values = np.repeat(base_values, hp_count)[: len(base_index)]
    kinetic = np.zeros((len(values), 2, 3), dtype=np.complex128)
    kinetic[:, 1, :] = values[:, None]
    with h5py.File(path, "w") as h5:
        h5.attrs["emt_operator_schema_version"] = 3
        h5.attrs["n_zn"] = spec["n_zn"]
        h5.attrs["noise_scheme"] = spec["noise_scheme"]
        h5.attrs["hp_num_vectors"] = hp_count
        h5.attrs["flow_times"] = [0.0, 0.207936]
        h5.attrs["effective_n_inversions"] = len(values)
        raw = h5.require_group("raw")
        raw.create_dataset("base_noise_index", data=base_index)
        raw.create_dataset("hp_index", data=hp_index)
        ringed = h5.require_group("derived/ringed")
        ringed.create_dataset("kinetic_pervec", data=kinetic)
        ringed.create_dataset("kinetic_spacetime", data=np.mean(kinetic, axis=(0, -1)))
    return base_values


def test_cumulative_statistics_use_complete_base_cost():
    series = RingedBaseSeries(
        mode="z4_hp16",
        path=Path("synthetic.h5"),
        n_zn=4,
        noise_scheme="hierarchical_probing",
        hp_count=16,
        flow_index=1,
        flow_time=0.207936,
        base_values=np.asarray([1.0, 3.0, 5.0]) + 1j,
    )
    stats = cumulative_ringed_statistics(series)
    np.testing.assert_array_equal(stats.solves, [16, 32, 48])
    np.testing.assert_allclose(stats.mean_real, [1.0, 2.0, 3.0])
    assert np.isnan(stats.sem_real[0])
    np.testing.assert_allclose(stats.sem_real[1:], [1.0, 2.0 / np.sqrt(3.0)])
    np.testing.assert_array_equal(statistics_at_solves(stats, [16, 48]), [0, 2])


def test_loader_groups_hp_vectors_and_rejects_incomplete_base(tmp_path):
    path = tmp_path / "complete.h5"
    expected = _write_canonical(path, "z2_hp16", n_base=3)
    series = load_ringed_base_series(path, "z2_hp16", flow_index=1)
    np.testing.assert_allclose(series.base_values, expected)

    incomplete = tmp_path / "incomplete.h5"
    _write_canonical(incomplete, "z2_hp16", n_base=3, incomplete_last_base=True)
    try:
        load_ringed_base_series(incomplete, "z2_hp16", flow_index=1)
    except ValueError as err:
        assert "incomplete HP indices" in str(err)
    else:
        raise AssertionError("incomplete HP base should be rejected")


def test_full_analysis_writes_five_png_and_pdf_figures(tmp_path):
    mode_files = {}
    for mode in MODE_SPECS:
        path = tmp_path / f"{mode}.h5"
        _write_canonical(path, mode)
        mode_files[mode] = path
    output = tmp_path / "results"
    analyze_ringed_stochastic_comparison(mode_files, output)
    assert (output / "cumulative_statistics.csv").is_file()
    assert (output / "endpoint_summary.csv").is_file()
    assert (output / "variance_ratios_2048.csv").is_file()
    assert (output / "report.md").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 5
    assert len(list((output / "figures").glob("*.pdf"))) == 5


def test_run_manifest_has_24_disjoint_512_solve_jobs():
    jobs = fixed_cost_job_manifest()
    assert len(jobs) == 24
    assert all(job["solves"] == 512 for job in jobs)
    for mode in MODE_SPECS:
        selected = [job for job in jobs if job["mode"] == mode]
        assert sum(job["solves"] for job in selected) == 2048
        assert selected[0]["base_start"] == 0
        assert selected[-1]["base_stop"] == 2048 // MODE_SPECS[mode]["hp_count"]
        assert all(
            left["base_stop"] == right["base_start"]
            for left, right in zip(selected, selected[1:])
        )


def test_8192_solve_manifest_has_96_complete_base_jobs():
    jobs = fixed_cost_job_manifest(total_solves=8192, chunks=16)
    assert len(jobs) == 96
    assert all(job["solves"] == 512 for job in jobs)
    for mode, spec in MODE_SPECS.items():
        selected = [job for job in jobs if job["mode"] == mode]
        assert selected[0]["base_start"] == 0
        assert selected[-1]["base_stop"] == 8192 // spec["hp_count"]
        assert sum(job["solves"] for job in selected) == 8192
