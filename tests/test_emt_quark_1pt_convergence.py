import csv

import h5py
import numpy as np
import pytest

from application.analysis_helper.emt_quark_1pt_convergence import (
    analyze_quark_1pt_convergence,
    cumulative_statistics,
    load_quark_1pt_base_series,
)


def _write_emt(path, incomplete=False):
    values = np.asarray([1.5, 3.5])
    with h5py.File(path, "w") as h5:
        h5.attrs.update(dict(output_kind="emt_complete_base_statistics_v2",
            shard_schema="disconnected_1pt_base_parts_v4", emt_operator_schema_version=5,
            base_values_volume_normalized=True, n_base_noise=2, vectors_per_base=2,
            spin_color_dilution="none", effective_n_inversions=4,
            qext=[[0, 0, 0, 0]], flow_times=[0., 0.2], global_lattice_size=[2, 2, 2, 3]))
        h5.create_dataset("base/base_noise_index", data=[0, 0] if incomplete else [0, 1])
        h5.create_dataset("base/Tmunu/T44", data=np.broadcast_to(values[:, None, None, None], (2, 1, 2, 3)))
        h5.create_dataset("base/ringed/kinetic_spacetime", data=np.broadcast_to(10*values[:, None], (2, 2)))


def test_loads_complete_base_ringed_and_t44(tmp_path):
    path = tmp_path / "emt.h5"
    _write_emt(path)
    series = load_quark_1pt_base_series(path, label="HP2", flow_index=1)
    np.testing.assert_allclose(series.ringed_base_values, [15, 35])
    np.testing.assert_allclose(series.emt_base_values, [1.5, 3.5])
    assert series.solves_per_base == 2
    assert series.qext == (0, 0, 0, 0)
    assert series.component == "T44"

    stats = cumulative_statistics(series.emt_base_values, series.solves_per_base)
    np.testing.assert_array_equal(stats.solves, [2, 4])
    np.testing.assert_allclose(stats.mean, [1.5, 2.5])
    np.testing.assert_allclose(stats.sem_real[1], 1.0)


def test_rejects_partial_hp_base(tmp_path):
    path = tmp_path / "bad.h5"
    _write_emt(path, incomplete=True)
    with pytest.raises(ValueError, match="contiguous"):
        load_quark_1pt_base_series(path)


def test_analysis_writes_two_figures_and_tables(tmp_path):
    path = tmp_path / "emt.h5"
    _write_emt(path)
    output = tmp_path / "analysis"
    analyze_quark_1pt_convergence([("HP2", path)], output, flow_index=1)
    assert (output / "ringed_kinetic_convergence.png").is_file()
    assert (output / "t44_convergence.pdf").is_file()
    with (output / "endpoint_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["observable"] for row in rows] == ["ringed_kinetic", "T44"]
    assert all(row["solves"] == "4" for row in rows)
