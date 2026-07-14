import csv

import h5py
import numpy as np
import pytest

from application.analysis_helper.emt_quark_1pt_convergence import (
    analyze_quark_1pt_convergence,
    cumulative_statistics,
    load_quark_1pt_base_series,
)
from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS


def _write_emt(path, hp_index=(0, 1, 0, 1)):
    n_source = len(hp_index)
    volume = 8
    target_t44 = np.arange(1, n_source + 1, dtype=np.float64)
    kinetic = (10 * target_t44)[:, None, None] * np.ones((n_source, 2, 3))
    derivative = np.zeros((n_source, 16, 4, 1, 2, 3), dtype=np.complex128)
    gamma_t = GAMMA_LABELS.index("T")
    derivative[:, gamma_t, 3, 0, 1, :] = target_t44[:, None] * volume
    with h5py.File(path, "w") as h5:
        h5.attrs["emt_operator_schema_version"] = 3
        h5.attrs["qext"] = [[0, 0, 0, 0]]
        h5.attrs["flow_times"] = [0.0, 0.2]
        h5.attrs["noise_scheme"] = "hierarchical_probing"
        h5.attrs["hp_num_vectors"] = 2
        h5.attrs["volume_norm"] = volume
        h5.create_dataset("gamma_list", data=np.asarray(GAMMA_LABELS, dtype="S"))
        raw = h5.require_group("raw")
        raw.create_dataset("derivative_bilinear_pervec", data=derivative)
        raw.create_dataset("base_noise_index", data=np.repeat(np.arange(n_source // 2), 2))
        raw.create_dataset("hp_index", data=np.asarray(hp_index, dtype=np.int32))
        ringed = h5.require_group("derived/ringed")
        ringed.create_dataset("kinetic_pervec", data=kinetic)


def test_loads_complete_base_ringed_and_t44(tmp_path):
    path = tmp_path / "emt.h5"
    _write_emt(path)
    series = load_quark_1pt_base_series(path, label="HP2", flow_index=1)
    np.testing.assert_allclose(series.ringed_base_values, [15, 35])
    np.testing.assert_allclose(series.emt_base_values, [1.5, 3.5])
    assert series.hp_count == 2
    assert series.qext == (0, 0, 0, 0)
    assert series.component == "T44"

    stats = cumulative_statistics(series.emt_base_values, series.hp_count)
    np.testing.assert_array_equal(stats.solves, [2, 4])
    np.testing.assert_allclose(stats.mean, [1.5, 2.5])
    np.testing.assert_allclose(stats.sem_real[1], 1.0)


def test_rejects_partial_hp_base(tmp_path):
    path = tmp_path / "bad.h5"
    _write_emt(path, hp_index=(0, 0, 0, 1))
    with pytest.raises(ValueError, match="incomplete"):
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
