import h5py
import numpy as np

from pyquda_measurement_utils.pion_current_background_response_vibe_develop import (
    save_pion_EMFF_background_response_hdf5,
)


def test_background_response_hdf5_schema_v3_records_relative_and_absolute_tau(tmp_path):
    tag = tmp_path / "pion_EMFF_background_response" / "schema"
    records = [
        {
            "current_gamma": "T",
            "sink_gamma": "5",
            "src_gamma": "5",
            "tau_window": "restricted",
            "tau_min": 2,
            "tau_relative_list": [2, 3, 4],
            "tau_absolute_list": [7, 0, 1],
            "response_sign": 1,
            "finite_difference_derivative_sign": -1,
            "pf": [0, 0, 1],
            "qext": [0, 0, 2],
            "pi": [0, 0, -1],
            "tsep": 6,
            "q_index": 0,
            "c2_tsep": 2 + 0j,
            "explicit_summed_c3": 4 + 0j,
            "response_c2_like": 4 + 1e-14j,
            "response_R_sum": 2 + 5e-15j,
            "explicit_R_sum": 2 + 0j,
            "difference": 1e-14j,
            "relative_difference": 2.5e-15,
            "explicit_c3_all_tau": np.arange(8),
            "response_corr_all_t": np.arange(8) + 1,
            "c2_all_t": np.arange(8) + 2,
        }
    ]

    save_pion_EMFF_background_response_hdf5(
        str(tag),
        records,
        attrs={
            "lat_tag": "S8T32",
            "no_per_tau_response_propagator_cache": True,
            "current_gamma_list": np.asarray(["T", "Z"], dtype="S"),
            "source_gamma_label": "5",
            "source_gamma_mode": "fixed",
            "source_position": np.asarray([1, 2, 3, 5], dtype=np.int32),
            "source_time": 5,
        },
    )

    with h5py.File(f"{tag}.h5", "r") as h5:
        assert h5.attrs["measurement"] == "pion_EMFF_background_response"
        assert h5.attrs["schema_version"] == "3"
        assert h5.attrs["time_axis"] == "source_relative"
        assert h5.attrs["no_per_tau_response_propagator_cache"]
        assert h5.attrs["source_gamma_label"] == "5"
        assert h5.attrs["source_gamma_mode"] == "fixed"
        summary = h5["summary"]
        np.testing.assert_array_equal(summary["record_index"][:], [0])
        assert summary["current_gamma"][0].decode() == "T"
        assert summary["tau_window"][0].decode() == "restricted"
        np.testing.assert_array_equal(summary["pf"][:], [[0, 0, 1]])
        np.testing.assert_array_equal(summary["qext"][:], [[0, 0, 2]])
        np.testing.assert_array_equal(summary["pi"][:], [[0, 0, -1]])
        np.testing.assert_array_equal(summary["tsep"][:], [6])
        np.testing.assert_allclose(summary["relative_difference"][:], [2.5e-15])
        np.testing.assert_allclose(summary["response_R_sum"][:], [2 + 5e-15j])
        np.testing.assert_allclose(summary["explicit_R_sum"][:], [2 + 0j])

        record = h5["results/record_0000"]
        assert record.attrs["current_gamma"] == "T"
        assert record.attrs["tau_window"] == "restricted"
        np.testing.assert_array_equal(record["pf"][:], [0, 0, 1])
        np.testing.assert_array_equal(record["qext"][:], [0, 0, 2])
        np.testing.assert_array_equal(record["pi"][:], [0, 0, -1])
        np.testing.assert_array_equal(record["tau_relative_list"][:], [2, 3, 4])
        np.testing.assert_array_equal(record["tau_absolute_list"][:], [7, 0, 1])
        assert not record.attrs["tau_list_is_all_time_slices"]
        assert record["tsep"][()] == 6
        np.testing.assert_allclose(record["response_R_sum"][()], 2 + 5e-15j)
