import h5py
import numpy as np

from pyquda_measurement_utils.pion_current_background_response_vibe_develop import (
    save_pion_current_current_response_hdf5,
)


def test_current_current_response_hdf5_schema(tmp_path):
    tag = tmp_path / "current_current" / "schema"
    records = [
        {
            "first_current_gamma": "T",
            "second_current_gamma": "T",
            "sink_gamma": "5",
            "src_gamma": "5",
            "first_tau_window": "restricted",
            "second_tau_window": "restricted",
            "first_tau_min": 1,
            "second_tau_min": 1,
            "first_tau_relative_list": [1, 2],
            "first_tau_absolute_list": [7, 0],
            "second_tau_relative_list": [1, 2],
            "second_tau_absolute_list": [7, 0],
            "response_sign": 1,
            "pf": [0, 0, 0],
            "first_qext": [0, 0, 1],
            "second_qext": [0, 0, -1],
            "total_qext": [0, 0, 0],
            "pi": [0, 0, 0],
            "tsep": 3,
            "c2_tsep": 2 + 0j,
            "response_c2_like": 4 + 1j,
            "response_R_sum": 2 + 0.5j,
            "response_corr_all_t": np.arange(4),
            "c2_all_t": np.arange(4) + 1,
        }
    ]

    save_pion_current_current_response_hdf5(
        str(tag),
        records,
        attrs={
            "lat_tag": "S8T32",
            "source_gamma_label": "5",
            "source_gamma_mode": "fixed",
        },
    )

    with h5py.File(f"{tag}.h5", "r") as h5:
        assert h5.attrs["measurement"] == "pion_current_current_response"
        assert h5.attrs["schema_version"] == "2"
        assert h5.attrs["time_axis"] == "source_relative"
        assert h5.attrs["current_order"] == "Dinv_O2_Dinv_O1_S"
        assert h5.attrs["source_gamma_label"] == "5"
        assert h5.attrs["source_gamma_mode"] == "fixed"
        summary = h5["summary"]
        assert summary["first_current_gamma"][0].decode() == "T"
        np.testing.assert_array_equal(summary["first_qext"][:], [[0, 0, 1]])
        np.testing.assert_array_equal(summary["second_qext"][:], [[0, 0, -1]])
        np.testing.assert_array_equal(summary["total_qext"][:], [[0, 0, 0]])
        np.testing.assert_allclose(summary["response_R_sum"][:], [2 + 0.5j])

        record = h5["results/record_0000"]
        np.testing.assert_array_equal(record["first_tau_relative_list"][:], [1, 2])
        np.testing.assert_array_equal(record["first_tau_absolute_list"][:], [7, 0])
        np.testing.assert_array_equal(record["second_tau_relative_list"][:], [1, 2])
        np.testing.assert_array_equal(record["second_tau_absolute_list"][:], [7, 0])
        assert not record.attrs["first_tau_list_is_all_time_slices"]
        assert not record.attrs["second_tau_list_is_all_time_slices"]
