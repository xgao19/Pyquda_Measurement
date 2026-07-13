import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for EMT HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.io_corr import (
    save_emt_gluon_1pt_hdf5,
    save_emt_meson_2pt_hdf5,
    save_emt_quark_3pt_hdf5,
)


def test_emt_gluon_1pt_hdf5_schema_and_upper_triangle(tmp_path):
    tag = str(tmp_path / "EMTgluon1pt" / "schema")
    tmunu_t = np.arange(4 * 4 * 5, dtype=np.float64).reshape(4, 4, 5)

    attrs = {
        "measurement": "EMT_gluon_1pt",
        "config_num": 17,
        "flow_epsilon": 0.207936,
        "flow_times": np.asarray([0.0, 0.207936]),
    }
    save_emt_gluon_1pt_hdf5(tag, tmunu_t, attrs=attrs)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["measurement"] == "EMT_gluon_1pt"
        assert h5.attrs["config_num"] == 17
        assert h5.attrs["flow_epsilon"] == 0.207936
        np.testing.assert_array_equal(h5.attrs["flow_times"], [0.0, 0.207936])
        assert bool(h5["Tmunu"].attrs["upper_triangle_only"])
        assert h5["Tmunu/T22"].shape == (5,)
        assert "Tmunu/T21" not in h5


def test_emt_meson_2pt_and_quark_3pt_schema(tmp_path):
    c2_tag = str(tmp_path / "EMTmeson2pt" / "schema")
    c2 = np.zeros((16, 3, 8), dtype=np.complex128)
    gamma_list = ["5", "T"]
    momentum_list = [[0, 0, 0, 0], [1, 0, 0, 0]]

    save_emt_meson_2pt_hdf5(c2_tag, c2, gamma_list, momentum_list, attrs={"measurement": "EMT_meson_2pt"})

    with h5py.File(c2_tag + ".h5", "r") as h5:
        assert h5.attrs["measurement"] == "EMT_meson_2pt"
        assert h5["C2"].shape == c2.shape
        assert [item.decode() for item in h5["gamma_list"][...]] == gamma_list
        np.testing.assert_array_equal(h5["momentum_list"][...], np.asarray(momentum_list, dtype=np.int32))

    c3_tag = str(tmp_path / "EMTquark3pt" / "schema")
    c3_chi = np.zeros((2, 3, 4), dtype=np.complex128)
    c3_tmunu = np.zeros((2, 3, 4, 4, 4), dtype=np.complex128)
    qlist = [[0, 0, 0, 0], [1, 0, 0, 0]]
    save_emt_quark_3pt_hdf5(
        c3_tag,
        c3_chi,
        c3_tmunu,
        momentum_transfer_list=qlist,
        attrs={
            "measurement": "EMT_quark_3pt",
            "operator_normalization": "unringed_flowed_bilinear",
            "ringed_normalization_applied": False,
        },
    )

    with h5py.File(c3_tag + ".h5", "r") as h5:
        assert h5.attrs["measurement"] == "EMT_quark_3pt"
        assert h5.attrs["operator_normalization"] == "unringed_flowed_bilinear"
        assert not h5.attrs["ringed_normalization_applied"]
        assert "C2" not in h5
        assert h5["C3_chi"].shape == c3_chi.shape
        assert h5["C3_Tmunu"].shape == c3_tmunu.shape
        np.testing.assert_array_equal(h5["momentum_transfer_list"][...], np.asarray(qlist, dtype=np.int32))
