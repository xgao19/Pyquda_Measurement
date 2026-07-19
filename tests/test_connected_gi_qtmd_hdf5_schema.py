import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np
import pytest

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for connected qTMD HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.qtmd_operator_utils import (
    create_gi_qtmd_wilsonline_index_lists,
)
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_BASIS_SCHEMA,
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
)
from pyquda_measurement_utils.io_corr import save_connected_qtmd_hdf5
import pyquda_measurement_utils.io_corr as io_corr


def _qmax1_momentum_list():
    return [[x, y, 0, 0] for x in range(-1, 2) for y in range(-1, 2)]


def _nonzero_staple_index_list():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([1], 2, 1)
    return dir0 + dir1


def test_connected_gi_qtmd_qmax1_hdf5_layout_and_metadata(tmp_path):
    plist = _qmax1_momentum_list()
    w_index_list = _nonzero_staple_index_list()
    tsep = 2
    corr = np.arange(
        len(w_index_list)
        * len(plist)
        * len(GAMMA_LABELS)
        * (tsep + 2),
        dtype=np.float64,
    )
    corr = corr.reshape(
        len(w_index_list), len(plist), len(GAMMA_LABELS), tsep + 2
    ).astype(np.complex128)
    tag = str(tmp_path / "qTMD" / "S8T32.qTMD.0.GI_qTMD.ex.x0y0z0t0.schema")
    attrs = {
        "src_interpolator": "5",
        "sink_interpolator": "5",
        "operator_gamma_basis": "all_16",
        "source_gamma_label": "5",
        "source_gamma_mode": "fixed",
        "staple_convention": "fixed_length",
        "staple_mode": "link_cache",
        "qmax": 1,
        "eta": 1,
        "b_z_max": 2,
        "b_T_max": 1,
    }

    save_connected_qtmd_hdf5(
        corr, tag, plist, w_index_list, tsep, attrs=attrs
    )

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["qtmd_hdf5_schema"] == "connected_qtmd_dense_v1"
        assert h5.attrs["corr_axes"] == "wilson,momentum,gamma,time"
        assert h5.attrs["wilson_index_columns"] == (
            "bT,bz,eta,transverse_direction"
        )
        assert h5.attrs["t_separation"] == tsep
        assert h5.attrs["gamma_basis_schema"] == GAMMA_BASIS_SCHEMA
        assert h5.attrs["staple_convention"] == "fixed_length"
        assert h5.attrs["staple_mode"] == "link_cache"
        assert h5.attrs["qmax"] == 1
        assert h5.attrs["src_interpolator"] == "5"
        assert h5.attrs["sink_interpolator"] == "5"
        assert h5.attrs["operator_gamma_basis"] == "all_16"
        assert h5.attrs["source_gamma_label"] == "5"
        assert h5.attrs["source_gamma_mode"] == "fixed"

        assert "SS" not in h5
        np.testing.assert_array_equal(h5["corr"][...], corr)
        np.testing.assert_array_equal(h5["momentum_list"][...], plist)
        np.testing.assert_array_equal(
            h5["wilson_index_list"][...], w_index_list
        )
        assert [
            label.decode() for label in h5["gamma_list"][...]
        ] == list(GAMMA_LABELS)
        np.testing.assert_array_equal(
            h5["gamma_pyquda_ids"][...], PYQUDA_GAMMA_IDS
        )


def test_connected_qtmd_dense_writer_rejects_partial_gamma_axis(tmp_path):
    tag = str(tmp_path / "partial")
    with pytest.raises(ValueError, match="gamma"):
        save_connected_qtmd_hdf5(
            np.zeros((1, 1, 1, 4), dtype=np.complex128),
            tag,
            [[0, 0, 0, 0]],
            [[0, 0, 0, 0]],
            2,
        )


def test_legacy_connected_qtmd_writers_are_removed():
    assert not hasattr(io_corr, "save_qTMD_proton_hdf5_noRoll")
    assert not hasattr(io_corr, "save_qTMD_pion_hdf5_noRoll")


def test_connected_gi_qtmd_qmax1_layout_has_expected_momentum_and_wilson_counts():
    plist = _qmax1_momentum_list()
    w_index_list = _nonzero_staple_index_list()

    assert len(plist) == 9
    assert {tuple(p[:3]) for p in plist} == {(x, y, 0) for x in range(-1, 2) for y in range(-1, 2)}
    assert len(w_index_list) == 12
    assert [0, -2, 1, 0] in w_index_list
    assert [1, 2, 1, 1] in w_index_list
