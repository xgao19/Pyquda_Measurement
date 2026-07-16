import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for connected qTMD HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import create_gi_qtmd_wilsonline_index_lists
from pyquda_measurement_utils.io_corr import save_qTMD_pion_hdf5_noRoll


class DummyLatticeInfo:
    mpi_rank = 0


def _qmax1_momentum_list():
    return [[x, y, 0, 0] for x in range(-1, 2) for y in range(-1, 2)]


def _nonzero_staple_index_list():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([1], 2, 1)
    return dir0 + dir1


def test_connected_gi_qtmd_qmax1_hdf5_layout_and_metadata(tmp_path):
    gammalist = ["5", "T"]
    plist = _qmax1_momentum_list()
    w_index_list = _nonzero_staple_index_list()
    tsep = 2
    corr = np.arange(len(w_index_list) * len(plist) * len(gammalist) * (tsep + 2), dtype=np.float64)
    corr = corr.reshape(len(w_index_list), len(plist), len(gammalist), tsep + 2).astype(np.complex128)
    tag = str(tmp_path / "qTMD" / "S8T32.qTMD.0.GI_qTMD.ex.x0y0z0t0.schema")
    attrs = {
        "src_interpolator": "fixed_g5",
        "sink_interpolator": "5",
        "operator_gamma": "T",
        "staple_convention": "fixed_length",
        "staple_mode": "link_cache",
        "qmax": 1,
        "eta": 1,
        "b_z_max": 2,
        "b_T_max": 1,
    }

    save_qTMD_pion_hdf5_noRoll(corr, tag, gammalist, plist, w_index_list, tsep, DummyLatticeInfo(), attrs=attrs)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["staple_convention"] == "fixed_length"
        assert h5.attrs["staple_mode"] == "link_cache"
        assert h5.attrs["qmax"] == 1
        assert h5.attrs["src_interpolator"] == "fixed_g5"
        assert h5.attrs["sink_interpolator"] == "5"
        assert h5.attrs["operator_gamma"] == "T"

        datasets = []
        h5.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        assert len(datasets) == len(gammalist) * len(plist) * len(w_index_list)

        expected = "SS/5/PX-1PY-1PZ0/b_X/eta1/bT0/bz-2"
        assert expected in datasets
        assert h5[expected].shape == (tsep + 2,)
        expected_widx = w_index_list.index([0, -2, 1, 0])
        np.testing.assert_array_equal(h5[expected][...], corr[expected_widx, 0, 0, :])

        assert "SS/T/PX1PY1PZ0/b_Y/eta1/bT1/bz2" in datasets


def test_connected_gi_qtmd_qmax1_layout_has_expected_momentum_and_wilson_counts():
    plist = _qmax1_momentum_list()
    w_index_list = _nonzero_staple_index_list()

    assert len(plist) == 9
    assert {tuple(p[:3]) for p in plist} == {(x, y, 0) for x in range(-1, 2) for y in range(-1, 2)}
    assert len(w_index_list) == 12
    assert [0, -2, 1, 0] in w_index_list
    assert [1, 2, 1, 1] in w_index_list
