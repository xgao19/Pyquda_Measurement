import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for pion soft-factor HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.io_corr import (
    save_pion_soft_factor_c2pt_hdf5_noRoll,
    save_pion_soft_factor_hdf5_noRoll,
    save_pion_soft_factor_qTMDWF_hdf5_noRoll,
)


class DummyLatticeInfo:
    mpi_rank = 0


def test_pion_soft_factor_4pt_hdf5_schema(tmp_path):
    tag = str(tmp_path / "pion_soft_factor" / "schema")
    pion_src_keys = ["Z5-X5"]
    pion_sink_keys = ["Z5-X5"]
    gamma1_keys = ["5", "I"]
    gamma2_keys = ["5", "I"]
    bT_dir = [0, 1]
    bT_length = 2
    tseplist = [2, 4]
    corr = np.arange(
        len(tseplist) * len(pion_src_keys) * len(gamma1_keys) * len(bT_dir) * (bT_length + 1) * 4,
        dtype=np.float64,
    )
    corr = corr.reshape(len(tseplist), len(pion_src_keys), len(gamma1_keys), len(bT_dir), bT_length + 1, 4)

    save_pion_soft_factor_hdf5_noRoll(
        corr,
        tag,
        pion_src_keys,
        pion_sink_keys,
        gamma1_keys,
        gamma2_keys,
        bT_dir,
        bT_length,
        tseplist,
        DummyLatticeInfo(),
    )

    with h5py.File(tag + ".h5", "r") as h5:
        path = "srcZ5-X5_sinkZ5-X5/5_5/bX_0/ts2"
        assert path in h5
        assert h5[path].shape == (4,)
        np.testing.assert_array_equal(h5[path][...], corr[0, 0, 0, 0, 0])
        assert "srcZ5-X5_sinkZ5-X5/I_I/bY_2/ts4" in h5
        assert "srcZ5-X5_sinkZ5-X5/5_I/bX_0/ts2" not in h5
        assert "srcZ5-X5_sinkZ5-X5/I_5/bX_0/ts2" not in h5


def test_pion_soft_factor_diagnostic_c2pt_and_qtmdwf_schema(tmp_path):
    c2_tag = str(tmp_path / "pion_soft_factor_c2pt" / "schema")
    qtmdwf_tag = str(tmp_path / "pion_soft_factor_qTMDWF" / "schema")
    momentum = [0, 0, 2]

    c2 = np.arange(2 * 8, dtype=np.float64).reshape(2, 8).astype(np.complex128)
    save_pion_soft_factor_c2pt_hdf5_noRoll(c2, c2_tag, "Z5-X5", ["Z5-X5", "5"], momentum, DummyLatticeInfo())

    with h5py.File(c2_tag + ".h5", "r") as h5:
        assert "SS/Z5-X5/Z5-X5/PX0PY0PZ2" in h5
        assert "SS/Z5-X5/5/PX0PY0PZ2" in h5
        np.testing.assert_array_equal(h5["SS/Z5-X5/5/PX0PY0PZ2"][...], c2[1])

    bT_dir = [0, 1]
    bT_length = 1
    bz_length = 2
    corr_count = len(bT_dir) * (bT_length + 1) * (bz_length + 1)
    qtmdwf = np.arange(corr_count * 8, dtype=np.float64).reshape(corr_count, 8).astype(np.complex128)
    save_pion_soft_factor_qTMDWF_hdf5_noRoll(qtmdwf, qtmdwf_tag, "Z5-X5", momentum, bT_dir, bT_length, bz_length, DummyLatticeInfo())

    with h5py.File(qtmdwf_tag + ".h5", "r") as h5:
        assert "SP/Z5-X5/PX0PY0PZ2/b_X/bT0/bz0" in h5
        assert "SP/Z5-X5/PX0PY0PZ2/b_Y/bT1/bz2" in h5
        np.testing.assert_array_equal(h5["SP/Z5-X5/PX0PY0PZ2/b_X/bT0/bz0"][...], qtmdwf[0])
