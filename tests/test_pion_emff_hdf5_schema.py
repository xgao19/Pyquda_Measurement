import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for pion EMFF HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.io_corr import save_pion_EMFF_hdf5_noRoll, save_proton_c2pt_hdf5


class DummyLatticeInfo:
    mpi_rank = 0


def test_pion_emff_hdf5_schema_preserves_current_gamma_and_qext(tmp_path):
    tag = str(tmp_path / "pion_EMFF" / "S8T32.pion_EMFF.0.EMFF.ex.x0y0z0t0.schema")
    gammalist = ["T", "Z5"]
    qlist = [[0, 0, 0, 0], [1, -1, 0, 0]]
    tsep = 4
    corr = np.arange(len(qlist) * len(gammalist) * (tsep + 2), dtype=np.float64)
    corr = corr.reshape(len(qlist), len(gammalist), tsep + 2).astype(np.complex128)

    save_pion_EMFF_hdf5_noRoll(corr, tag, gammalist, qlist, tsep, DummyLatticeInfo())

    with h5py.File(tag + ".h5", "r") as h5:
        assert "SS/T/PX0PY0PZ0" in h5
        assert "SS/T/PX1PY-1PZ0" in h5
        assert "SS/Z5/PX1PY-1PZ0" in h5
        assert h5["SS/T/PX0PY0PZ0"].shape == (tsep + 2,)
        np.testing.assert_array_equal(h5["SS/Z5/PX1PY-1PZ0"][...], corr[1, 1, :])


def test_pion_emff_synthetic_ratio_plateau_is_one(tmp_path):
    tsep = 4
    c2_tag = str(tmp_path / "c2pt" / "S8T32.c2pt.0.CG.ex.x0y0z0t0.schema")
    emff_tag = str(tmp_path / "pion_EMFF" / "S8T32.pion_EMFF.0.EMFF.ex.x0y0z0t0.schema")

    c2 = np.zeros((1, 1, tsep + 3), dtype=np.complex128)
    c2[0, 0, :] = 2.5 + 0.25j
    c3 = np.zeros((1, 1, tsep + 2), dtype=np.complex128)
    c3[0, 0, :] = c2[0, 0, tsep]

    save_proton_c2pt_hdf5(c2, c2_tag, ["5"], [[0, 0, 0, 0]])
    save_pion_EMFF_hdf5_noRoll(c3, emff_tag, ["T"], [[0, 0, 0, 0]], tsep, DummyLatticeInfo())

    with h5py.File(c2_tag + ".h5", "r") as c2_h5, h5py.File(emff_tag + ".h5", "r") as c3_h5:
        c2_ts = c2_h5["SS/5/PX0PY0PZ0"][tsep]
        c3_tau = c3_h5["SS/T/PX0PY0PZ0"][1:tsep]
        ratio = c3_tau / c2_ts
        np.testing.assert_allclose(ratio, np.ones_like(ratio), atol=0, rtol=0)
