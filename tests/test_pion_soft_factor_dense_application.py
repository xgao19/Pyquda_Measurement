"""Exercise the active saved-propagator application against the dense API."""

from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace

import h5py
import numpy as np


def test_contract_application_writes_ordered_collections_with_time_wrap(tmp_path, monkeypatch):
    import pyquda
    import pyquda_utils
    from pyquda_measurement_utils import pion_soft_factor_vibe_develop as soft

    info = SimpleNamespace(global_size=[8] * 4, size=[8] * 4, mpi_rank=0, t_boundary=-1)
    comm = SimpleNamespace(Get_rank=lambda: 0, Barrier=lambda: None)
    monkeypatch.setattr(pyquda, "init", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pyquda, "getMPIComm", lambda: comm)
    monkeypatch.setattr(pyquda_utils, "io", SimpleNamespace(
        readNERSCGauge=lambda *args, **kwargs: SimpleNamespace(latt_info=info)), raising=False)
    monkeypatch.setenv("PION_SOFT_LAT_TAG", "testlat")
    monkeypatch.setenv("PION_SOFT_SM_TAG", "1HYP_wall")

    def load(self, tag):
        t = int(re.search(r"\.x0y0z0t(\d+)\.", tag).group(1))
        p = int(re.search(r"\.qx0qy0qz(-?\d+)$", tag).group(1))
        return SimpleNamespace(t=t, p=p)

    def ss(self, lattice, fw, bw, kf, kb, label):
        assert fw.t == bw.t == 7 and fw.p == kf[2] and bw.p == -kb[2]
        return np.arange(8) + 100 * fw.p - bw.p

    def sp(self, lattice, fw, bw, momentum, label):
        assert fw.p - bw.p == momentum[2]
        return np.ones((16, 12, 8), complex) * (100 * fw.p - bw.p) + np.arange(8)

    def c4(self, lattice, fw, bw, sink_bw, sink_fw, momentum):
        assert sink_bw.p == -bw.p and sink_fw.p == -fw.p
        assert sink_bw.t == sink_fw.t and sink_bw.t in (1, 3)
        assert fw.p - bw.p == momentum[2]
        return np.ones((1, 6, 2, 3, 8), complex) * (100 * fw.p - bw.p) + np.arange(8), None, None

    monkeypatch.setattr(soft.pion_soft_factor, "load_wall_propagator", load)
    monkeypatch.setattr(soft.pion_soft_factor, "contract_wall_2pt", ss)
    monkeypatch.setattr(soft.pion_soft_factor, "contract_tmdwf_check", sp)
    monkeypatch.setattr(soft.pion_soft_factor, "contract_soft_factor", c4)
    script = Path(__file__).resolve().parents[1] / "application/pion_soft_factor/perlmutter/Pyquda_pion_soft_factor_contract.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--data_dir", str(tmp_path), "--config_num", "12",
                                      "--t_start", "7", "--t_count", "1", "--bT_dir", "0,1",
                                      "--bT_length", "2", "--bz_length", "1", "--tsep_list", "2,4"])
    runpy.run_path(str(script), run_name="__main__")
    paths = list(tmp_path.glob("*/*/*.h5"))
    assert len(paths) == 4
    for path in paths:
        assert ".wallt7." in path.name and ".qz4_5" in path.name
        with h5py.File(path, "r") as handle:
            np.testing.assert_array_equal(handle["meson_momentum_list"][:, 2], [8, 9, 9, 10])
            assert handle.attrs["soft_factor_storage_schema"] == "pion_soft_factor_dense_v1"
            if "C4" in handle:
                assert handle["C4"].shape == (4, 6, 6, 8)
                assert handle["C4"][1, 0, 0, 0] == 412
                assert handle["C4"][2, 0, 0, 0] == 511
            elif "SP" in handle:
                assert handle["SP"].shape == (4, 16, 12, 8)
                assert handle["displacement_vectors"].shape == (12, 3)
            else:
                assert handle["SS"].shape == (4, 1, 8)
