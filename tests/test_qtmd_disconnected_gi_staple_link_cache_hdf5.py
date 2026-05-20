from pathlib import Path
from unittest import SkipTest

import h5py
import numpy as np


DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "application"
    / "qTMD_disconnected_1pt"
    / "perlmutter"
    / "data"
    / "qTMD1pt"
)

DEFAULT_FILES = {
    "direct": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_qTMD_bz2_eta1_bT1.direct_covdev.h5",
    "cache": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_qTMD_bz2_eta1_bT1.link_cache.h5",
}
TEST_REQUIRES = "external_hdf5"


def _read_required_outputs():
    missing = [str(path) for path in DEFAULT_FILES.values() if not path.exists()]
    if missing:
        raise SkipTest("GI qTMD link-cache HDF5 comparison files are not present: " + ", ".join(missing))

    outputs = {}
    for kind, path in DEFAULT_FILES.items():
        with h5py.File(path, "r") as h5:
            outputs[kind] = {
                "mode": h5.attrs["gi_qtmd_staple_mode"],
                "raw": h5["raw/loop_pervec"][...],
                "w_index": h5["W_index_list"][...],
                "momentum": h5["momentum_list"][...],
            }
    return outputs


def test_disconnected_gi_qtmd_link_cache_hdf5_matches_direct_covdev():
    outputs = _read_required_outputs()

    assert outputs["direct"]["mode"] == "direct_covdev"
    assert outputs["cache"]["mode"] == "link_cache"
    np.testing.assert_array_equal(outputs["direct"]["w_index"], outputs["cache"]["w_index"])
    np.testing.assert_array_equal(outputs["direct"]["momentum"], outputs["cache"]["momentum"])
    np.testing.assert_allclose(outputs["direct"]["raw"], outputs["cache"]["raw"], atol=5e-13, rtol=1e-13)


if __name__ == "__main__":
    try:
        out = _read_required_outputs()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)

    diff = np.abs(out["direct"]["raw"] - out["cache"]["raw"])
    print("[qTMD disconnected GI staple link-cache HDF5 sanity check]")
    print(f"raw maxdiff: {np.max(diff):.16e}")
    print(f"raw meandiff: {np.mean(diff):.16e}")
    raise SystemExit(0 if np.max(diff) <= 5e-13 else 1)
