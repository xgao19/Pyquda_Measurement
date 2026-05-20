from pathlib import Path
from unittest import SkipTest

import h5py
import numpy as np


DATA_ROOT = Path("/tmp/pyquda_connected_gi_qtmd_consistency")
TEST_REQUIRES = "external_hdf5"


def _h5_files(label, mode):
    base = DATA_ROOT / label / mode / "qTMD"
    if not base.exists():
        raise SkipTest(f"Missing connected GI qTMD output directory: {base}")
    files = sorted(base.glob("*.h5"))
    if not files:
        raise SkipTest(f"No HDF5 outputs found under {base}")
    return files


def _compare_h5_files(left, right):
    with h5py.File(left, "r") as h5_left, h5py.File(right, "r") as h5_right:
        left_datasets = []
        h5_left.visititems(lambda name, obj: left_datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        right_datasets = []
        h5_right.visititems(lambda name, obj: right_datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        assert left_datasets == right_datasets

        for name in left_datasets:
            np.testing.assert_allclose(h5_left[name][...], h5_right[name][...], atol=1e-12, rtol=1e-12)


def _test_label(label):
    cache_files = _h5_files(label, "link_cache")
    direct_files = _h5_files(label, "direct_covdev")
    assert [path.name for path in cache_files] == [path.name for path in direct_files]

    for cache_file, direct_file in zip(cache_files, direct_files):
        _compare_h5_files(cache_file, direct_file)


def test_pion_connected_gi_qtmd_link_cache_matches_direct_covdev():
    _test_label("pion")


def test_nucleon_connected_gi_qtmd_link_cache_matches_direct_covdev():
    _test_label("nucleon")


if __name__ == "__main__":
    for label in ("pion", "nucleon"):
        try:
            _test_label(label)
        except SkipTest as err:
            print(f"SKIP {label}: {err}")
            continue
        print(f"PASS {label}: link_cache matches direct_covdev")
