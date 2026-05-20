from pathlib import Path
from unittest import SkipTest

import numpy as np

TEST_REQUIRES = "gpu"


def _get_xp(arr):
    module = type(arr).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _run_link_cache_check():
    try:
        from pyquda import init
        from pyquda.field import LatticeFermion
        from pyquda_utils import io
        from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
            build_gi_qtmd_staple_link,
            create_fermion_TMD_GI,
            create_fermion_TMD_GI_from_link,
        )
        from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
    except Exception as err:
        raise SkipTest(f"PyQUDA environment is not available: {err}") from err

    try:
        init([1, 1, 1, 1], enable_mps=True)
    except Exception as err:
        raise SkipTest(f"PyQUDA could not initialize: {err}") from err

    gauge_path = (
        Path(__file__).resolve().parents[1]
        / "test_gauge"
        / "S8T32_wilson_b6.cg.1e-08.0"
    )
    if not gauge_path.exists():
        raise SkipTest(f"test gauge is not present: {gauge_path}")

    gauge = io.readNERSCGauge(str(gauge_path))
    gauge.gauge_dirac.loadGauge(gauge)

    fermion = LatticeFermion(gauge.latt_info)
    xp = _get_xp(fermion.data)
    values = xp.arange(fermion.data.size, dtype=xp.float64).reshape(fermion.data.shape)
    fermion.data[:] = (values % 17) + 1j * (values % 23)

    cases = [
        [0, 2, 1, 0],
        [0, -2, 1, 1],
        [1, 2, 1, 0],
        [1, -2, 1, 1],
    ]
    for W_index in cases:
        direct = create_fermion_TMD_GI(gauge, fermion, W_index)
        staple_link = build_gi_qtmd_staple_link(gauge, W_index)
        from_link = create_fermion_TMD_GI_from_link(staple_link, fermion, W_index)
        np.testing.assert_allclose(array_to_numpy(from_link.data), array_to_numpy(direct.data), atol=1e-12, rtol=1e-12)


def test_gi_qtmd_link_cache_matches_direct_covdev():
    _run_link_cache_check()


if __name__ == "__main__":
    try:
        _run_link_cache_check()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)
    print("[GI qTMD link-cache sanity check]")
    print("cached staple transporters match direct covDev on the S8T32 test gauge")
