from unittest import SkipTest

import numpy as np


def _get_xp(arr):
    module = type(arr).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _run_identity_gauge_check():
    try:
        from pyquda import init
        from pyquda.field import LatticeFermion, LatticeGauge, LatticeInfo
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

    latt_info = LatticeInfo([4, 4, 4, 8])
    gauge = LatticeGauge(latt_info)
    gauge.gauge_dirac.loadGauge(gauge)

    fermion = LatticeFermion(latt_info)
    xp = _get_xp(fermion.data)
    values = xp.arange(fermion.data.size, dtype=xp.float64).reshape(fermion.data.shape)
    fermion.data[:] = values + 1j * (values + 1)

    cases = [
        [0, 0, 0, 0],
        [0, 2, 1, 0],
        [0, -2, 1, 1],
        [3, 4, 3, 0],
        [2, -4, 3, 1],
    ]
    for b_T, b_z, eta, transverse_direction in cases:
        actual = create_fermion_TMD_GI(gauge, fermion, [b_T, b_z, eta, transverse_direction])
        staple_link = build_gi_qtmd_staple_link(gauge, [b_T, b_z, eta, transverse_direction])
        actual_from_link = create_fermion_TMD_GI_from_link(staple_link, fermion, [b_T, b_z, eta, transverse_direction])
        expected = fermion.shift(b_T, transverse_direction).shift(b_z, 2)
        np.testing.assert_allclose(array_to_numpy(actual.data), array_to_numpy(expected.data), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(array_to_numpy(actual_from_link.data), array_to_numpy(actual.data), atol=1e-12, rtol=1e-12)


def test_gi_qtmd_identity_gauge_matches_coordinate_shift():
    _run_identity_gauge_check()


if __name__ == "__main__":
    try:
        _run_identity_gauge_check()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)
    print("[GI qTMD identity-gauge sanity check]")
    print("GI staple covariant shifts match CG coordinate shifts on unit gauge")
