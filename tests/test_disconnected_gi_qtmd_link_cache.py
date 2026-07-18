import os
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
            create_fermion_TMD_GI_from_link,
        )
        from qtmd_gi_reference import create_fermion_TMD_GI
        from pyquda_measurement_utils.Disconnected_utils_vibe_develop import array_to_numpy
    except Exception as err:
        raise SkipTest(f"PyQUDA environment is not available: {err}") from err

    try:
        geometry = [
            int(value)
            for value in os.environ.get(
                "PYQUDA_GI_QTMD_MPI_GEOMETRY", "1.1.1.1"
            ).split(".")
        ]
        init(geometry, enable_mps=False)
    except Exception as err:
        raise SkipTest(f"PyQUDA could not initialize: {err}") from err

    gauge_path = (
        Path(__file__).resolve().parents[1]
        / "test_gauge"
        / "S8T8_wilson_b6.0"
    )
    if not gauge_path.exists():
        raise SkipTest(f"test gauge is not present: {gauge_path}")

    gauge = io.readNERSCGauge(str(gauge_path))
    gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
    gauge.gauge_dirac.loadGauge(gauge)

    fermion = LatticeFermion(gauge.latt_info)
    xp = _get_xp(fermion.data)
    for spin in range(4):
        for color in range(3):
            fermion.data[..., spin, color] = (
                1 + 3 * spin + color
            ) + 1j * (2 + spin + 5 * color)

    cases = [
        [0, 2, 1, 0],
        [0, -2, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 2, 2, 0],
        [1, -2, 2, 1],
    ]
    for W_index in cases:
        direct = create_fermion_TMD_GI(gauge, fermion, W_index)
        staple_link = build_gi_qtmd_staple_link(gauge, W_index)
        from_link = create_fermion_TMD_GI_from_link(staple_link, fermion, W_index)
        np.testing.assert_allclose(array_to_numpy(from_link.data), array_to_numpy(direct.data), atol=1e-12, rtol=1e-12)

    for b_z in (-2, 2):
        pdf = fermion.copy()
        direction = 2 if b_z > 0 else 6
        for _ in range(abs(b_z)):
            pdf = gauge.pure_gauge.covDev(pdf, direction)
        w_index = [0, b_z, abs(b_z) // 2, 0]
        staple_link = build_gi_qtmd_staple_link(gauge, w_index)
        qtmd_pdf_limit = create_fermion_TMD_GI_from_link(
            staple_link, fermion, w_index
        )
        np.testing.assert_allclose(
            array_to_numpy(qtmd_pdf_limit.data),
            array_to_numpy(pdf.data),
            atol=1e-12,
            rtol=1e-12,
        )


def test_gi_qtmd_link_cache_matches_direct_covdev():
    _run_link_cache_check()


if __name__ == "__main__":
    try:
        _run_link_cache_check()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)
    print("[GI qTMD link-cache sanity check]")
    print("cached staple transporters match direct covDev on the HYP-smeared S8T8 test gauge")
