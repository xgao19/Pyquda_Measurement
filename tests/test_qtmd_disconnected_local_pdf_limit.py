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
    "GI_PDF": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_PDF.h5",
    "CG_PDF": DATA_DIR / "S8T32.qTMD1pt.local_check.CG_PDF.h5",
    "CG_qTMD": DATA_DIR / "S8T32.qTMD1pt.local_check.CG_qTMD.h5",
}


def _read_required_outputs():
    missing = [str(path) for path in DEFAULT_FILES.values() if not path.exists()]
    if missing:
        raise SkipTest("local/PDF sanity-check HDF5 files are not present: " + ", ".join(missing))

    outputs = {}
    for kind, path in DEFAULT_FILES.items():
        with h5py.File(path, "r") as h5:
            outputs[kind] = {
                "operator_kind": h5.attrs["operator_kind"],
                "raw": h5["raw/loop_pervec"][...],
                "w_index": h5["W_index_list"][...],
                "momentum": h5["momentum_list"][...],
                "gamma5_bx": h5["avg/SS/5/PX0PY0PZ0/b_X/eta0/bT0/bz0"][...],
            }
            if kind == "CG_qTMD":
                outputs[kind]["gamma5_by"] = h5["avg/SS/5/PX0PY0PZ0/b_Y/eta0/bT0/bz0"][...]
    return outputs


def test_disconnected_qtmd_local_pdf_limit():
    outputs = _read_required_outputs()

    np.testing.assert_array_equal(outputs["GI_PDF"]["w_index"], [[0, 0, 0, 0]])
    np.testing.assert_array_equal(outputs["CG_PDF"]["w_index"], [[0, 0, 0, 0]])
    np.testing.assert_array_equal(
        outputs["CG_qTMD"]["w_index"],
        [[0, 0, 0, 0], [0, 0, 0, 1]],
    )
    for kind in outputs:
        np.testing.assert_array_equal(outputs[kind]["momentum"], [[0, 0, 0, 0]])

    np.testing.assert_allclose(outputs["GI_PDF"]["raw"], outputs["CG_PDF"]["raw"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        outputs["GI_PDF"]["raw"][:, 0],
        outputs["CG_qTMD"]["raw"][:, 0],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outputs["GI_PDF"]["raw"][:, 0],
        outputs["CG_qTMD"]["raw"][:, 1],
        atol=0.0,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        outputs["GI_PDF"]["gamma5_bx"],
        outputs["CG_PDF"]["gamma5_bx"],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outputs["GI_PDF"]["gamma5_bx"],
        outputs["CG_qTMD"]["gamma5_bx"],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outputs["GI_PDF"]["gamma5_bx"],
        outputs["CG_qTMD"]["gamma5_by"],
        atol=0.0,
        rtol=0.0,
    )


if __name__ == "__main__":
    try:
        out = _read_required_outputs()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)
    checks = {
        "GI_PDF vs CG_PDF raw": np.max(np.abs(out["GI_PDF"]["raw"] - out["CG_PDF"]["raw"])),
        "GI_PDF vs CG_qTMD b_X raw": np.max(np.abs(out["GI_PDF"]["raw"][:, 0] - out["CG_qTMD"]["raw"][:, 0])),
        "GI_PDF vs CG_qTMD b_Y raw": np.max(np.abs(out["GI_PDF"]["raw"][:, 0] - out["CG_qTMD"]["raw"][:, 1])),
        "GI_PDF vs CG_PDF gamma5 avg": np.max(np.abs(out["GI_PDF"]["gamma5_bx"] - out["CG_PDF"]["gamma5_bx"])),
        "GI_PDF vs CG_qTMD b_X gamma5 avg": np.max(np.abs(out["GI_PDF"]["gamma5_bx"] - out["CG_qTMD"]["gamma5_bx"])),
        "GI_PDF vs CG_qTMD b_Y gamma5 avg": np.max(np.abs(out["GI_PDF"]["gamma5_bx"] - out["CG_qTMD"]["gamma5_by"])),
    }
    print("[qTMD disconnected local/PDF limit sanity check]")
    for label, value in checks.items():
        print(f"{label}: {value:.16e}")
    raise SystemExit(0 if all(value == 0.0 for value in checks.values()) else 1)
