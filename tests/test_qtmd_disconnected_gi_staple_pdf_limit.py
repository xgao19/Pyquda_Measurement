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
    "GI_PDF": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_PDF_bz2.h5",
    "GI_qTMD": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_qTMD_bz2_eta1_bT0.h5",
}


def _read_required_outputs():
    missing = [str(path) for path in DEFAULT_FILES.values() if not path.exists()]
    if missing:
        raise SkipTest("GI qTMD/PDF-limit sanity-check HDF5 files are not present: " + ", ".join(missing))

    outputs = {}
    for kind, path in DEFAULT_FILES.items():
        with h5py.File(path, "r") as h5:
            outputs[kind] = {
                "operator_kind": h5.attrs["operator_kind"],
                "raw": h5["raw/loop_pervec"][...],
                "w_index": h5["W_index_list"][...].tolist(),
                "momentum": h5["momentum_list"][...],
            }
    return outputs


def _w_index_position(outputs, kind, w_index):
    return outputs[kind]["w_index"].index(w_index)


def _maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def test_disconnected_gi_qtmd_staple_pdf_limit():
    outputs = _read_required_outputs()

    assert outputs["GI_PDF"]["operator_kind"] == "GI_PDF"
    assert outputs["GI_qTMD"]["operator_kind"] == "GI_qTMD"
    np.testing.assert_array_equal(outputs["GI_PDF"]["momentum"], [[0, 0, 0, 0]])
    np.testing.assert_array_equal(outputs["GI_qTMD"]["momentum"], [[0, 0, 0, 0]])

    for bz in (2, -2):
        pdf_pos = _w_index_position(outputs, "GI_PDF", [0, bz, 0, 0])
        tmd_x_pos = _w_index_position(outputs, "GI_qTMD", [0, bz, 1, 0])
        tmd_y_pos = _w_index_position(outputs, "GI_qTMD", [0, bz, 1, 1])

        np.testing.assert_allclose(
            outputs["GI_PDF"]["raw"][:, pdf_pos],
            outputs["GI_qTMD"]["raw"][:, tmd_x_pos],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            outputs["GI_PDF"]["raw"][:, pdf_pos],
            outputs["GI_qTMD"]["raw"][:, tmd_y_pos],
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    try:
        out = _read_required_outputs()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)

    print("[qTMD disconnected GI staple/PDF limit sanity check]")
    status = 0
    for bz in (2, -2):
        pdf_pos = _w_index_position(out, "GI_PDF", [0, bz, 0, 0])
        tmd_x_pos = _w_index_position(out, "GI_qTMD", [0, bz, 1, 0])
        tmd_y_pos = _w_index_position(out, "GI_qTMD", [0, bz, 1, 1])
        checks = {
            f"bz={bz} GI_PDF vs GI_qTMD b_X raw": _maxdiff(out["GI_PDF"]["raw"][:, pdf_pos], out["GI_qTMD"]["raw"][:, tmd_x_pos]),
            f"bz={bz} GI_PDF vs GI_qTMD b_Y raw": _maxdiff(out["GI_PDF"]["raw"][:, pdf_pos], out["GI_qTMD"]["raw"][:, tmd_y_pos]),
        }
        for label, value in checks.items():
            print(f"{label}: {value:.16e}")
            status |= int(value != 0.0)
    raise SystemExit(status)
