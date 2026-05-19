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
    "GI_PDF": DATA_DIR / "S8T32.qTMD1pt.smoke.GI_PDF_bz1.h5",
    "CG_PDF": DATA_DIR / "S8T32.qTMD1pt.local_check.CG_PDF_bz1.h5",
    "CG_qTMD": DATA_DIR / "S8T32.qTMD1pt.smoke.CG_qTMD_bz1bt1.h5",
}


def _read_required_outputs():
    missing = [str(path) for path in DEFAULT_FILES.values() if not path.exists()]
    if missing:
        raise SkipTest("nonzero-bz sanity-check HDF5 files are not present: " + ", ".join(missing))

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


def test_disconnected_qtmd_nonzero_bz_shift_convention():
    outputs = _read_required_outputs()

    expected_pdf_indices = [[0, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0]]
    assert outputs["GI_PDF"]["w_index"] == expected_pdf_indices
    assert outputs["CG_PDF"]["w_index"] == expected_pdf_indices
    for kind in outputs:
        np.testing.assert_array_equal(outputs[kind]["momentum"], [[0, 0, 0, 0]])

    for bz in (1, -1):
        pdf_pos = _w_index_position(outputs, "CG_PDF", [0, bz, 0, 0])
        tmd_x_pos = _w_index_position(outputs, "CG_qTMD", [0, bz, 0, 0])
        tmd_y_pos = _w_index_position(outputs, "CG_qTMD", [0, bz, 0, 1])
        gi_pos = _w_index_position(outputs, "GI_PDF", [0, bz, 0, 0])

        np.testing.assert_allclose(
            outputs["CG_PDF"]["raw"][:, pdf_pos],
            outputs["CG_qTMD"]["raw"][:, tmd_x_pos],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            outputs["CG_PDF"]["raw"][:, pdf_pos],
            outputs["CG_qTMD"]["raw"][:, tmd_y_pos],
            atol=0.0,
            rtol=0.0,
        )

        gi_cg_maxdiff = _maxdiff(outputs["GI_PDF"]["raw"][:, gi_pos], outputs["CG_PDF"]["raw"][:, pdf_pos])
        assert gi_cg_maxdiff > 0.0


if __name__ == "__main__":
    try:
        out = _read_required_outputs()
    except SkipTest as err:
        print(f"SKIP: {err}")
        raise SystemExit(0)

    print("[qTMD disconnected nonzero-bz sanity check]")
    for bz in (1, -1):
        pdf_pos = _w_index_position(out, "CG_PDF", [0, bz, 0, 0])
        tmd_x_pos = _w_index_position(out, "CG_qTMD", [0, bz, 0, 0])
        tmd_y_pos = _w_index_position(out, "CG_qTMD", [0, bz, 0, 1])
        gi_pos = _w_index_position(out, "GI_PDF", [0, bz, 0, 0])
        checks = {
            f"bz={bz} CG_PDF vs CG_qTMD b_X raw": _maxdiff(out["CG_PDF"]["raw"][:, pdf_pos], out["CG_qTMD"]["raw"][:, tmd_x_pos]),
            f"bz={bz} CG_PDF vs CG_qTMD b_Y raw": _maxdiff(out["CG_PDF"]["raw"][:, pdf_pos], out["CG_qTMD"]["raw"][:, tmd_y_pos]),
            f"bz={bz} GI_PDF vs CG_PDF raw": _maxdiff(out["GI_PDF"]["raw"][:, gi_pos], out["CG_PDF"]["raw"][:, pdf_pos]),
        }
        for label, value in checks.items():
            print(f"{label}: {value:.16e}")

    raise SystemExit(0)
