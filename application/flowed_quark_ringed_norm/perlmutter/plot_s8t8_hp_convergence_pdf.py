#!/usr/bin/env python3
"""Plot the S8T8 stochastic-vs-HP ringed-normalization benchmark as a PDF.

This intentionally avoids matplotlib so it can run in a lean PyQUDA
environment.  The output is a simple vector PDF with two panels.
"""

import argparse
import csv
import math
from pathlib import Path


CASES = ["zn1024", "hp64x16", "hp4x256"]
COLORS = {
    "zn1024": (0.10, 0.32, 0.70),
    "hp64x16": (0.80, 0.24, 0.12),
    "hp4x256": (0.25, 0.55, 0.20),
}
LABELS = {
    "zn1024": "pure stochastic",
    "hp64x16": "stochastic HP16",
    "hp4x256": "stochastic HP256",
}


def _read_rows(csv_path):
    rows = []
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            converted = {}
            for key, value in row.items():
                if key in {"case", "label"}:
                    converted[key] = value
                elif value == "nan":
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


class PdfCanvas:
    def __init__(self, width=612, height=792):
        self.width = width
        self.height = height
        self.ops = []

    @staticmethod
    def _esc(text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def op(self, text):
        self.ops.append(text)

    def rgb(self, color):
        self.op(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG "
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg"
        )

    def text(self, x, y, label, size=10, color=(0, 0, 0), align="left"):
        self.rgb(color)
        approx_w = 0.52 * size * len(str(label))
        if align == "center":
            x -= approx_w / 2
        elif align == "right":
            x -= approx_w
        self.op(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({self._esc(label)}) Tj ET")

    def line(self, x1, y1, x2, y2, color=(0, 0, 0), width=1):
        self.rgb(color)
        self.op(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def circle(self, x, y, r, color):
        k = 0.55228475 * r
        self.rgb(color)
        self.op(
            f"{x+r:.2f} {y:.2f} m "
            f"{x+r:.2f} {y+k:.2f} {x+k:.2f} {y+r:.2f} {x:.2f} {y+r:.2f} c "
            f"{x-k:.2f} {y+r:.2f} {x-r:.2f} {y+k:.2f} {x-r:.2f} {y:.2f} c "
            f"{x-r:.2f} {y-k:.2f} {x-k:.2f} {y-r:.2f} {x:.2f} {y-r:.2f} c "
            f"{x+k:.2f} {y-r:.2f} {x+r:.2f} {y-k:.2f} {x+r:.2f} {y:.2f} c f"
        )

    def write(self, path):
        stream = "\n".join(self.ops) + "\n"
        objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(stream.encode('latin1'))} >>\nstream\n{stream}endstream",
        ]
        pdf = ["%PDF-1.4\n"]
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(sum(len(part.encode("latin1")) for part in pdf))
            pdf.append(f"{idx} 0 obj\n{obj}\nendobj\n")
        xref_offset = sum(len(part.encode("latin1")) for part in pdf)
        pdf.append(f"xref\n0 {len(objects)+1}\n")
        pdf.append("0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.append(f"{off:010d} 00000 n \n")
        pdf.append(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")
        path.write_bytes("".join(pdf).encode("latin1"))


def _nice_range(values, pad_frac=0.12):
    lo, hi = min(values), max(values)
    if lo == hi:
        return lo - 1, hi + 1
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _draw_panel(canvas, rows, panel, y_key, title, ylabel, error_key=None):
    solves = sorted({int(row["solves"]) for row in rows})
    x0, y0, pw, ph = panel
    values = [float(row[y_key]) for row in rows]
    if error_key:
        for row in rows:
            err = row[error_key]
            if err != "nan":
                values.extend([float(row[y_key]) - float(err), float(row[y_key]) + float(err)])
    ymin, ymax = _nice_range(values)
    xmin, xmax = math.log2(min(solves)), math.log2(max(solves))

    def x_pos(solve_count):
        return x0 + (math.log2(solve_count) - xmin) / (xmax - xmin) * pw

    def y_pos(value):
        return y0 + (float(value) - ymin) / (ymax - ymin) * ph

    canvas.line(x0, y0, x0 + pw, y0, (0, 0, 0), 0.8)
    canvas.line(x0, y0, x0, y0 + ph, (0, 0, 0), 0.8)
    canvas.line(x0 + pw, y0, x0 + pw, y0 + ph, (0.75, 0.75, 0.75), 0.5)
    canvas.line(x0, y0 + ph, x0 + pw, y0 + ph, (0.75, 0.75, 0.75), 0.5)
    for idx in range(5):
        yy = y0 + ph * idx / 4
        if idx not in (0, 4):
            canvas.line(x0, yy, x0 + pw, yy, (0.88, 0.88, 0.88), 0.4)
        value = ymin + (ymax - ymin) * idx / 4
        canvas.text(x0 - 8, yy - 3, f"{value:.6g}", 8, align="right")
    for solve_count in solves:
        xx = x_pos(solve_count)
        canvas.line(xx, y0, xx, y0 - 4, (0, 0, 0), 0.7)
        canvas.text(xx, y0 - 18, str(solve_count), 8, align="center")

    canvas.text(x0 + pw / 2, y0 + ph + 18, title, 12, align="center")
    canvas.text(x0 - 58, y0 + ph / 2, ylabel, 10, align="center")
    canvas.text(x0 + pw / 2, y0 - 38, "matched solves", 10, align="center")

    for case in CASES:
        case_rows = sorted([row for row in rows if row["case"] == case], key=lambda row: row["solves"])
        if not case_rows:
            continue
        points = [(x_pos(int(row["solves"])), y_pos(float(row[y_key]))) for row in case_rows]
        color = COLORS[case]
        canvas.rgb(color)
        path = []
        for idx, (xx, yy) in enumerate(points):
            path.append(f"{xx:.2f} {yy:.2f} " + ("m" if idx == 0 else "l"))
        canvas.op("1.6 w " + " ".join(path) + " S")
        for row, (xx, yy) in zip(case_rows, points):
            if error_key and row[error_key] != "nan":
                err = float(row[error_key])
                ylo, yhi = y_pos(float(row[y_key]) - err), y_pos(float(row[y_key]) + err)
                canvas.line(xx, ylo, xx, yhi, color, 0.7)
                canvas.line(xx - 3, ylo, xx + 3, ylo, color, 0.7)
                canvas.line(xx - 3, yhi, xx + 3, yhi, color, 0.7)
            canvas.circle(xx, yy, 3.2, color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", default=str(Path(__file__).resolve().parent / "benchmark/s8t8_hp_convergence"))
    args = parser.parse_args()

    root = Path(args.bench_root).resolve()
    rows = _read_rows(root / "summary.csv")
    canvas = PdfCanvas()
    canvas.text(306, 755, "S8T8 flowed-quark ringed normalization convergence", 15, align="center")
    canvas.text(
        306,
        736,
        "Pure stochastic vs stochastic HP16/HP256; flow step 1; matched solves",
        10,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    _draw_panel(canvas, rows, (82, 455, 494, 255), "K_real", "K_spacetime(flow=1), real part", "K real", "K_block_sem_abs")
    _draw_panel(canvas, rows, (82, 115, 494, 255), "Z_bilinear_real", "Z_ring_bilinear(flow=1), real part", "Z real")

    for idx, case in enumerate(CASES):
        y = 704 - 16 * idx
        canvas.line(365, y, 393, y, COLORS[case], 1.8)
        canvas.circle(379, y, 3.2, COLORS[case])
        canvas.text(401, y - 3, LABELS[case], 9)

    out = root / "s8t8_hp_convergence_results.pdf"
    canvas.write(out)
    print(out)


if __name__ == "__main__":
    main()
