import os
from pathlib import Path

from pyquda import init
from pyquda_utils import io
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT
from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import parse_optional_multigrid_blocks
from pyquda_measurement_utils.io_corr import get_emt_meson_2pt_file_tag, get_emt_quark_3pt_file_tag

# ============================================================
# Argument parsing
# ============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
parser.add_argument("--src_interpolator", type=str, default="5", help="Source interpolator gamma label")
parser.add_argument("--sink_interpolator", type=str, default="5", help="Sink interpolator gamma label")
parser.add_argument("--mg-block", default="8.8.4.4", help="X.Y.Z.T[;...] or none")
args = parser.parse_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# ============================================================
# Shared configuration
# ============================================================
# Production knobs: ensemble paths, output tags, source position,
# momentum grid, sink separations, smearing, and gradient-flow schedule.
repo_root = Path(__file__).resolve().parents[3]
data_dir = os.environ.get("EMT_DATA_DIR", str(repo_root / "application" / "EMT_meson" / "perlmutter" / "data"))
gauge_path = os.environ.get(
    "EMT_GAUGE_PATH",
    str(repo_root / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
)
lat_tag = os.environ.get("EMT_LAT_TAG", "S8T32")
sm_tag = os.environ.get("EMT_SM_TAG", "1HYP_GSRC_W1_k0")
qext = [[x, y, z, 0] for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)]
parameters = {
    "config_num": conf,
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "CG_GaussSmear": True,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": "wilson",
    "flow_epsilon": 0.207936,
    "flow_steps": 10,
    "gauge_preprocessing": "HYP(1,0.75,0.6,0.3,4)",
    "multigrid": parse_optional_multigrid_blocks(args.mg_block),
}
src_pos = [0, 0, 0, 0]
meson_2pt_tag = get_emt_meson_2pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)
quark_3pt_tag = get_emt_quark_3pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag, spin=5)

# ============================================================
# Initialize QUDA backend
# ============================================================
init(mpi_geometry, enable_mps=True)

# ============================================================
# Gauge field
# ============================================================
gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1

# ============================================================
# Measurement
# ============================================================
# Inverter knobs: mass, clover coefficient, tolerance, max iterations.
invPara = [0.236, 1.0372, 1e-10, 300]  # mf, csw, prec, cgMax

quark_emt = QuarkEMT(parameters)

quark_emt.connected_3pt(
    gauge,
    invPara,
    src_pos=src_pos,
    t_separations=[2, 3],
    spin=5,
    tag=quark_3pt_tag,
    c2_tag=meson_2pt_tag,
    src_interpolator=args.src_interpolator,
    sink_interpolator=args.sink_interpolator,
)
