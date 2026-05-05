import os

from pyquda import init
from pyquda_utils import io
from pyquda_measurement_utils.pion_EMT_vibe_develop import GluonEMT
from pyquda_measurement_utils.io_corr import get_emt_gluon_1pt_file_tag
from pyquda_measurement_utils.tools import mpi_print

# ============================================================
# Argument parsing
# ============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# ============================================================
# Shared configuration
# ============================================================
data_dir = os.environ.get("EMT_DATA_DIR", "/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data")
gauge_path = os.environ.get(
    "EMT_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = "l64c64a076"
sm_tag = "1HYP_GSRC_W10_k0"
src_pos = [0, 0, 0, 0]
qext = [[x, y, z, 0] for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)]
parameters = {
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": "wilson",
    "flow_epsilon": 0.1,
    "flow_steps": 10,
}
gluon_tag = get_emt_gluon_1pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)

# ============================================================
# Initialize QUDA backend
# ============================================================
init(mpi_geometry, enable_mps=True)

# ============================================================
# Gauge field
# ============================================================
gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
latt_info = gauge.latt_info

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

# ============================================================
# Measurement
# ============================================================
gluon_emt = GluonEMT(parameters)
gauge.toDevice()
gluon_out = os.environ.get("EMT_GLUON_1PT_OUT", gluon_tag)
gluon_emt.flowed_1pt(
    gauge,
    tag=gluon_out,
)
