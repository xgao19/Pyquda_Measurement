import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import EMTDisconnectedGluon1pt
from pyquda_measurement_utils.io_corr import get_emt_gluon_loop_file_tag
from pyquda_measurement_utils.tools import mpi_print


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_1PT_MPI_GEOMETRY", "1.1.1.1"))
args = parser.parse_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Production knobs: ensemble paths, output tags, momentum grid, and flow schedule.
data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
gauge_path = os.environ.get(
    "EMT_1PT_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
qmax = int(os.environ.get("EMT_1PT_QMAX", "0"))
qext = [[x, y, z, 0] for x in range(-qmax, qmax + 1) for y in range(-qmax, qmax + 1) for z in range(-qmax, qmax + 1)]
sm_tag = os.environ.get("EMT_1PT_SM_TAG", "1HYP_GSRC_W1_k0")

parameters = {
    "config_num": conf,
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": os.environ.get("EMT_1PT_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_1PT_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_1PT_FLOW_STEPS", "1")),
}
gluon_1pt_tag = get_emt_gluon_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
latt_info = gauge.latt_info

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

gluon_1pt = EMTDisconnectedGluon1pt(parameters)
gauge.toDevice()
gluon_1pt.flowed_1pt(
    gauge,
    tag=os.environ.get("EMT_1PT_GLUON_OUT", gluon_1pt_tag),
)
