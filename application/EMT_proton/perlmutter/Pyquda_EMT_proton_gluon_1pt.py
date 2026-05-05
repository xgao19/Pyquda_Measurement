import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonGluonEMT
from pyquda_measurement_utils.io_corr import get_emt_gluon_1pt_file_tag
from pyquda_measurement_utils.tools import mpi_print


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_PROTON_CONFIG_NUM", "0")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_PROTON_MPI_GEOMETRY", "1.1.1.1"))
args, unknown = parser.parse_known_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

data_dir = os.environ.get("EMT_PROTON_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
gauge_path = os.environ.get(
    "EMT_PROTON_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = os.environ.get("EMT_PROTON_LAT_TAG", "S8T32")
src_pos = parse_triplet(os.environ.get("EMT_PROTON_SRC_POS", "0.0.0")) + [
    int(os.environ.get("EMT_PROTON_SRC_T", "0"))
]
qmax = int(os.environ.get("EMT_PROTON_QMAX", "0"))
qext = [[x, y, z, 0] for x in range(-qmax, qmax + 1) for y in range(-qmax, qmax + 1) for z in range(-qmax, qmax + 1)]
sm_tag = os.environ.get("EMT_PROTON_SM_TAG", "1HYP_GSRC_W1_k0_5")

parameters = {
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": os.environ.get("EMT_PROTON_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_PROTON_FLOW_EPSILON", "0.1")),
    "flow_steps": int(os.environ.get("EMT_PROTON_FLOW_STEPS", "1")),
}
gluon_tag = get_emt_gluon_1pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
latt_info = gauge.latt_info

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

gluon_emt = ProtonGluonEMT(parameters)
gauge.toDevice()
gluon_emt.flowed_1pt(
    gauge,
    tag=os.environ.get("EMT_PROTON_GLUON_1PT_OUT", gluon_tag),
)
