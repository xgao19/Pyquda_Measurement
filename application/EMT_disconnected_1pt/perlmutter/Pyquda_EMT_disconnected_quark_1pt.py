import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import EMTDisconnectedQuark1pt
from pyquda_measurement_utils.io_corr import get_emt_quark_1pt_file_tag


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_1PT_CONFIG_NUM", "0")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_1PT_MPI_GEOMETRY", "1.1.1.1"))
args, unknown = parser.parse_known_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
gauge_path = os.environ.get(
    "EMT_1PT_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
src_pos = parse_triplet(os.environ.get("EMT_1PT_SRC_POS", "0.0.0")) + [
    int(os.environ.get("EMT_1PT_SRC_T", "0"))
]
qmax = int(os.environ.get("EMT_1PT_QMAX", "0"))
qext = [[x, y, z, 0] for x in range(-qmax, qmax + 1) for y in range(-qmax, qmax + 1) for z in range(-qmax, qmax + 1)]
sm_tag = os.environ.get("EMT_1PT_SM_TAG", "1HYP_GSRC_W1_k0")

parameters = {
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "CG_GaussSmear": False,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": os.environ.get("EMT_1PT_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_1PT_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_1PT_FLOW_STEPS", "1")),
    "noise_scheme": os.environ.get("EMT_1PT_NOISE_SCHEME", "zn"),
    "hp_num_vectors": int(os.environ.get("EMT_1PT_HP_NUM_VECTORS", "1")),
    "hp_ordering": os.environ.get("EMT_1PT_HP_ORDERING", "global_xyzt_gray_projected_to_evenodd"),
}
quark_1pt_tag = get_emt_quark_1pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1

invPara = [
    float(os.environ.get("EMT_1PT_MASS", "0.236")),
    float(os.environ.get("EMT_1PT_CSW", "1.0372")),
    float(os.environ.get("EMT_1PT_TOL", "1e-15")),
    int(os.environ.get("EMT_1PT_MAXITER", "300")),
]
randPara = [
    int(os.environ.get("EMT_1PT_N_VEC", "1")),
    int(os.environ.get("EMT_1PT_N_ZN", "2")),
    int(os.environ.get("EMT_1PT_RAND_SEED", str(conf))),
]

quark_1pt = EMTDisconnectedQuark1pt(parameters)
quark_1pt.flowed_fermionic_1pt(
    gauge,
    invPara,
    randPara,
    tag=os.environ.get("EMT_1PT_QUARK_OUT", quark_1pt_tag),
)
