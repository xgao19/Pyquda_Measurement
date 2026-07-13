import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import DisconnectedQuarkqTMD1pt
from pyquda_measurement_utils.io_corr import get_disconnected_qTMD_loop_file_tag


def parse_int_list(text):
    return [int(v) for v in text.replace(",", ".").split(".") if v]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("QTMD_1PT_MPI_GEOMETRY", "1.1.1.1"))
args = parser.parse_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

data_dir = os.environ.get("QTMD_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
gauge_path = os.environ.get(
    "QTMD_1PT_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = os.environ.get("QTMD_1PT_LAT_TAG", "S8T32")
qmax = int(os.environ.get("QTMD_1PT_QMAX", "0"))
q_range = range(-qmax, qmax + 1)
qext = [[x, y, 0, 0] for x in q_range for y in q_range]
qext_PDF = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
operator_kind = os.environ.get("QTMD_1PT_OPERATOR_KIND", "GI_PDF")
sm_tag = os.environ.get("QTMD_1PT_SM_TAG", f"1HYP_{operator_kind}_BZ{os.environ.get('QTMD_1PT_BZ', '0')}_BT{os.environ.get('QTMD_1PT_BT', '0')}")

parameters = {
    "config_num": conf,
    "eta": parse_int_list(os.environ.get("QTMD_1PT_ETA", "0")),
    "b_z": int(os.environ.get("QTMD_1PT_BZ", "0")),
    "b_T": int(os.environ.get("QTMD_1PT_BT", "0")),
    "qext": qext,
    "qext_PDF": qext_PDF,
    "noise_scheme": os.environ.get("QTMD_1PT_NOISE_SCHEME", "zn"),
    "hp_num_vectors": int(os.environ.get("QTMD_1PT_HP_NUM_VECTORS", "1")),
    "hp_ordering": os.environ.get("QTMD_1PT_HP_ORDERING", "global_xyzt_gray_projected_to_evenodd"),
    "gauge_preprocessing": os.environ.get(
        "QTMD_1PT_GAUGE_PREPROCESSING", "HYP(1,0.75,0.6,0.3,4)"
    ),
}
tag = get_disconnected_qTMD_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1

invPara = [
    float(os.environ.get("QTMD_1PT_MASS", "0.236")),
    float(os.environ.get("QTMD_1PT_CSW", "1.0372")),
    float(os.environ.get("QTMD_1PT_TOL", "1e-10")),
    int(os.environ.get("QTMD_1PT_MAXITER", "300")),
]
randPara = [
    int(os.environ.get("QTMD_1PT_N_VEC", "1")),
    int(os.environ.get("QTMD_1PT_N_ZN", "4")),
    int(os.environ.get("QTMD_1PT_RAND_SEED", "0")),
]

measurement = DisconnectedQuarkqTMD1pt(parameters)
measurement.measure_1pt(
    gauge,
    invPara,
    randPara,
    tag=os.environ.get("QTMD_1PT_OUT", tag),
    operator_kind=operator_kind,
    shard_dir=os.environ.get("QTMD_1PT_SHARD_DIR", os.path.join(data_dir, "qTMD1pt", "shards")),
    base_start=int(os.environ.get("QTMD_1PT_BASE_START", "0")),
    base_stop=int(os.environ.get("QTMD_1PT_BASE_STOP", str(randPara[0]))),
    block_interval_solves=int(os.environ.get("QTMD_1PT_BLOCK_INTERVAL_SOLVES", "64")),
)
