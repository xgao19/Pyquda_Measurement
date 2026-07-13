import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT
from pyquda_measurement_utils.io_corr import (
    get_emt_quark_loop_file_tag,
    get_flowed_quark_ringed_norm_file_tag,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_PROTON_CONFIG_NUM", "0")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_PROTON_MPI_GEOMETRY", "1.1.1.1"))
args, unknown = parser.parse_known_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Production knobs: ensemble paths, output tags, momentum grid,
# gradient-flow schedule, and stochastic estimator.
data_dir = os.environ.get("EMT_PROTON_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
gauge_path = os.environ.get(
    "EMT_PROTON_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = os.environ.get("EMT_PROTON_LAT_TAG", "S8T32")
qmax = int(os.environ.get("EMT_PROTON_QMAX", "0"))
qext = [[x, y, z, 0] for x in range(-qmax, qmax + 1) for y in range(-qmax, qmax + 1) for z in range(-qmax, qmax + 1)]
sm_tag = os.environ.get("EMT_PROTON_SM_TAG", "1HYP_GSRC_W1_k0_5")

parameters = {
    "config_num": conf,
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "CG_GaussSmear": False,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "pol": ["PpUnpol"],
    "t_insert": 0,
    "flow_type": os.environ.get("EMT_PROTON_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_PROTON_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_PROTON_FLOW_STEPS", "1")),
    "gauge_preprocessing": os.environ.get(
        "EMT_PROTON_GAUGE_PREPROCESSING",
        "HYP(1,0.75,0.6,0.3,4)",
    ),
}
quark_1pt_tag = get_emt_quark_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag)
ringed_tag = get_flowed_quark_ringed_norm_file_tag(data_dir, lat_tag, conf, 0, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1

# Inverter knobs: mass, clover coefficient, tolerance, max iterations.
invPara = [
    float(os.environ.get("EMT_PROTON_MASS", "0.236")),
    float(os.environ.get("EMT_PROTON_CSW", "1.0372")),
    float(os.environ.get("EMT_PROTON_TOL", "1e-10")),
    int(os.environ.get("EMT_PROTON_MAXITER", "300")),
]
# Stochastic-source knobs: number of vectors, Z_n, counter-noise stream salt.
randPara = [
    int(os.environ.get("EMT_PROTON_N_VEC", "1")),
    int(os.environ.get("EMT_PROTON_N_ZN", "4")),
    int(os.environ.get("EMT_PROTON_RAND_SEED", "0")),
]

quark_emt = ProtonQuarkEMT(parameters)
quark_emt.flowed_fermionic_1pt(
    gauge,
    invPara,
    randPara,
    tag=os.environ.get("EMT_PROTON_QUARK_1PT_OUT", quark_1pt_tag),
    ringed_tag=os.environ.get("EMT_PROTON_RINGED_OUT", ringed_tag),
    shard_dir=os.environ.get("EMT_PROTON_SHARD_DIR", os.path.join(data_dir, "EMTc", "shards")),
    base_start=int(os.environ.get("EMT_PROTON_BASE_START", "0")),
    base_stop=int(os.environ.get("EMT_PROTON_BASE_STOP", str(randPara[0]))),
    block_interval_solves=int(os.environ.get("EMT_PROTON_BLOCK_INTERVAL_SOLVES", "64")),
)
