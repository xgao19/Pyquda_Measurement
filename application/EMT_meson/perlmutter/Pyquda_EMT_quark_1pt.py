import os

from pyquda import init
from pyquda_utils import io
from pyquda_measurement_utils.pion_EMT_vibe_develop import QuarkEMT
from pyquda_measurement_utils.io_corr import (
    get_emt_quark_loop_file_tag,
    get_flowed_quark_ringed_norm_file_tag,
)

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
# Production knobs: ensemble paths, output tags, momentum grid,
# gradient-flow schedule, and stochastic estimator.
data_dir = os.environ.get("EMT_DATA_DIR", "/global/cfs/cdirs/m3760/xgao/software/EMT_meson/data")
gauge_path = os.environ.get(
    "EMT_GAUGE_PATH",
    "/global/cfs/cdirs/m3760/xgao/software/Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0",
)
lat_tag = "l64c64a076"
sm_tag = "1HYP_GSRC_W10_k0"
qext = [[x, y, z, 0] for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)]
parameters = {
    "config_num": conf,
    "qext": qext,
    "pf": [0, 0, 0, 0],
    "p_2pt": qext,
    "CG_GaussSmear": False,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": 1.0,
    "flow_type": "wilson",
    "flow_epsilon": 0.207936,
    "flow_steps": 10,
    "gauge_preprocessing": "HYP(1,0.75,0.6,0.3,4)",
}
quark_1pt_tag = get_emt_quark_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag)
ringed_tag = get_flowed_quark_ringed_norm_file_tag(data_dir, lat_tag, conf, 0, sm_tag)

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
quark_emt = QuarkEMT(parameters)
# Inverter knobs: mass, clover coefficient, tolerance, max iterations.
invPara = [0.236, 1.0372, 1e-10, 300]
# Stochastic-source knobs: number of vectors, Z_n, counter-noise stream salt.
randPara = [1, 4, 0]

quark_emt.flowed_fermionic_1pt(
    gauge,
    invPara,
    randPara,
    tag=os.environ.get("EMT_QUARK_1PT_OUT", quark_1pt_tag),
    ringed_tag=os.environ.get("EMT_RINGED_OUT", ringed_tag),
)
