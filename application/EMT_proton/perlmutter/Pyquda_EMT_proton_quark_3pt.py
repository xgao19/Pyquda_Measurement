import argparse
import os

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT
from pyquda_measurement_utils.io_corr import (
    get_emt_proton_2pt_file_tag,
    get_emt_proton_quark_3pt_file_tag,
)


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


def parse_int_list(text):
    return [int(v) for v in text.split(",") if v]


def parse_str_list(text):
    return [v for v in text.split(",") if v]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_PROTON_CONFIG_NUM", "0")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_PROTON_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--interpolator", type=str, default=os.environ.get("EMT_PROTON_INTERPOLATOR", "5"))
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
pf = parse_triplet(os.environ.get("EMT_PROTON_PF", "0.0.0")) + [0]
t_separations = parse_int_list(os.environ.get("EMT_PROTON_T_SEPS", "2"))
pol_list = parse_str_list(os.environ.get("EMT_PROTON_POL", "PpUnpol"))
width = float(os.environ.get("EMT_PROTON_WIDTH", "1.0"))
boost_in = parse_triplet(os.environ.get("EMT_PROTON_BOOST_IN", "0.0.0"))
boost_out = parse_triplet(os.environ.get("EMT_PROTON_BOOST_OUT", "0.0.0"))
sm_tag = os.environ.get(
    "EMT_PROTON_SM_TAG",
    f"1HYP_GSRC_W{width:g}_k0_{args.interpolator}",
)

parameters = {
    "qext": qext,
    "pf": pf,
    "p_2pt": qext,
    "CG_GaussSmear": True,
    "pos_boost": boost_in,
    "neg_boost": boost_out,
    "boost_in": boost_in,
    "boost_out": boost_out,
    "width": width,
    "pol": pol_list,
    "t_insert": max(t_separations),
    "save_propagators": False,
    "flow_type": os.environ.get("EMT_PROTON_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_PROTON_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_PROTON_FLOW_STEPS", "1")),
}

spin = int(os.environ.get("EMT_PROTON_SPIN", "5"))
c2_tag = get_emt_proton_2pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)
quark_3pt_tag = get_emt_proton_quark_3pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag, spin=spin)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1

invPara = [
    float(os.environ.get("EMT_PROTON_MASS", "0.236")),
    float(os.environ.get("EMT_PROTON_CSW", "1.0372")),
    float(os.environ.get("EMT_PROTON_TOL", "1e-15")),
    int(os.environ.get("EMT_PROTON_MAXITER", "300")),
]

quark_emt = ProtonQuarkEMT(parameters)
quark_emt.connected_3pt(
    gauge,
    invPara,
    src_pos=src_pos,
    t_separations=t_separations,
    spin=spin,
    tag=os.environ.get("EMT_PROTON_3PT_OUT", quark_3pt_tag),
    c2_tag=os.environ.get("EMT_PROTON_2PT_OUT", c2_tag),
    interpolator=args.interpolator,
)
