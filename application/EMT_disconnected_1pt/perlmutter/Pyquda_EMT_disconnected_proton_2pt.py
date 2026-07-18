import argparse
import os
from pathlib import Path

from pyquda import init
from pyquda_utils import core, io

from pyquda_measurement_utils.io_corr import get_emt_proton_2pt_file_tag
from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT
from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import parse_optional_multigrid_blocks
from pyquda_measurement_utils.tools import mpi_print


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


def parse_str_list(text):
    return [v for v in text.split(",") if v]


def parse_int_list(text):
    return [
        int(value)
        for value in text.replace(".", ",").split(",")
        if value
    ]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_1PT_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--interpolator", type=str, default=os.environ.get("EMT_DISC_INTERPOLATOR", "5"))
parser.add_argument("--mg-block", default="8.8.4.4", help="X.Y.Z.T[;...] or none")
parser.add_argument("--t_separations", type=parse_int_list, default=[2])
args = parser.parse_args()
if not args.t_separations:
    parser.error("--t_separations must contain at least one integer")

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
repo_root = Path(__file__).resolve().parents[3]
gauge_path = os.environ.get(
    "EMT_1PT_GAUGE_PATH",
    str(repo_root / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
)
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
src_pos = parse_triplet(os.environ.get("EMT_1PT_SRC_POS", "0.0.0")) + [
    int(os.environ.get("EMT_1PT_SRC_T", "0"))
]

loop_qmax = int(os.environ.get("EMT_1PT_QMAX", "0"))
p2pt_qmax = int(os.environ.get("EMT_DISC_P2PT_QMAX", str(loop_qmax)))
qext = [[x, y, z, 0] for x in range(-loop_qmax, loop_qmax + 1) for y in range(-loop_qmax, loop_qmax + 1) for z in range(-loop_qmax, loop_qmax + 1)]
p_2pt = [[x, y, z, 0] for x in range(-p2pt_qmax, p2pt_qmax + 1) for y in range(-p2pt_qmax, p2pt_qmax + 1) for z in range(-p2pt_qmax, p2pt_qmax + 1)]
pf = parse_triplet(os.environ.get("EMT_DISC_PF", "0.0.0")) + [0]
width = float(os.environ.get("EMT_DISC_WIDTH", "1.0"))
boost_in = parse_triplet(os.environ.get("EMT_DISC_BOOST_IN", "0.0.0"))
boost_out = parse_triplet(os.environ.get("EMT_DISC_BOOST_OUT", "0.0.0"))
pol_list = parse_str_list(os.environ.get("EMT_DISC_POL", "PpUnpol"))
t_separations = args.t_separations
sm_tag = os.environ.get(
    "EMT_DISC_SM_TAG",
    f"1HYP_GSRC_W{width:g}_k0_{args.interpolator}",
)

parameters = {
    "config_num": conf,
    "qext": qext,
    "pf": pf,
    "p_2pt": p_2pt,
    "CG_GaussSmear": True,
    "boost_in": boost_in,
    "boost_out": boost_out,
    "width": width,
    "pol": pol_list,
    "t_separations": t_separations,
    "flow_type": os.environ.get("EMT_1PT_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_1PT_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_1PT_FLOW_STEPS", "1")),
    "gauge_preprocessing": "HYP(1,0.75,0.6,0.3,dir_ignore=-1)",
    "multigrid": parse_optional_multigrid_blocks(args.mg_block),
}

c2_tag = get_emt_proton_2pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info

mass = float(os.environ.get("EMT_DISC_MASS", os.environ.get("EMT_1PT_MASS", "0.236")))
csw = float(os.environ.get("EMT_DISC_CSW", os.environ.get("EMT_1PT_CSW", "1.0372")))
tol = float(os.environ.get("EMT_DISC_TOL", os.environ.get("EMT_1PT_TOL", "1e-10")))
maxiter = int(os.environ.get("EMT_DISC_MAXITER", os.environ.get("EMT_1PT_MAXITER", "300")))
multigrid = parameters["multigrid"]

mpi_print(latt_info, f"disconnected proton 2pt multigrid block: {multigrid}")
dirac = core.getDirac(latt_info, mass, tol, maxiter, 1.0, csw, csw, multigrid)
dirac.loadGauge(gauge)
mpi_print(latt_info, "Disconnected proton 2pt inverter ready.")

quark_emt = ProtonQuarkEMT(parameters)
prop_fw = quark_emt._make_source_prop(
    dirac, gauge, src_pos, restore_original_gauge=False
)
quark_emt.contract_proton_2pt(
    latt_info,
    prop_fw,
    src_pos,
    tag=os.environ.get("EMT_DISC_C2_OUT", c2_tag),
    interpolator=args.interpolator,
    attrs={
        "measurement": "proton_2pt",
        "config_num": conf,
        "mass": mass,
        "csw": csw,
        "tol": tol,
        "maxiter": maxiter,
        "gauge_preprocessing": parameters["gauge_preprocessing"],
        "t_boundary": latt_info.t_boundary,
        "source_position": src_pos,
        "p_2pt": p_2pt,
        "source_interpolator": args.interpolator,
        "sink_interpolator": "all_16_gamma_scan",
        "gaussian_smearing": True,
        "source_smearing": True,
        "sink_smearing": True,
        "sequential_smearing": True,
        "smearing_width": width,
        "source_boost": boost_in,
        "sink_boost": boost_out,
        "dataset_axes": "gamma,p,t",
    },
)
