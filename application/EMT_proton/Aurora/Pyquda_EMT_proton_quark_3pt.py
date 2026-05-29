import argparse
import os
from pathlib import Path

from pyquda import init
from pyquda_utils import io


def parse_triplet(text):
    return [int(v) for v in str(text).replace(",", ".").split(".") if v]


def parse_int_list(text):
    return [int(v) for v in str(text).replace(".", ",").split(",") if v]


def parse_str_list(text):
    return [v for v in str(text).replace(".", ",").split(",") if v]


parser = argparse.ArgumentParser()
parser.add_argument("--stream", type=str, default=os.environ.get("EMT_PROTON_STREAM", "b"))
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_PROTON_CONFIG_NUM", "220")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_PROTON_MPI_GEOMETRY", "1.5.4.5"))
parser.add_argument("--interpolator", type=str, default=os.environ.get("EMT_PROTON_INTERPOLATOR", "5"))
args, unknown = parser.parse_known_args()

stream = args.stream
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

init(
    mpi_geometry,
    enable_mps=True,
    backend="dpnp",
    backend_target="sycl",
    resource_path=os.environ.get("QUDA_RESOURCE_PATH", ".cache"),
)

from pyquda_measurement_utils.io_corr import (  # noqa: E402
    get_emt_proton_2pt_file_tag,
    get_emt_proton_quark_3pt_file_tag,
)
from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT  # noqa: E402
from pyquda_measurement_utils.tools import mpi_print, srcLoc_distri_eq  # noqa: E402


script_dir = Path(__file__).resolve().parent
data_dir = os.environ.get(
    "EMT_PROTON_DATA_DIR",
    f"/lus/flare/projects/StructNGB/xgao/run/l80c80a050/proton_EMT_pyquda/data_{stream}",
)
gauge_path_template = os.environ.get(
    "EMT_PROTON_GAUGE_PATH",
    "/lus/flare/projects/StructNGB/xgao/ensembles/s8080b7596/gauge_fixed/{stream}/"
    "l8080f21b7596m00101m0202{stream}.coulomb.1e-14.{conf}",
)
gauge_path = gauge_path_template.format(stream=stream, conf=conf)

lat_tag = os.environ.get("EMT_PROTON_LAT_TAG", "l80c80a050")
qmax = int(os.environ.get("EMT_PROTON_QMAX", "0"))
qext = [
    [x, y, z, 0]
    for x in range(-qmax, qmax + 1)
    for y in range(-qmax, qmax + 1)
    for z in range(-qmax, qmax + 1)
]
pf = parse_triplet(os.environ.get("EMT_PROTON_PF", "0.0.0")) + [0]
t_separations = parse_int_list(os.environ.get("EMT_PROTON_T_SEPS", "9"))
pol_list = parse_str_list(os.environ.get("EMT_PROTON_POL", "PpUnpol"))
width = float(os.environ.get("EMT_PROTON_WIDTH", "13.0"))
boost_in = parse_triplet(os.environ.get("EMT_PROTON_BOOST_IN", "0.0.0"))
boost_out = parse_triplet(os.environ.get("EMT_PROTON_BOOST_OUT", "0.0.0"))
sm_tag = os.environ.get(
    "EMT_PROTON_SM_TAG",
    f"1HYP_GSRC_W{width:g}_k0_{args.interpolator}",
)

src_shift = parse_triplet(os.environ.get("EMT_PROTON_SRC_SHIFT", "7.11.13")) + [
    int(os.environ.get("EMT_PROTON_SRC_T_SHIFT", "23"))
]
num_src = int(os.environ.get("EMT_PROTON_NUM_SRC", "1"))

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

if not t_separations:
    raise ValueError("EMT_PROTON_T_SEPS must contain at least one source-sink separation.")
if not pol_list:
    raise ValueError("EMT_PROTON_POL must contain at least one polarization label.")

Path(data_dir).mkdir(parents=True, exist_ok=True)

gauge = io.readNERSCGauge(
    gauge_path,
    checksum=bool(int(os.environ.get("EMT_PROTON_CHECKSUM", "0"))),
    link_trace=bool(int(os.environ.get("EMT_PROTON_LINK_TRACE", "0"))),
    plaquette=bool(int(os.environ.get("EMT_PROTON_READ_PLAQUETTE", "0"))),
)
gauge.hypSmear(1, 0.75, 0.6, 0.3, int(os.environ.get("EMT_PROTON_HYP_PROJECT", "-1")))
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info

src_origin = [(conf + src_shift[i]) % latt_info.global_size[i] for i in range(4)]
src_positions = srcLoc_distri_eq(latt_info.global_size, src_origin)[:num_src]

invPara = [
    float(os.environ.get("EMT_PROTON_MASS", "-0.0386")),
    float(os.environ.get("EMT_PROTON_CSW", "1.03094")),
    float(os.environ.get("EMT_PROTON_TOL", "1e-10")),
    int(os.environ.get("EMT_PROTON_MAXITER", "5000")),
]

mpi_print(latt_info, f"--stream {stream}")
mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--gauge_path {gauge_path}")
mpi_print(latt_info, f"--data_dir {data_dir}")
mpi_print(latt_info, f"--qmax {qmax}")
mpi_print(latt_info, f"--t_separations {t_separations}")
mpi_print(latt_info, f"--pol_list {pol_list}")
mpi_print(latt_info, f"--num_src {num_src}")
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

quark_emt = ProtonQuarkEMT(parameters)
for src_pos in src_positions:
    c2_tag = get_emt_proton_2pt_file_tag(data_dir, lat_tag, conf, 0, src_pos, sm_tag)
    quark_3pt_tag = get_emt_proton_quark_3pt_file_tag(
        data_dir,
        lat_tag,
        conf,
        0,
        src_pos,
        sm_tag,
        spin=spin,
    )
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
