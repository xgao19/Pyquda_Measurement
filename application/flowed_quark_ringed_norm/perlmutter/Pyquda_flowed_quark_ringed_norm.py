import argparse
import os

from pyquda import init
from pyquda_utils import io


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


def parse_mg_block(text):
    if str(text).strip().lower() in {"none", "off", "false", "0"}:
        return None
    blocks = []
    for block_text in str(text).replace("/", ";").replace("|", ";").split(";"):
        block_text = block_text.strip()
        if not block_text:
            continue
        block = [int(v) for v in block_text.replace(",", ".").split(".") if v]
        if len(block) != 4:
            raise ValueError(f"multigrid blocks must have four integers, got {block_text!r}")
        blocks.append(block)
    return blocks or None


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("FLOWED_RINGED_CONFIG_NUM", "0")))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("FLOWED_RINGED_MPI_GEOMETRY", "1.1.1.1"))
args, unknown = parser.parse_known_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

script_dir = os.path.dirname(__file__)
software_root = os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software")
measurement_root = os.environ.get("MEASUREMENT_ROOT", os.path.join(software_root, "Pyquda_Measurement"))
data_dir = os.environ.get("FLOWED_RINGED_DATA_DIR", os.path.join(script_dir, "data"))
gauge_path = os.environ.get(
    "FLOWED_RINGED_GAUGE_PATH",
    os.path.join(measurement_root, "test_gauge/S8T8_wilson_b6.0"),
)
lat_tag = os.environ.get("FLOWED_RINGED_LAT_TAG", "S8T8")
sm_tag = os.environ.get("FLOWED_RINGED_SM_TAG", "S8T8_ringed")
hyp_project = int(os.environ.get("FLOWED_RINGED_HYP_PROJECT", "-1"))
gauge_preprocessing = os.environ.get(
    "FLOWED_RINGED_GAUGE_PREPROCESSING",
    f"HYP(1,0.75,0.6,0.3,{hyp_project})",
)

parameters = {
    "flow_type": os.environ.get("FLOWED_RINGED_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("FLOWED_RINGED_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("FLOWED_RINGED_FLOW_STEPS", "1")),
    "noise_scheme": os.environ.get("FLOWED_RINGED_NOISE_SCHEME", "zn"),
    "hp_num_vectors": int(os.environ.get("FLOWED_RINGED_HP_NUM_VECTORS", "1")),
    "hp_ordering": os.environ.get("FLOWED_RINGED_HP_ORDERING", "interleaved_xyzt_binary_projected_to_evenodd"),
    "spin_color_dilution": os.environ.get("FLOWED_RINGED_SPIN_COLOR_DILUTION", "none"),
    "Nc": int(os.environ.get("FLOWED_RINGED_NC", "3")),
    "multigrid": parse_mg_block(os.environ.get("FLOWED_RINGED_MG_BLOCK", "8.8.4.4")),
    "gauge_preprocessing": gauge_preprocessing,
    "flavor_convention": os.environ.get(
        "FLOWED_RINGED_FLAVOR_CONVENTION",
        "single_flavor_trace_for_this_dirac_operator",
    ),
    "block_interval_solves": int(os.environ.get("FLOWED_RINGED_BLOCK_INTERVAL_SOLVES", "64")),
}

init(mpi_geometry, enable_mps=True)

from pyquda_measurement_utils.flowed_quark_ringed_norm import FlowedQuarkRingedNorm  # noqa: E402
from pyquda_measurement_utils.io_corr import get_flowed_quark_ringed_norm_file_tag  # noqa: E402
from pyquda_measurement_utils.tools import mpi_print  # noqa: E402


tag = get_flowed_quark_ringed_norm_file_tag(data_dir, lat_tag, conf, 0, sm_tag)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, hyp_project)
gauge.latt_info.t_boundary = int(os.environ.get("FLOWED_RINGED_T_BOUNDARY", "-1"))
latt_info = gauge.latt_info

invPara = [
    float(os.environ.get("FLOWED_RINGED_MASS", "0.236")),
    float(os.environ.get("FLOWED_RINGED_CSW", "1.0372")),
    float(os.environ.get("FLOWED_RINGED_TOL", "1e-10")),
    int(os.environ.get("FLOWED_RINGED_MAXITER", "300")),
]
randPara = [
    int(os.environ.get("FLOWED_RINGED_N_VEC", "1")),
    int(os.environ.get("FLOWED_RINGED_N_ZN", "2")),
    int(os.environ.get("FLOWED_RINGED_RAND_SEED", str(conf))),
]

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--gauge_path {gauge_path.format(conf=conf)}")
mpi_print(latt_info, f"--data_dir {data_dir}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--flow_type {parameters['flow_type']}")
mpi_print(latt_info, f"--flow_epsilon {parameters['flow_epsilon']}")
mpi_print(latt_info, f"--flow_steps {parameters['flow_steps']}")
mpi_print(latt_info, f"--noise_scheme {parameters['noise_scheme']}")
mpi_print(latt_info, f"--hp_num_vectors {parameters['hp_num_vectors']}")
mpi_print(latt_info, f"--hp_ordering {parameters['hp_ordering']}")
mpi_print(latt_info, f"--spin_color_dilution {parameters['spin_color_dilution']}")
mpi_print(latt_info, f"--block_interval_solves {parameters['block_interval_solves']}")
mpi_print(latt_info, f"--gauge_preprocessing {gauge_preprocessing}")

ringed_norm = FlowedQuarkRingedNorm(parameters)
ringed_norm.flowed_kinetic_norm(
    gauge,
    invPara,
    randPara,
    tag=os.environ.get("FLOWED_RINGED_OUT", tag),
)
