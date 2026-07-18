import argparse
import os
from pathlib import Path

from pyquda import init
from pyquda_utils import io

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    parse_multigrid_blocks,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    disconnected_sample_log_path,
)
from pyquda_measurement_utils.io_corr import (
    get_emt_quark_loop_file_tag,
)


def positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("EMT_1PT_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--mg-block", default="8.8.4.4")
parser.add_argument(
    "--flow-batch-size",
    type=positive_integer,
    default=1,
    help="number of stochastic sources flowed together (default: 1)",
)
args = parser.parse_args()

conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Production knobs: ensemble paths, output tags, momentum grid,
# gradient-flow schedule, and stochastic estimator.
data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
repo_root = Path(__file__).resolve().parents[3]
gauge_path = os.environ.get(
    "EMT_1PT_GAUGE_PATH",
    str(repo_root / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
)
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
qmax = int(os.environ.get("EMT_1PT_QMAX", "0"))
qzmax = int(os.environ.get("EMT_1PT_QZ_MAX", str(qmax)))
qext = [[x, y, z, 0] for x in range(-qmax, qmax + 1) for y in range(-qmax, qmax + 1) for z in range(-qzmax, qzmax + 1)]
setup_tag = os.environ.get("EMT_1PT_SETUP_TAG", "1HYP")

parameters = {
    "config_num": conf,
    "qext": qext,
    "flow_type": os.environ.get("EMT_1PT_FLOW_TYPE", "wilson"),
    "flow_epsilon": float(os.environ.get("EMT_1PT_FLOW_EPSILON", "0.207936")),
    "flow_steps": int(os.environ.get("EMT_1PT_FLOW_STEPS", "1")),
    "noise_scheme": os.environ.get("EMT_1PT_NOISE_SCHEME", "zn"),
    "hp_num_vectors": int(os.environ.get("EMT_1PT_HP_NUM_VECTORS", "1")),
    "hp_ordering": os.environ.get("EMT_1PT_HP_ORDERING", "interleaved_xyzt_binary_projected_to_evenodd"),
    "gauge_preprocessing": "HYP(1,0.75,0.6,0.3,dir_ignore=-1)",
    "multigrid": parse_multigrid_blocks(args.mg_block),
}
quark_1pt_tag = get_emt_quark_loop_file_tag(data_dir, lat_tag, conf, 0, setup_tag)

init(mpi_geometry, enable_mps=True)

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
gauge.latt_info.t_boundary = -1

# Inverter knobs: mass, clover coefficient, tolerance, max iterations.
invPara = [
    float(os.environ.get("EMT_1PT_MASS", "0.236")),
    float(os.environ.get("EMT_1PT_CSW", "1.0372")),
    float(os.environ.get("EMT_1PT_TOL", "1e-10")),
    int(os.environ.get("EMT_1PT_MAXITER", "300")),
]
# Stochastic-source knobs: number of vectors, Z_n, counter-noise stream salt.
randPara = [
    int(os.environ.get("EMT_1PT_N_VEC", "1")),
    int(os.environ.get("EMT_1PT_N_ZN", "4")),
    int(os.environ.get("EMT_1PT_RAND_SEED", "0")),
]
base_start = int(os.environ.get("EMT_1PT_BASE_START", "0"))
base_stop = int(os.environ.get("EMT_1PT_BASE_STOP", str(randPara[0])))
block_interval_solves = int(os.environ.get("EMT_1PT_BLOCK_INTERVAL_SOLVES", "64"))
shard_dir = os.environ.get("EMT_1PT_SHARD_DIR", os.path.join(data_dir, "EMTc", "shards"))

quark_1pt = EMTDisconnectedQuark1pt(parameters)
output_tag = os.environ.get("EMT_1PT_QUARK_OUT", quark_1pt_tag)
quark_1pt.flowed_fermionic_1pt(
    gauge,
    invPara,
    randPara,
    tag=output_tag,
    shard_dir=shard_dir,
    sample_log_file=disconnected_sample_log_path(data_dir, output_tag),
    base_start=base_start,
    base_stop=base_stop,
    block_interval_solves=block_interval_solves,
    flow_batch_size=args.flow_batch_size,
)
