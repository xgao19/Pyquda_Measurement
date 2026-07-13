import argparse
import os

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import finalize_emt_quark_1pt_shards
from pyquda_measurement_utils.io_corr import (
    get_emt_quark_loop_file_tag,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, required=True)
args = parser.parse_args()

conf = args.config_num
data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
setup_tag = os.environ.get("EMT_1PT_SETUP_TAG", "1HYP")
n_vec = int(os.environ.get("EMT_1PT_N_VEC", "1"))
canonical_tag = os.environ.get(
    "EMT_1PT_QUARK_OUT",
    get_emt_quark_loop_file_tag(data_dir, lat_tag, conf, 0, setup_tag),
)
shard_dir = os.environ.get("EMT_1PT_SHARD_DIR", os.path.join(data_dir, "EMTc", "shards"))

path = finalize_emt_quark_1pt_shards(shard_dir, canonical_tag, n_vec)
print("Finalized EMT quark 1pt with embedded ringed kinetic:", path, flush=True)
