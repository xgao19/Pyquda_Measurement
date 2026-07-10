import argparse
import os

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import finalize_emt_quark_1pt_shards
from pyquda_measurement_utils.io_corr import (
    get_emt_quark_loop_file_tag,
    get_flowed_quark_ringed_norm_file_tag,
)


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_1PT_CONFIG_NUM", "0")))
args = parser.parse_args()

conf = args.config_num
data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
sm_tag = os.environ.get("EMT_1PT_SM_TAG", "1HYP_GSRC_W1_k0")
n_vec = int(os.environ.get("EMT_1PT_N_VEC", "1"))
canonical_tag = os.environ.get(
    "EMT_1PT_QUARK_OUT",
    get_emt_quark_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag),
)
ringed_tag = os.environ.get(
    "EMT_1PT_RINGED_OUT",
    get_flowed_quark_ringed_norm_file_tag(data_dir, lat_tag, conf, 0, sm_tag),
)
shard_dir = os.environ.get("EMT_1PT_SHARD_DIR", os.path.join(data_dir, "EMTc", "shards"))

paths = finalize_emt_quark_1pt_shards(shard_dir, canonical_tag, ringed_tag, n_vec)
print("Finalized EMT quark 1pt:", paths[0], flush=True)
print("Finalized flowed-quark kinetic:", paths[1], flush=True)
