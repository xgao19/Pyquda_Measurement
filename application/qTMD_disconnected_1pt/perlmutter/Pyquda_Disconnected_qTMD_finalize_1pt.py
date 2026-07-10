import argparse
import os

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import finalize_disconnected_qtmd_1pt_shards
from pyquda_measurement_utils.io_corr import get_disconnected_qTMD_loop_file_tag


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("QTMD_1PT_CONFIG_NUM", "0")))
args = parser.parse_args()

conf = args.config_num
data_dir = os.environ.get("QTMD_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
lat_tag = os.environ.get("QTMD_1PT_LAT_TAG", "S8T32")
operator_kind = os.environ.get("QTMD_1PT_OPERATOR_KIND", "GI_PDF")
sm_tag = os.environ.get(
    "QTMD_1PT_SM_TAG",
    f"1HYP_{operator_kind}_BZ{os.environ.get('QTMD_1PT_BZ', '0')}_BT{os.environ.get('QTMD_1PT_BT', '0')}",
)
n_vec = int(os.environ.get("QTMD_1PT_N_VEC", "1"))
canonical_tag = os.environ.get(
    "QTMD_1PT_OUT",
    get_disconnected_qTMD_loop_file_tag(data_dir, lat_tag, conf, 0, sm_tag),
)
shard_dir = os.environ.get("QTMD_1PT_SHARD_DIR", os.path.join(data_dir, "qTMD1pt", "shards"))

path = finalize_disconnected_qtmd_1pt_shards(shard_dir, canonical_tag, n_vec)
print("Finalized disconnected qTMD 1pt:", path, flush=True)
