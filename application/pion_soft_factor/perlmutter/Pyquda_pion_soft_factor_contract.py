import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("PION_SOFT_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("PION_SOFT_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("PION_SOFT_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("PION_SOFT_DATA_DIR", ""))
parser.add_argument("--t_start", type=int, default=int(os.environ.get("PION_SOFT_T_START", 0)))
parser.add_argument("--t_count", type=int, default=int(os.environ.get("PION_SOFT_T_COUNT", 1)))
parser.add_argument("--quark_mom_z", type=str, default=os.environ.get("PION_SOFT_QUARK_MOM_Z", "4,5"))
parser.add_argument("--bT_dir", type=str, default=os.environ.get("PION_SOFT_BT_DIR", "0"))
parser.add_argument("--bT_length", type=int, default=int(os.environ.get("PION_SOFT_BT_LENGTH", 20)))
parser.add_argument("--bz_length", type=int, default=int(os.environ.get("PION_SOFT_BZ_LENGTH", 0)))
parser.add_argument("--tsep_list", type=str, default=os.environ.get("PION_SOFT_TSEP_LIST", "6,8,10"))
args, unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import io

from pyquda_measurement_utils.io_corr import (
    get_pion_soft_factor_c2pt_file_tag,
    get_pion_soft_factor_file_tag,
    get_pion_soft_factor_prop_file_tag,
    get_pion_soft_factor_qTMDWF_file_tag,
    save_pion_soft_factor_c2pt_hdf5_noRoll,
    save_pion_soft_factor_hdf5_noRoll,
    save_pion_soft_factor_qTMDWF_hdf5_noRoll,
)
from pyquda_measurement_utils.pion_soft_factor_vibe_develop import pion_soft_factor
from pyquda_measurement_utils.tools import mpi_print


software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("PION_SOFT_LAT_TAG", "S8T32")
sm_tag = os.environ.get("PION_SOFT_SM_TAG", "1HYP_wall")
conf = args.config_num

quark_mom = [[0, 0, int(nz)] for nz in args.quark_mom_z.split(",") if nz != ""]
bT_dir = [int(v) for v in args.bT_dir.split(",") if v != ""]
tsep_list = [int(v) for v in args.tsep_list.split(",") if v != ""]

parameters = {
    "quark_mom": quark_mom,
    "bT_dir": bT_dir,
    "bT_length": args.bT_length,
    "bz_length": args.bz_length,
    "tsep_list": tsep_list,
}
measurement = pion_soft_factor(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "pion_soft_factor").mkdir(parents=True, exist_ok=True)
    (data_dir / "pion_soft_factor_c2pt").mkdir(parents=True, exist_ok=True)
    (data_dir / "pion_soft_factor_qTMDWF").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--config_num {conf}")
    print(f"--mpi_geometry {args.mpi_geometry}")
    print(f"--quark_mom {quark_mom}")
    print(f"--bT_dir {bT_dir}")
    print(f"--bT_length {args.bT_length}")
    print(f"--tsep_list {tsep_list}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
Lt = latt_info.global_size[3]
t_count = Lt if args.t_count <= 0 else args.t_count
tslice_list = [(args.t_start + dt) % Lt for dt in range(t_count)]

for tslice in tslice_list:
    ss_values, sp_values = [], []
    c4_values = {dt: [] for dt in tsep_list}
    quark_pairs = np.asarray([[fw, bw] for fw in quark_mom for bw in quark_mom], dtype=np.int32)
    if len(measurement.pion_channel_pairs) != 1:
        raise ValueError("Dense C4 output requires one fixed pion source/sink channel")
    pair_label = next(iter(measurement.pion_channel_pairs))
    source_label, sink_label = pair_label.split("__")
    source_matrix = measurement.pion_channel_pairs[pair_label][0]
    pos = [0, 0, 0, tslice]
    for quark_mom_fw in quark_mom:
        for quark_mom_bw in quark_mom:
            t0_pair = time.time()
            meson_mom = [
                quark_mom_fw[0] + quark_mom_bw[0],
                quark_mom_fw[1] + quark_mom_bw[1],
                quark_mom_fw[2] + quark_mom_bw[2],
            ]
            mpi_print(latt_info, f"START pion soft factor tslice={tslice} fw={quark_mom_fw} bw={quark_mom_bw} pion_mom={meson_mom}")

            prop_fw_tag = get_pion_soft_factor_prop_file_tag(str(data_dir), lat_tag, conf, "CG.wall", pos, sm_tag, quark_mom_fw)
            prop_bw_src_tag = get_pion_soft_factor_prop_file_tag(str(data_dir), lat_tag, conf, "CG.wall", pos, sm_tag, [-quark_mom_bw[0], -quark_mom_bw[1], -quark_mom_bw[2]])
            prop_fw = measurement.load_wall_propagator(prop_fw_tag)
            prop_bw_src = measurement.load_wall_propagator(prop_bw_src_tag)

            pion_pair_labels = list(measurement.pion_channel_pairs)
            pair_ss = []
            for pion_pair_label in pion_pair_labels:
                t0 = time.time()
                c2pt = measurement.contract_wall_2pt(
                    latt_info, prop_fw, prop_bw_src, quark_mom_fw, quark_mom_bw, pion_pair_label
                )
                qTMDWF = measurement.contract_tmdwf_check(
                    latt_info, prop_fw, prop_bw_src, meson_mom, pion_pair_label
                )
                if latt_info.mpi_rank == 0:
                    c2pt = np.roll(c2pt, -tslice, axis=-1)
                    qTMDWF = np.roll(qTMDWF, -tslice, axis=-1)
                    pair_ss.append(c2pt)
                    sp_values.append(qTMDWF)
                mpi_print(latt_info, f"DONE diagnostic c2pt/qTMDWF pion_pair={pion_pair_label} time={time.time() - t0}s")

            if latt_info.mpi_rank == 0:
                ss_values.append(np.asarray(pair_ss))
            for tsep in tsep_list:
                sink_t = (tslice + tsep) % Lt
                sink_pos = [0, 0, 0, sink_t]
                prop_sink_fw_tag = get_pion_soft_factor_prop_file_tag(str(data_dir), lat_tag, conf, "CG.wall", sink_pos, sm_tag, [-quark_mom_fw[0], -quark_mom_fw[1], -quark_mom_fw[2]])
                prop_sink_bw_tag = get_pion_soft_factor_prop_file_tag(str(data_dir), lat_tag, conf, "CG.wall", sink_pos, sm_tag, quark_mom_bw)
                prop_sink_fw = measurement.load_wall_propagator(prop_sink_fw_tag)
                prop_sink_bw = measurement.load_wall_propagator(prop_sink_bw_tag)
                corr, _, _ = measurement.contract_soft_factor(
                    latt_info,
                    prop_fw,
                    prop_bw_src,
                    prop_sink_bw,
                    prop_sink_fw,
                    meson_mom,
                )
                if latt_info.mpi_rank == 0:
                    corr = np.roll(corr, -tslice, axis=-1)
                    c4_values[tsep].append(corr[0].reshape(
                        len(measurement.gamma_channel_pairs), -1, Lt))
                mpi_print(latt_info, f"DONE tsep={tsep}")

            mpi_print(latt_info, f"DONE pion soft factor pair time={time.time() - t0_pair}s")

    if latt_info.mpi_rank == 0:
        qz = [mom[2] for mom in quark_mom]
        channel_tag = sm_tag + ".src" + source_label.replace("-", "m").replace("+", "p")
        c4_channel_tag = channel_tag + ".sink" + sink_label.replace("-", "m").replace("+", "p")
        attrs = {"config_num": conf, "source_time": tslice,
                 "lattice": latt_info.global_size, "gauge_path": gauge_path.format(conf=conf)}
        def displacements(sp=False):
            result = []
            for direction in bT_dir:
                for b in range(args.bT_length + 1):
                    for z in range((args.bz_length if sp else 0) + 1):
                        vector = [0, 0, 0]
                        vector[direction] += b
                        vector[2] += z
                        result.append(vector)
            return result
        c2pt_tag = get_pion_soft_factor_c2pt_file_tag(
            data_dir, lat_tag, conf, 0, tslice, channel_tag, qz)
        qTMDWF_tag = get_pion_soft_factor_qTMDWF_file_tag(
            data_dir, lat_tag, conf, 0, tslice, channel_tag, qz)
        save_pion_soft_factor_c2pt_hdf5_noRoll(
            np.asarray(ss_values), c2pt_tag, quark_pairs, measurement.pion_channel_pairs, attrs=attrs)
        save_pion_soft_factor_qTMDWF_hdf5_noRoll(
            np.asarray(sp_values), qTMDWF_tag, quark_pairs,
            source_label, source_matrix, displacements(sp=True), attrs=attrs)
        for tsep in tsep_list:
            tag = get_pion_soft_factor_file_tag(data_dir, lat_tag, conf, 0, tslice, c4_channel_tag, qz, tsep)
            save_pion_soft_factor_hdf5_noRoll(
                np.asarray(c4_values[tsep]), tag, quark_pairs, measurement.pion_channel_pairs,
                measurement.gamma_channel_pairs, displacements(), tsep, attrs=attrs)
