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
parser.add_argument("--t_count", type=int, default=int(os.environ.get("PION_SOFT_T_COUNT", 0)))
parser.add_argument("--quark_mom_z", type=str, default=os.environ.get("PION_SOFT_QUARK_MOM_Z", "4,5"))
parser.add_argument("--do_gauge_fix", type=int, default=int(os.environ.get("PION_SOFT_DO_GAUGE_FIX", 0)))
args, unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, io

from pyquda_measurement_utils.io_corr import get_pion_soft_factor_prop_file_tag
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
momenta_to_save = []
for mom in quark_mom:
    momenta_to_save.append(mom)
    momenta_to_save.append([-mom[0], -mom[1], -mom[2]])
momenta_to_save = [list(mom) for mom in dict.fromkeys(tuple(mom) for mom in momenta_to_save)]

parameters = {
    "quark_mom": quark_mom,
    "bT_dir": [0],
    "bT_length": 1,
    "bz_length": 0,
    "tsep_list": [2],
}
measurement = pion_soft_factor(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "pion_soft_factor_prop").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--config_num {conf}")
    print(f"--mpi_geometry {args.mpi_geometry}")
    print(f"--quark_mom {quark_mom}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
if args.do_gauge_fix:
    gauge.fixingOVR(3, 50000, 1000, 1.5, 1e-7, 100, 1)
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
mass = float(os.environ.get("PION_SOFT_MASS", 0.236))
csw_r = float(os.environ.get("PION_SOFT_CSW", 1.0372))
csw_t = csw_r
tol = float(os.environ.get("PION_SOFT_TOL", 1e-15))
maxiter = int(os.environ.get("PION_SOFT_MAXITER", 300))
multigrid = [[max(1, latt_info.global_size[0] // 1), max(1, latt_info.global_size[1] // 1), max(1, latt_info.global_size[2] // 2), max(1, latt_info.global_size[3] // 8)]]

dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)

Lt = latt_info.global_size[3]
t_count = Lt if args.t_count <= 0 else args.t_count
tslice_list = [(args.t_start + dt) % Lt for dt in range(t_count)]

for tslice in tslice_list:
    src_pos = [0, 0, 0, tslice]
    for mom in momenta_to_save:
        t0 = time.time()
        tag = get_pion_soft_factor_prop_file_tag(str(data_dir), lat_tag, conf, "CG.wall", src_pos, sm_tag, mom)
        if Path(tag + ".h5").exists() and int(os.environ.get("PION_SOFT_OVERWRITE_PROP", 0)) == 0:
            mpi_print(latt_info, f"SKIP existing wall propagator: {tag}.h5")
            continue
        mpi_print(latt_info, f"START wall propagator tslice={tslice} momentum={mom}")
        prop = measurement.create_wall_propagator(dirac, latt_info, tslice, mom)
        measurement.save_wall_propagator(
            prop,
            tag,
            attrs={
                "lat_tag": lat_tag,
                "config_num": conf,
                "sm_tag": sm_tag,
                "tslice": tslice,
                "quark_momentum": np.asarray(mom, dtype=np.int32),
            },
        )
        mpi_print(latt_info, f"DONE wall propagator tslice={tslice} momentum={mom} time={time.time() - t0}s")
