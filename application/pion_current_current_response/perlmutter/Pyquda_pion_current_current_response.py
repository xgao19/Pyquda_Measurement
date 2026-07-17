import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


def parse_momentum(text):
    values = [int(item) for item in str(text).replace(",", ".").split(".") if item]
    if len(values) != 3:
        raise ValueError(f"Expected three momentum components, got {text!r}")
    return values


def parse_position(text):
    values = [int(item) for item in str(text).replace(",", ".").split(".") if item]
    if len(values) != 4:
        raise ValueError(f"Expected four source-position components, got {text!r}")
    return values


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("PION_CC_RESPONSE_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("PION_CC_RESPONSE_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("PION_CC_RESPONSE_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("PION_CC_RESPONSE_DATA_DIR", ""))
parser.add_argument("--pf", type=str, default=os.environ.get("PION_CC_RESPONSE_PF", "0.0.0"))
parser.add_argument("--first_qext", type=str, default=os.environ.get("PION_CC_RESPONSE_FIRST_QEXT", "0.0.1"))
parser.add_argument("--second_qext", type=str, default=os.environ.get("PION_CC_RESPONSE_SECOND_QEXT", "0.0.-1"))
parser.add_argument("--tsep", type=int, default=int(os.environ.get("PION_CC_RESPONSE_TSEP", 2)))
parser.add_argument("--first_current_gamma", type=str, default=os.environ.get("PION_CC_RESPONSE_FIRST_GAMMA", "T"))
parser.add_argument("--second_current_gamma", type=str, default=os.environ.get("PION_CC_RESPONSE_SECOND_GAMMA", "T"))
parser.add_argument("--first_tau_window", type=str, default=os.environ.get("PION_CC_RESPONSE_FIRST_TAU_WINDOW", "restricted"))
parser.add_argument("--second_tau_window", type=str, default=os.environ.get("PION_CC_RESPONSE_SECOND_TAU_WINDOW", "restricted"))
parser.add_argument("--first_tau_min", type=int, default=int(os.environ.get("PION_CC_RESPONSE_FIRST_TAU_MIN", 1)))
parser.add_argument("--second_tau_min", type=int, default=int(os.environ.get("PION_CC_RESPONSE_SECOND_TAU_MIN", 1)))
parser.add_argument("--width", type=float, default=float(os.environ.get("PION_CC_RESPONSE_WIDTH", 1.0)))
parser.add_argument("--mass", type=float, default=float(os.environ.get("PION_CC_RESPONSE_MASS", 0.236)))
parser.add_argument("--tol", type=float, default=float(os.environ.get("PION_CC_RESPONSE_TOL", 1e-15)))
parser.add_argument("--maxiter", type=int, default=int(os.environ.get("PION_CC_RESPONSE_MAXITER", 300)))
parser.add_argument("--src_pos", type=str, default=os.environ.get("PION_CC_RESPONSE_SRC_POS", "0.0.0.0"))
args, _unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.pion_current_background_response_vibe_develop import (
    contract_current_current_response_pion_2pt,
    infer_source_momentum,
    invert_current_current_response_propagator,
    response_at_sink_time,
    response_ratio,
    relative_tau_to_absolute,
    roll_to_source_relative,
    save_pion_current_current_response_hdf5,
    tau_window_list,
)
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt_multi_src_gamma,
    source_gamma_provenance,
)
from pyquda_measurement_utils.tools import mpi_print


software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("PION_CC_RESPONSE_LAT_TAG", "S8T32")
conf = args.config_num

pf = parse_momentum(args.pf)
first_qext = parse_momentum(args.first_qext)
second_qext = parse_momentum(args.second_qext)
total_qext = [first_qext[i] + second_qext[i] for i in range(3)]
src_pos = parse_position(args.src_pos)
sink_gamma_label = "5"
src_gamma = "5"

if getMPIComm().Get_rank() == 0:
    (data_dir / "current_current_response").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--pf {pf}")
    print(f"--first_qext {first_qext}")
    print(f"--second_qext {second_qext}")
    print(f"--total_qext {total_qext}")
    print(f"--tsep {args.tsep}")
    print(f"--first_current_gamma {args.first_current_gamma}")
    print(f"--second_current_gamma {args.second_current_gamma}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
csw_r = float(os.environ.get("PION_CC_RESPONSE_CSW", 1.0372))
csw_t = csw_r
multigrid = [[
    max(1, latt_info.global_size[0] // 1),
    max(1, latt_info.global_size[1] // 1),
    max(1, latt_info.global_size[2] // 2),
    max(1, latt_info.global_size[3] // 8),
]]
dirac = core.getDirac(latt_info, args.mass, args.tol, args.maxiter, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)

t0 = time.time()
srcD = source.propagator(latt_info, "point", src_pos)
srcD_pos = boosted_smearing(srcD, w=args.width, boost=[0, 0, 0])
prop_pos = core.invertPropagator(dirac, srcD_pos, 1, 0)
prop_neg = prop_pos.copy()
mpi_print(latt_info, f"TIME ordinary pion source propagator {time.time() - t0}s")

sink_phases = MomentumPhase(latt_info).getPhases([[-pf[0], -pf[1], -pf[2]]], src_pos)
first_phases = MomentumPhase(latt_info).getPhases([first_qext], src_pos)
second_phases = MomentumPhase(latt_info).getPhases([second_qext], src_pos)

t0 = time.time()
prop_pos_sink = boosted_smearing(prop_pos.copy(), w=args.width, boost=[0, 0, 0])
prop_neg_sink = boosted_smearing(prop_neg.copy(), w=args.width, boost=[0, 0, 0])
c2_corr = contract_pion_2pt_multi_src_gamma(
    latt_info,
    prop_pos_sink,
    prop_neg_sink,
    sink_phases,
    [src_gamma],
)[src_gamma]
c2_corr = getMPIComm().bcast(c2_corr, root=0)
c2_corr = roll_to_source_relative(c2_corr, src_pos[3])
mpi_print(latt_info, f"TIME ordinary pion C2 {time.time() - t0}s")

first_tau_relative_list = tau_window_list(
    args.tsep,
    latt_info.global_size[3],
    args.first_tau_window,
    args.first_tau_min,
)
second_tau_relative_list = tau_window_list(
    args.tsep,
    latt_info.global_size[3],
    args.second_tau_window,
    args.second_tau_min,
)
first_tau_absolute_list = relative_tau_to_absolute(
    first_tau_relative_list,
    src_pos[3],
    latt_info.global_size[3],
)
second_tau_absolute_list = relative_tau_to_absolute(
    second_tau_relative_list,
    src_pos[3],
    latt_info.global_size[3],
)

t0 = time.time()
cc_response_prop = invert_current_current_response_propagator(
    dirac,
    prop_pos,
    first_phases[0],
    second_phases[0],
    source_time=src_pos[3],
    first_current_gamma=args.first_current_gamma,
    second_current_gamma=args.second_current_gamma,
    first_tau_relative_list=first_tau_relative_list,
    second_tau_relative_list=second_tau_relative_list,
    response_sign=1,
)
cc_response_prop = boosted_smearing(cc_response_prop, w=args.width, boost=[0, 0, 0])
cc_response_corr = contract_current_current_response_pion_2pt(
    latt_info,
    cc_response_prop,
    prop_neg_sink,
    sink_phases,
    src_gamma=src_gamma,
)
cc_response_corr = getMPIComm().bcast(cc_response_corr, root=0)
cc_response_corr = roll_to_source_relative(cc_response_corr, src_pos[3])
cc_response_value = response_at_sink_time(
    cc_response_corr,
    sink_gamma=sink_gamma_label,
    p_index=0,
    tsep=args.tsep,
)
mpi_print(latt_info, f"TIME current-current response {time.time() - t0}s")

c2_value = response_at_sink_time(c2_corr, sink_gamma=sink_gamma_label, p_index=0, tsep=args.tsep)
records = [
    {
        "first_current_gamma": args.first_current_gamma,
        "second_current_gamma": args.second_current_gamma,
        "sink_gamma": sink_gamma_label,
        "src_gamma": src_gamma,
        "first_tau_window": args.first_tau_window,
        "second_tau_window": args.second_tau_window,
        "first_tau_min": args.first_tau_min,
        "second_tau_min": args.second_tau_min,
        "first_tau_relative_list": first_tau_relative_list,
        "first_tau_absolute_list": first_tau_absolute_list,
        "second_tau_relative_list": second_tau_relative_list,
        "second_tau_absolute_list": second_tau_absolute_list,
        "response_sign": 1,
        "pf": pf,
        "first_qext": first_qext,
        "second_qext": second_qext,
        "total_qext": total_qext,
        "pi": infer_source_momentum(pf, total_qext),
        "tsep": args.tsep,
        "c2_tsep": c2_value,
        "response_c2_like": cc_response_value,
        "response_R_sum": response_ratio(cc_response_value, c2_value),
        "response_corr_all_t": np.asarray(cc_response_corr)[0, 0],
        "c2_all_t": np.asarray(c2_corr)[0, 0],
    }
]

if latt_info.mpi_rank == 0:
    out_tag = data_dir / "current_current_response" / (
        f"{lat_tag}.pion_current_current_response.{conf}.src{src_gamma}"
        f".x{src_pos[0]}y{src_pos[1]}z{src_pos[2]}t{src_pos[3]}"
        f".pf{pf[0]}_{pf[1]}_{pf[2]}"
        f".q1{first_qext[0]}_{first_qext[1]}_{first_qext[2]}"
        f".q2{second_qext[0]}_{second_qext[1]}_{second_qext[2]}.dt{args.tsep}"
    )
    save_pion_current_current_response_hdf5(
        str(out_tag),
        records,
        attrs={
            "lat_tag": lat_tag,
            "config_num": conf,
            "source_position": np.asarray(src_pos, dtype=np.int32),
            "source_time": int(src_pos[3]),
            "time_axis": "source_relative",
            "no_per_tau_response_propagator_cache": True,
            **source_gamma_provenance(src_gamma),
        },
    )
    print("[pion current-current response]")
    print(
        f"gamma1={args.first_current_gamma} gamma2={args.second_current_gamma} "
        f"pf={pf} q1={first_qext} q2={second_qext} total_q={total_qext} "
        f"pi={records[0]['pi']} tsep={args.tsep} R_response={records[0]['response_R_sum']}"
    )
    print(f"output = {out_tag}.h5")
