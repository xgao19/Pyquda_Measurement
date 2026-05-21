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


def parse_momentum_list(text):
    return [parse_momentum(item) for item in str(text).split(";") if item]


def parse_int_list(text):
    return [int(item) for item in str(text).replace(",", ".").split(".") if item]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("PION_EMFF_BG_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("PION_EMFF_BG_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("PION_EMFF_BG_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("PION_EMFF_BG_DATA_DIR", ""))
parser.add_argument("--pf", type=str, default=os.environ.get("PION_EMFF_BG_PF", "0.0.0"))
parser.add_argument("--qext", type=str, default=os.environ.get("PION_EMFF_BG_QEXT", "0.0.0"))
parser.add_argument("--qext_list", type=str, default=os.environ.get("PION_EMFF_BG_QEXT_LIST", ""))
parser.add_argument("--tsep", type=int, default=int(os.environ.get("PION_EMFF_BG_TSEP", 2)))
parser.add_argument("--tsep_list", type=str, default=os.environ.get("PION_EMFF_BG_TSEP_LIST", ""))
parser.add_argument("--current_gammas", type=str, default=os.environ.get("PION_EMFF_BG_CURRENT_GAMMAS", "T"))
parser.add_argument("--tau_window", type=str, default=os.environ.get("PION_EMFF_BG_TAU_WINDOW", "all"))
parser.add_argument("--tau_min", type=int, default=int(os.environ.get("PION_EMFF_BG_TAU_MIN", 1)))
parser.add_argument("--width", type=float, default=float(os.environ.get("PION_EMFF_BG_WIDTH", 1.0)))
parser.add_argument("--mass", type=float, default=float(os.environ.get("PION_EMFF_BG_MASS", 0.236)))
parser.add_argument("--tol", type=float, default=float(os.environ.get("PION_EMFF_BG_TOL", 1e-15)))
parser.add_argument("--maxiter", type=int, default=int(os.environ.get("PION_EMFF_BG_MAXITER", 300)))
args, _unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, gamma, io, phase, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_meson_bw_seq_pyquda
from pyquda_measurement_utils.io_corr import ensure_parent_dir
from pyquda_measurement_utils.pion_EMFF_background_response_vibe_develop import (
    contract_response_pion_2pt,
    infer_source_momentum,
    invert_local_current_response_propagator,
    response_at_sink_time,
    response_ratio,
    save_pion_EMFF_background_response_hdf5,
    summed_explicit_emff,
    tau_window_list,
)
from pyquda_measurement_utils.pion_EMFF_vibe_develop import pion_EMFF
from pyquda_measurement_utils.pion_utils_vibe_develop import contract_pion_2pt_multi_src_gamma, my_gammas
from pyquda_measurement_utils.tools import mpi_print


software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("PION_EMFF_BG_LAT_TAG", "S8T32")
conf = args.config_num

pf = parse_momentum(args.pf)
qext_list = parse_momentum_list(args.qext_list) if args.qext_list else [parse_momentum(args.qext)]
tsep_list = parse_int_list(args.tsep_list) if args.tsep_list else [args.tsep]
current_gammas = [item for item in args.current_gammas.replace(",", ".").split(".") if item]
src_pos = [0, 0, 0, 0]
sink_gamma_label = "5"
src_gamma = "fixed_g5"

parameters = {
    "qext": [qext + [0] for qext in qext_list],
    "pf": pf + [0],
    "p_2pt": [pf + [0]],
    "pos_boost_src": [0, 0, 0],
    "pos_boost_sink": [0, 0, 0],
    "neg_boost_src": [0, 0, 0],
    "neg_boost_sink": [0, 0, 0],
    "width": args.width,
    "t_insert": tsep_list,
    "save_propagators": False,
}
measurement = pion_EMFF(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "background_response").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--pf {pf}")
    print(f"--qext_list {qext_list}")
    print(f"--tsep_list {tsep_list}")
    print(f"--current_gammas {current_gammas}")
    print(f"--tau_window {args.tau_window}")
    print(f"--tau_min {args.tau_min}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
csw_r = float(os.environ.get("PION_EMFF_BG_CSW", 1.0372))
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
mpi_print(latt_info, f"TIME ordinary pion C2 {time.time() - t0}s")

records = []

for iq, qext in enumerate(qext_list):
    q_phases = phase.MomentumPhase(latt_info).getPhases([qext], src_pos)
    for tsep in tsep_list:
        t0 = time.time()
        seq_bw_prop = create_meson_bw_seq_pyquda(
            dirac,
            prop_neg_sink.copy(),
            src_pos,
            pf,
            tsep,
            gamma.gamma(15),
            args.width,
            [0, 0, 0],
        )
        c3_by_src = measurement.contract_EMFF_multi_src_gamma(
            latt_info,
            prop_pos,
            seq_bw_prop,
            q_phases,
            [src_gamma],
        )
        c3 = c3_by_src[src_gamma]
        mpi_print(latt_info, f"TIME explicit EMFF q={qext} tsep={tsep}: {time.time() - t0}s")

        tau_list = tau_window_list(tsep, latt_info.global_size[3], args.tau_window, args.tau_min)
        c2_value = response_at_sink_time(c2_corr, sink_gamma=sink_gamma_label, p_index=0, tsep=tsep)

        for current_gamma in current_gammas:
            explicit_sum = summed_explicit_emff(c3, current_gamma=current_gamma, q_index=0, tau_list=tau_list)

            t0 = time.time()
            response_prop = invert_local_current_response_propagator(
                dirac,
                prop_pos,
                q_phases[0],
                current_gamma=current_gamma,
                tau_list=tau_list,
                response_sign=1,
            )
            response_prop = boosted_smearing(response_prop, w=args.width, boost=[0, 0, 0])
            response_corr = contract_response_pion_2pt(
                latt_info,
                response_prop,
                prop_neg_sink,
                sink_phases,
                src_gamma=src_gamma,
            )
            response_value = response_at_sink_time(
                response_corr,
                sink_gamma=sink_gamma_label,
                p_index=0,
                tsep=tsep,
            )
            mpi_print(
                latt_info,
                f"TIME response EMFF q={qext} tsep={tsep} gamma={current_gamma}: {time.time() - t0}s",
            )

            diff = response_value - explicit_sum
            denom = max(abs(explicit_sum), 1e-300)
            rel_diff = abs(diff) / denom
            records.append(
                {
                    "current_gamma": current_gamma,
                    "sink_gamma": sink_gamma_label,
                    "src_gamma": src_gamma,
                    "tau_window": args.tau_window,
                    "tau_min": args.tau_min,
                    "tau_list": tau_list,
                    "response_sign": 1,
                    "finite_difference_derivative_sign": -1,
                    "pf": pf,
                    "qext": qext,
                    "pi": infer_source_momentum(pf, qext),
                    "tsep": tsep,
                    "q_index": iq,
                    "c2_tsep": c2_value,
                    "explicit_summed_c3": explicit_sum,
                    "response_c2_like": response_value,
                    "response_R_sum": response_ratio(response_value, c2_value),
                    "explicit_R_sum": response_ratio(explicit_sum, c2_value),
                    "difference": diff,
                    "relative_difference": rel_diff,
                    "explicit_c3_all_tau": np.asarray(c3)[my_gammas.index(current_gamma), 0],
                    "response_corr_all_t": np.asarray(response_corr)[0, 0],
                    "c2_all_t": np.asarray(c2_corr)[0, 0],
                }
            )

if latt_info.mpi_rank == 0:
    out_tag = data_dir / "background_response" / (
        f"{lat_tag}.pion_EMFF_background_response.{conf}.pf{pf[0]}_{pf[1]}_{pf[2]}"
        f".nq{len(qext_list)}.ntsep{len(tsep_list)}.{args.tau_window}"
    )
    out = Path(f"{out_tag}.h5")
    ensure_parent_dir(out)
    save_pion_EMFF_background_response_hdf5(
        str(out_tag),
        records,
        attrs={
            "lat_tag": lat_tag,
            "config_num": conf,
            "src_pos": np.asarray(src_pos, dtype=np.int32),
            "pf": np.asarray(pf, dtype=np.int32),
            "qext_list": np.asarray(qext_list, dtype=np.int32),
            "tsep_list": np.asarray(tsep_list, dtype=np.int32),
            "current_gamma_list": np.asarray(current_gammas, dtype="S"),
            "tau_window": args.tau_window,
            "tau_min": args.tau_min,
            "no_per_tau_response_propagator_cache": True,
        },
    )
    print("[pion EMFF background response]")
    for record in records:
        print(
            "gamma={current_gamma} pf={pf} qext={qext} pi={pi} tsep={tsep} "
            "window={tau_window} R_response={response_R_sum} R_explicit={explicit_R_sum} "
            "rel_diff={relative_difference}".format(**record)
        )
    print(f"output = {out}")
