import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


def parse_boost(text):
    return [int(i) for i in text.split(".")]


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("PION_EMFF_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("PION_EMFF_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("PION_EMFF_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("PION_EMFF_DATA_DIR", ""))
parser.add_argument("--num_src", type=int, default=int(os.environ.get("PION_EMFF_NUM_SRC", 1)))
parser.add_argument("--qmax", type=int, default=int(os.environ.get("PION_EMFF_QMAX", 1)))
parser.add_argument("--pf", type=str, default=os.environ.get("PION_EMFF_PF", "0.0.0"))
parser.add_argument("--t_insert", type=str, default=os.environ.get("PION_EMFF_T_INSERT", "2"))
parser.add_argument("--width", type=float, default=float(os.environ.get("PION_EMFF_WIDTH", 1.0)))
parser.add_argument("--pos_boost_src", type=str, default=os.environ.get("PION_EMFF_POS_BOOST_SRC", os.environ.get("PION_EMFF_POS_BOOST", "0.0.0")))
parser.add_argument("--pos_boost_sink", type=str, default=os.environ.get("PION_EMFF_POS_BOOST_SINK", os.environ.get("PION_EMFF_POS_BOOST", "0.0.0")))
parser.add_argument("--neg_boost_src", type=str, default=os.environ.get("PION_EMFF_NEG_BOOST_SRC", os.environ.get("PION_EMFF_NEG_BOOST", "0.0.0")))
parser.add_argument("--neg_boost_sink", type=str, default=os.environ.get("PION_EMFF_NEG_BOOST_SINK", os.environ.get("PION_EMFF_NEG_BOOST", "0.0.0")))
parser.add_argument("--src_interpolator", type=str, default=os.environ.get("PION_EMFF_SRC_INTERPOLATOR", "fixed_g5"))
parser.add_argument("--src_interpolators", type=str, default=os.environ.get("PION_EMFF_SRC_INTERPOLATORS", ""))
parser.add_argument("--sink_interpolator", type=str, default=os.environ.get("PION_EMFF_SINK_INTERPOLATOR", "5"))
args, unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, gamma, io, phase, source
from pyquda_utils.phase import MomentumPhase

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.bw_seq_pyquda import create_meson_bw_seq_pyquda
from pyquda_measurement_utils.io_corr import (
    get_c2pt_file_tag,
    get_pion_EMFF_file_tag,
    get_sample_log_tag,
    save_pion_EMFF_hdf5_noRoll,
)
from pyquda_measurement_utils.pion_EMFF_vibe_develop import my_gammas, pion_EMFF
from pyquda_measurement_utils.tools import mpi_print, srcLoc_distri_eq


def sync_backend_array(arr):
    stream = getattr(arr, "stream", None)
    if stream is not None:
        stream.synchronize()
    queue = getattr(arr, "sycl_queue", None)
    if queue is not None:
        queue.wait()


def gamma_from_label(label):
    gamma_map = {
        "5": 15,
        "T": 8,
        "T5": 7,
        "X": 1,
        "X5": 14,
        "Y": 2,
        "Y5": 13,
        "Z": 4,
        "Z5": 11,
        "I": 0,
        "SXT": 9,
        "SXY": 3,
        "SXZ": 5,
        "SYT": 10,
        "SYZ": 6,
        "SZT": 12,
    }
    if label not in gamma_map:
        raise ValueError(f"Invalid sink interpolator: {label}")
    return gamma.gamma(gamma_map[label])


def parse_src_interpolators(text, fallback):
    if not text:
        return [fallback]
    return [item for item in text.replace(",", ".").split(".") if item]


def parse_int_list(text):
    return [int(item) for item in str(text).replace(",", ".").split(".") if item]


software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("PION_EMFF_LAT_TAG", "S8T32")
conf = args.config_num

pf = parse_boost(args.pf) + [0]
t_insert_list = parse_int_list(args.t_insert)
if not t_insert_list:
    raise ValueError("--t_insert must contain at least one integer, for example 2 or 2.4")
q_range = range(-args.qmax, args.qmax + 1)
qext = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
p_2pt = qext
parameters = {
    "qext": qext,
    "pf": pf,
    "p_2pt": p_2pt,
    "pos_boost_src": parse_boost(args.pos_boost_src),
    "pos_boost_sink": parse_boost(args.pos_boost_sink),
    "neg_boost_src": parse_boost(args.neg_boost_src),
    "neg_boost_sink": parse_boost(args.neg_boost_sink),
    "width": args.width,
    "t_insert": t_insert_list,
    "save_propagators": False,
}
boost_tag = (
    f"posSrc{''.join(str(v) for v in parameters['pos_boost_src'])}"
    f"_posSink{''.join(str(v) for v in parameters['pos_boost_sink'])}"
    f"_negSrc{''.join(str(v) for v in parameters['neg_boost_src'])}"
    f"_negSink{''.join(str(v) for v in parameters['neg_boost_sink'])}"
)
c2_boost_tag = (
    f"posSrc{''.join(str(v) for v in parameters['pos_boost_src'])}"
    f"_negSrc{''.join(str(v) for v in parameters['neg_boost_src'])}"
)
sm_tag = os.environ.get("PION_EMFF_SM_TAG", f"1HYP_GSRC_W{args.width:g}_k0_{args.sink_interpolator}.{boost_tag}")
c2_sm_tag = f"1HYP_GSRC_W{args.width:g}_k0_{args.sink_interpolator}.{c2_boost_tag}"
src_interpolators = parse_src_interpolators(args.src_interpolators, args.src_interpolator)
measurement = pion_EMFF(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "sample_log_emff").mkdir(parents=True, exist_ok=True)
    (data_dir / "c2pt").mkdir(parents=True, exist_ok=True)
    (data_dir / "pion_EMFF").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--config_num {conf}")
    print(f"--mpi_geometry {args.mpi_geometry}")
    print(f"--t_insert {t_insert_list}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
mass = float(os.environ.get("PION_EMFF_MASS", 0.236))
csw_r = float(os.environ.get("PION_EMFF_CSW", 1.0372))
csw_t = csw_r
tol = float(os.environ.get("PION_EMFF_TOL", 1e-15))
maxiter = int(os.environ.get("PION_EMFF_MAXITER", 300))
multigrid = [[max(1, latt_info.global_size[0] // 1), max(1, latt_info.global_size[1] // 1), max(1, latt_info.global_size[2] // 2), max(1, latt_info.global_size[3] // 8)]]

dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)

L = latt_info.global_size
src_shift = np.array([0, 0, 0, 0])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)[: args.num_src]

sink_gamma = gamma_from_label(args.sink_interpolator)

for pos in src_positions:
    t0_pos = time.time()
    mpi_print(latt_info, f"START source: {pos} tseps {t_insert_list}")

    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcD_pos = boosted_smearing(srcD, w=parameters["width"], boost=parameters["pos_boost_src"])
    mpi_print(latt_info, f"TIME PyQUDA: Generating positive boosted source {time.time() - t0}s")

    t0 = time.time()
    prop_pos = core.invertPropagator(dirac, srcD_pos, 1, 0)
    mpi_print(latt_info, f"TIME PyQUDA: Positive propagator inversion {time.time() - t0}s")

    if parameters["neg_boost_src"] == parameters["pos_boost_src"]:
        prop_neg = prop_pos.copy()
    else:
        t0 = time.time()
        srcD_neg = boosted_smearing(srcD, w=parameters["width"], boost=parameters["neg_boost_src"])
        prop_neg = core.invertPropagator(dirac, srcD_neg, 1, 0)
        mpi_print(latt_info, f"TIME PyQUDA: Negative propagator inversion {time.time() - t0}s")

    t0 = time.time()
    c2_tag = get_c2pt_file_tag(str(data_dir), lat_tag, conf, "EMFF.ex", pos, c2_sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    c2_tags_by_src = {
        src_interpolator: f"{c2_tag}.src{src_interpolator}"
        for src_interpolator in src_interpolators
    }
    measurement.contract_2pt_pion_multi_src_gamma(latt_info, prop_pos.copy(), prop_neg.copy(), phases_2pt, c2_tags_by_src)
    mpi_print(latt_info, f"TIME PyQUDA: Pion 2pt contraction {time.time() - t0}s")

    t0 = time.time()
    prop_neg_sink = boosted_smearing(prop_neg.copy(), w=parameters["width"], boost=parameters["neg_boost_sink"])
    mpi_print(latt_info, f"TIME PyQUDA: Pion EMFF sink smearing {time.time() - t0}s")

    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_EMFF = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)

    for t_insert in t_insert_list:
        pf_tag = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{t_insert}"
        sample_log_file = data_dir / "sample_log_emff" / f"{conf}_{sm_tag}_{pf_tag}"
        sample_log_tag = get_sample_log_tag(str(conf), pos, f"{sm_tag}_{pf_tag}")
        if latt_info.mpi_rank == 0:
            sample_log_file.touch(exist_ok=True)
        mpi_print(latt_info, f"START: {sample_log_tag}")

        t0_tsep = time.time()
        t0 = time.time()
        seq_bw_prop = create_meson_bw_seq_pyquda(
            dirac,
            prop_neg_sink.copy(),
            pos,
            parameters["pf"],
            t_insert,
            sink_gamma,
            parameters["width"],
            parameters["pos_boost_sink"],
        )
        mpi_print(latt_info, f"TIME PyQUDA: Pion EMFF sequential propagator dt{t_insert} {time.time() - t0}s")

        t0 = time.time()
        pion_EMFFs_by_src = measurement.contract_EMFF_multi_src_gamma(
            latt_info,
            prop_pos,
            seq_bw_prop,
            phases_EMFF,
            src_interpolators,
        )
        mpi_print(latt_info, f"contract_EMFF dt{t_insert} over: src_interpolators {src_interpolators} {time.time() - t0}s")

        for src_interpolator in src_interpolators:
            pion_EMFFs = pion_EMFFs_by_src[src_interpolator]
            mpi_print(latt_info, f"pion_EMFFs[{src_interpolator}].shape {np.shape(pion_EMFFs)}")
            if latt_info.mpi_rank == 0:
                pion_EMFFs = np.roll(pion_EMFFs, -pos[3], axis=-1)
                pion_EMFFs = pion_EMFFs[:, :, : t_insert + 2]
                pion_EMFFs = np.transpose(pion_EMFFs, (1, 0, 2))
                tag = get_pion_EMFF_file_tag(
                    str(data_dir),
                    lat_tag,
                    conf,
                    "EMFF.ex",
                    pos,
                    f"{sm_tag}.src{src_interpolator}.{pf_tag}",
                )
                mpi_print(latt_info, f"Saving pion EMFF src {src_interpolator}: {tag}")
                save_pion_EMFF_hdf5_noRoll(
                    pion_EMFFs,
                    tag,
                    my_gammas,
                    parameters["qext"],
                    t_insert,
                    latt_info,
                )

        if latt_info.mpi_rank == 0:
            with sample_log_file.open("a+") as f:
                f.write(sample_log_tag + "\n")
        mpi_print(latt_info, f"DONE: {sample_log_tag} total {time.time() - t0_tsep}s")
    sync_backend_array(prop_pos.data)

    mpi_print(latt_info, f"DONE source: {pos} total {time.time() - t0_pos}s")
