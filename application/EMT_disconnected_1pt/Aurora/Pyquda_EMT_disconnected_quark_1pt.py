import argparse
from pathlib import Path
from time import perf_counter

from pyquda import getMPIComm, init
from pyquda_utils import io

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    aggregate_emt_quark_1pt_bases,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    NESTED_HP_ORDERING,
    disconnected_sample_log_path,
    effective_n_inversions,
)
from pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop import (
    parse_optional_multigrid_blocks,
)
from pyquda_measurement_utils.io_corr import get_emt_quark_loop_file_tag
from pyquda_measurement_utils.tools import mpi_print, mpi_timer_print


# Physics and estimator settings; matching the connected a076 setup except mass.
LAT_TAG = "l64c64a076"
GAUGE_PATH_TEMPLATE = (
    "/lus/flare/projects/StructNGB/ensemble/"
    "l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/"
    "l6464f21b7130m00119m0322a.{config}.coulomb.1e-14"
)
MASS = -0.015
CSW = 1.0372
TOL = 1e-10
MAXITER = 5000
QMAX = 2
FLOW_TYPE = "symanzik"
FLOW_EPSILON = 0.09
FLOW_STEPS = 10
DEFAULT_MG_BLOCK = "4.4.4.4;4.4.2.2"
HYP_PROJECT = -1
NOISE_SCHEME = "hierarchical_probing"
HP_NUM_VECTORS = 512
SPIN_COLOR_DILUTION = "point"
N_ZN = 4
RAND_SEED = 0
# Sixty-four complete spin-color-diluted probes per shard.
BLOCK_INTERVAL_SOLVES = 768


class TimedStrangeBenchmark(EMTDisconnectedQuark1pt):
    """Print completed-source timings without waiting for a shard write."""

    def _invert_and_measure_batch(
        self, U, dirac, source_records, momentum_projectors, timers=None,
        restore_original_gauge=True,
    ):
        source_records = list(source_records)
        source_timers = self._new_batch_timers()
        start = perf_counter()
        raw = super()._invert_and_measure_batch(
            U, dirac, source_records, momentum_projectors,
            timers=source_timers,
            restore_original_gauge=restore_original_gauge,
        )
        total = perf_counter() - start
        # Preserve the library's cumulative shard timing and measurement results.
        if timers is not None:
            for key, value in source_timers.items():
                timers[key] += value
        count = len(source_records)
        fields = {
            "source_first": source_records[0][0],
            "source_last": source_records[-1][0],
            "base": source_records[0][1],
            "probe": source_records[0][2],
            "spin": source_records[0][4],
            "color": source_records[0][5],
            "sources": count,
            "first_batch": not restore_original_gauge,
        }
        restore = float(source_timers["restore"])
        inversion = float(source_timers["invert"])
        flow = float(source_timers["flow"].sum())
        contraction = float(source_timers["contract"].sum())
        mpi_timer_print(
            U.latt_info, "strange_source_compute", total, **fields,
            per_source=total / count,
            restore_per_source=restore / count,
            invert_per_source=inversion / count,
            flow_per_source=flow / count,
            contract_per_source=contraction / count,
            other_per_source=(total - restore - inversion - flow - contraction) / count,
        )
        for stage in ("flow", "contract"):
            mpi_timer_print(
                U.latt_info, f"strange_source_{stage}",
                float(source_timers[stage].sum()), **fields,
                per_step_per_source=[
                    round(float(value) / count, 6) for value in source_timers[stage]
                ],
            )
        return raw


def main():
    parser = argparse.ArgumentParser(description="Strange EMT 1pt HP512/SC12 benchmark")
    parser.add_argument("--config_num", type=int, default=1050)
    parser.add_argument("--mpi_geometry", default="2.2.4.4")
    parser.add_argument("--mg-block", default=DEFAULT_MG_BLOCK)
    parser.add_argument("--n-base-noise", type=int, default=1)
    parser.add_argument("--flow-batch-size", type=int, default=24)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--resource-path", type=Path)
    args = parser.parse_args()
    if args.n_base_noise <= 0 or args.flow_batch_size <= 0:
        parser.error("--n-base-noise and --flow-batch-size must be positive")

    benchmark_dir = Path(__file__).resolve().parent
    data_dir = (args.data_dir or benchmark_dir / "data").resolve()
    resource_path = (args.resource_path or benchmark_dir / "cache").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    resource_path.mkdir(parents=True, exist_ok=True)
    mpi_geometry = [int(value) for value in args.mpi_geometry.split(".")]
    qext = [
        [qx, qy, qz, 0]
        for qx in range(-QMAX, QMAX + 1)
        for qy in range(-QMAX, QMAX + 1)
        for qz in range(-QMAX, QMAX + 1)
    ]
    setup_tag = (
        f"strange_m{MASS:g}_1HYP_Z4_HP{HP_NUM_VECTORS}_SC12_nested_"
        f"Q{QMAX}_{FLOW_TYPE}_eps{FLOW_EPSILON:g}_F{FLOW_STEPS}"
    )
    canonical_tag = get_emt_quark_loop_file_tag(
        str(data_dir), LAT_TAG, args.config_num, 0, setup_tag
    )
    shard_dir = data_dir / "EMTquarkLoop" / "shards"
    sample_log_file = disconnected_sample_log_path(data_dir, canonical_tag)

    init(
        mpi_geometry,
        enable_mps=True,
        backend="dpnp",
        backend_target="sycl",
        resource_path=str(resource_path),
    )
    gauge_path = GAUGE_PATH_TEMPLATE.format(config=args.config_num)
    gauge = io.readNERSCGauge(
        gauge_path, checksum=False, link_trace=False, plaquette=False
    )
    gauge.hypSmear(1, 0.75, 0.6, 0.3, HYP_PROJECT)
    gauge.latt_info.t_boundary = -1
    latt_info = gauge.latt_info

    parameters = {
        "config_num": args.config_num,
        "qext": qext,
        "flow_type": FLOW_TYPE,
        "flow_epsilon": FLOW_EPSILON,
        "flow_steps": FLOW_STEPS,
        "noise_scheme": NOISE_SCHEME,
        "hp_num_vectors": HP_NUM_VECTORS,
        "hp_ordering": NESTED_HP_ORDERING,
        "spin_color_dilution": SPIN_COLOR_DILUTION,
        "gauge_preprocessing": "HYP(1,0.75,0.6,0.3,dir_ignore=-1)",
        "multigrid": parse_optional_multigrid_blocks(args.mg_block),
        "flavor_convention": "single_strange_flavor_trace_for_this_dirac_operator",
    }
    n_inversions = effective_n_inversions(
        args.n_base_noise, NOISE_SCHEME, HP_NUM_VECTORS,
        spin_color_dilution=SPIN_COLOR_DILUTION,
    )
    for key, value in {
        "config": args.config_num,
        "gauge": gauge_path,
        "mass": MASS,
        "mpi_geometry": mpi_geometry,
        "multigrid": parameters["multigrid"],
        "q_count": len(qext),
        "hp_ordering": NESTED_HP_ORDERING,
        "hp_num_vectors": HP_NUM_VECTORS,
        "spin_color_dilution": SPIN_COLOR_DILUTION,
        "n_base_noise": args.n_base_noise,
        "effective_n_inversions": n_inversions,
        "flow_batch_size": args.flow_batch_size,
        "resource_path": resource_path,
        "shard_dir": shard_dir,
        "sample_log": sample_log_file,
    }.items():
        mpi_print(latt_info, f"{key}={value}")

    TimedStrangeBenchmark(parameters).flowed_fermionic_1pt(
        gauge,
        [MASS, CSW, TOL, MAXITER],
        [args.n_base_noise, N_ZN, RAND_SEED],
        tag=canonical_tag,
        shard_dir=shard_dir,
        sample_log_file=sample_log_file,
        base_start=0,
        base_stop=args.n_base_noise,
        block_interval_solves=BLOCK_INTERVAL_SOLVES,
        flow_batch_size=args.flow_batch_size,
        save_raw_per_vector=False,
    )
    getMPIComm().Barrier()
    if latt_info.mpi_rank == 0:
        output = aggregate_emt_quark_1pt_bases(
            shard_dir, canonical_tag, args.n_base_noise
        )
        print(f"Finalized strange EMT independent bases: {output}", flush=True)


if __name__ == "__main__":
    main()
