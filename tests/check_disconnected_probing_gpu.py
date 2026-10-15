"""Explicit HIP/SYCL MPI smoke entry; GPU kernels require allocated devices."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi-geometry", default="1.1.1.1")
    parser.add_argument("--backend-target", choices=("hip", "sycl"), default="hip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheme", choices=("zn", "hierarchical_probing", "multiplier_probing"), default="hierarchical_probing")
    parser.add_argument("--hp-count", dest="probe_count", type=int, choices=(2, 32, 512), default=2)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-emt", action="store_true", help="also run one complete base, one flow interval and EMT aggregation")
    args = parser.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    geometry = [int(s) for s in args.mpi_geometry.split(".")]
    if len(geometry) != 4 or any(n <= 0 or 8 % n for n in geometry) or np.prod(geometry) != comm.size:
        raise ValueError("geometry must divide 8^4 and match MPI world size")
    error = None
    if comm.rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "quda-cache").mkdir()
        except OSError as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)
    os.environ["QUDA_RESOURCE_PATH"] = str(args.output / "quda-cache")
    from pyquda import init
    backend = "dpnp" if args.backend_target == "sycl" else "cupy"
    init(geometry, backend=backend, backend_target=args.backend_target, enable_mps=False)
    if args.backend_target == "hip":
        import cupy as cp
        if not cp.cuda.runtime.is_hip:
            raise RuntimeError("the HIP target requires CuPy HIP")
    from pyquda_measurement_utils.tools import array_to_numpy
    from pyquda.field import LatticeInfo
    from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
        NESTED_HP_ORDERING, apply_hierarchical_probe, apply_multiplier_probe,
        make_counter_zn_noise_fermion, broadcast_multiplier_plan,
        vectors_per_base, iter_noise_base_probe_interval,
    )
    info = LatticeInfo([8]*4)
    m1 = dict(version=1, shape=[8]*4, distance=1, colors=2, sigma=[1]*4)
    xi = make_counter_zn_noise_fermion(info, 11, 0)
    for k in range(2):
        np.testing.assert_allclose(
            array_to_numpy(apply_hierarchical_probe(xi, k, NESTED_HP_ORDERING).data),
            array_to_numpy(apply_multiplier_probe(xi, k, m1).data), rtol=0, atol=1e-14)
    plan = broadcast_multiplier_plan(args.plan, comm) if args.scheme == "multiplier_probing" else None
    if args.plan is not None and plan is None:
        raise ValueError("--plan requires --scheme multiplier_probing")
    count = vectors_per_base(args.scheme, args.probe_count, plan)
    coords = info.coordinate()
    ids = np.asarray(coords[0] + 8*(coords[1] + 8*(coords[2] + 8*coords[3]))).ravel()
    digest = hashlib.sha256()
    for _, _, _, source, _, _ in iter_noise_base_probe_interval(info, 0, 0, count, 4, args.scheme,
            args.probe_count, NESTED_HP_ORDERING, 11, 0, plan):
        pieces = comm.gather((ids, array_to_numpy(source.data).reshape(-1, 4, 3)), root=0)
        if comm.rank == 0:
            site_ids = np.concatenate([piece[0] for piece in pieces])
            np.testing.assert_array_equal(np.sort(site_ids), np.arange(8**4))
            values = np.concatenate([piece[1] for piece in pieces])[np.argsort(site_ids)]
            digest.update(values.astype("<c16").tobytes())
    result = dict(scheme=args.scheme, vectors_per_base=count, geometry=geometry,
                  global_source_sha256=digest.hexdigest(), hip=args.backend_target == "hip",
                  backend=backend, backend_target=args.backend_target, multiplier_plan=plan)
    if args.run_emt:
        from pyquda_utils import io
        from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
            EMTDisconnectedQuark1pt, aggregate_emt_quark_1pt_bases,
        )
        repo = Path(__file__).resolve().parents[1]
        gauge = io.readNERSCGauge(str(repo / "test_gauge/S8T8_wilson_b6.0"))
        gauge.latt_info.t_boundary = -1
        options = dict(config_num=11, qext=[[0, 0, 0, 0]], flow_type="wilson", flow_epsilon=0.01,
                       flow_steps=1, noise_scheme=args.scheme, hp_num_vectors=args.probe_count,
                       multigrid=None, gauge_preprocessing="none")
        if plan is not None:
            options["multiplier_plan"] = plan
        measurement = EMTDisconnectedQuark1pt(options)
        tag = str(args.output / "EMTquarkLoop")
        measurement.flowed_fermionic_1pt(gauge, [0.236, 1.0372, 1e-10, 1000], [1, 4, 0], tag,
            shard_dir=args.output / "shards", sample_log_file=args.output / "sample.log",
            block_interval_solves=6, flow_batch_size=2)
        if comm.rank == 0:
            result["base_statistics"] = aggregate_emt_quark_1pt_bases(args.output / "shards", tag, 1)
    if comm.rank == 0:
        with (args.output / "smoke.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
