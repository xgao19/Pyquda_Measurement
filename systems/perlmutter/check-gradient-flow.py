#!/usr/bin/env python3
"""Minimal PyQUDA gradient-flow smoke test for Perlmutter.

This script expects the Perlmutter QUDA / PyQUDA environment to already be
prepared, for example by sourcing `activate-venv-quda.sh`.
"""

from __future__ import annotations

import os

from pyquda_utils import core, io


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> int:
    gauge_path = os.environ["GAUGE_PATH"]
    grid_size = parse_csv_ints(os.environ.get("GRID_SIZE", "1,1,1,1"))
    latt_size = parse_csv_ints(os.environ.get("LATT_SIZE", "8,8,8,8"))
    flow_steps = int(os.environ.get("FLOW_STEPS", "1"))
    flow_epsilon = float(os.environ.get("FLOW_EPSILON", "0.01"))
    quda_path = os.environ["QUDA_PATH"]

    print(f"python={os.sys.version.split()[0]}")
    print(f"gauge_path={gauge_path}")
    print(f"grid_size={grid_size}")
    print(f"latt_size={latt_size}")

    core.init(
        grid_size=grid_size,
        latt_size=latt_size,
        resource_path=quda_path,
        enable_tuning=False,
    )

    gauge = io.readNERSCGauge(gauge_path, checksum=False, plaquette=False, link_trace=False)
    print(f"lattice={gauge.latt_info.global_size}")
    print(f"plaquette_before={gauge.plaquette()}")

    wilson = gauge.copy()
    wilson.gradientGaugeFlow("wilson", flow_steps, flow_epsilon, compute_plaquette=True)
    print(f"wilson_after={wilson.plaquette()}")

    symanzik = gauge.copy()
    symanzik.gradientGaugeFlow("symanzik", flow_steps, flow_epsilon, compute_plaquette=True)
    print(f"symanzik_after={symanzik.plaquette()}")

    fermion = core.MultiLatticeFermion(gauge.latt_info, 1)
    fermion.data[:] = 0
    fermion.data[0, 0, 0, 0, 0, 0, 0, 0] = 1 + 1j

    flowed_fermion = gauge.gradientFlow(fermion, "wilson", flow_steps, flow_epsilon, True)
    print(f"fermion_norm2={flowed_fermion.norm2()}")
    print(f"fermion_sample={flowed_fermion.data[0, 0, 0, 0, 0, 0, 0, 0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
