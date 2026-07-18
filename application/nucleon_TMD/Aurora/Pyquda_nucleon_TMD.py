#!/usr/bin/env python3
"""Aurora entry point for the shared connected-proton qTMD runner."""

from pathlib import Path
import sys


application_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(application_dir))

from shared_runner import PlatformDefaults, run


run(
    PlatformDefaults(
        name="aurora",
        mpi_geometry="1.5.4.5",
        gauge_path=(
            "/lus/flare/projects/StructNGB/xgao/ensembles/"
            "s8080b7596/gauge_fixed/{stream}/"
            "l8080f21b7596m00101m0202{stream}."
            "coulomb.1e-14.{conf}"
        ),
        data_dir=(
            "/lus/flare/projects/StructNGB/xgao/run/l80c80a050/"
            "nucleon_TMD_pyquda/data_{stream}"
        ),
        lat_tag="l80c80a050",
        mass=-0.0386,
        csw=1.03094,
        tol=1e-10,
        maxiter=5000,
        width=13.0,
        num_src=2,
        qmax=2,
        b_z=24,
        b_T=24,
        eta=12,
        t_separations=(9,),
        stream="c",
        source_shift=(7, 11, 13, 23),
        init_kwargs={
            "backend": "dpnp",
            "backend_target": "sycl",
            "resource_path": ".cache",
        },
    )
)
