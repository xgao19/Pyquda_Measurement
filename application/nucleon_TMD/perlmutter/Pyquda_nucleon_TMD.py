#!/usr/bin/env python3
"""Perlmutter entry point for the shared connected-proton qTMD runner."""

import os
from pathlib import Path
import sys


application_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(application_dir))

from shared_runner import PlatformDefaults, run


software_root = Path(
    os.environ.get(
        "SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"
    )
)

run(
    PlatformDefaults(
        name="perlmutter",
        mpi_geometry="1.1.1.1",
        gauge_path=str(
            software_root
            / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0"
        ),
        data_dir=str(Path(__file__).resolve().parent / "data"),
        lat_tag="S8T32",
        mass=0.236,
        csw=1.0372,
        tol=1e-15,
        maxiter=300,
        width=1.0,
        num_src=1,
        qmax=0,
        b_z=2,
        b_T=1,
        eta=1,
        t_separations=(2,),
    )
)
