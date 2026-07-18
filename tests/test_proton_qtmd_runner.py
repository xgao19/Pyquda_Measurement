import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop import (
    parse_optional_multigrid_blocks,
)
from pyquda_measurement_utils.proton_qTMD_pyquda import proton_TMD
from pyquda_measurement_utils.tools import array_to_numpy


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "application/nucleon_TMD/shared_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "proton_qtmd_shared_runner_test", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _defaults(module):
    return module.PlatformDefaults(
        name="test",
        mpi_geometry="1.1.1.1",
        gauge_path="gauge.{conf}",
        data_dir="data",
        lat_tag="S8T8",
        mass=0.1,
        csw=1.0,
        tol=1e-10,
        maxiter=100,
        width=1.0,
        num_src=1,
        qmax=0,
        b_z=1,
        b_T=1,
        eta=1,
        t_separations=(2,),
    )


def test_proton_qtmd_configuration_and_unknown_arguments_are_strict():
    runner = _load_runner()
    parser = runner.build_parser(_defaults(runner))
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--config_num", "7", "--unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config_num", "7", "--t_" + "insert", "2"]
        )
    args = parser.parse_args(
        [
            "--config_num",
            "7",
            "--mg-block",
            "4.4.4.4;4.4.4.4",
            "--t_separations",
            "2",
        ]
    )
    assert args.config_num == 7
    assert args.t_separations == (2,)
    assert parse_optional_multigrid_blocks(args.mg_block) == [
        [4, 4, 4, 4],
        [4, 4, 4, 4],
    ]
    assert parse_optional_multigrid_blocks("none") is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config_num", "7", "--t_separations", "2,3"]
        )


def test_proton_qtmd_resume_precedes_source_and_is_hdf5_independent():
    source = RUNNER_PATH.read_text()
    assert source.index("read_sample_log_entries(sample_log_file)") < source.index(
        "for source_position in source_positions:"
    )
    assert source.index("if entry in completed:") < source.index(
        'source.propagator('
    )
    assert source.index("append_sample_log_entry(sample_log_file, entry)") > source.index(
        "_save_pdf("
    )
    assert "h5py" not in source
    assert "fingerprint" not in source
    sample_log_section = source[
        source.index("sample_log_file =") :
        source.index("for source_position in source_positions:")
    ]
    assert "mg_block" not in sample_log_section


def test_proton_qtmd_shell_requires_configuration_before_environment_setup():
    wrappers = [
        REPO_ROOT
        / "application/nucleon_TMD/perlmutter/run_nucleon_TMD.sh",
        REPO_ROOT / "application/nucleon_TMD/Aurora/sub_TMD.sh",
    ]
    for wrapper in wrappers:
        missing = subprocess.run(
            ["bash", str(wrapper)], capture_output=True, text=True
        )
        unknown = subprocess.run(
            ["bash", str(wrapper), "--config_num", "7", "--bad"],
            capture_output=True,
            text=True,
        )
        assert missing.returncode == 2
        assert unknown.returncode == 2
        assert "--config_num" in missing.stderr


def test_proton_qtmd_has_one_shared_runner_and_two_thin_platform_entries():
    perlmutter = (
        REPO_ROOT
        / "application/nucleon_TMD/perlmutter/Pyquda_nucleon_TMD.py"
    ).read_text()
    aurora = (
        REPO_ROOT
        / "application/nucleon_TMD/Aurora/Pyquda_nucleon_TMD.py"
    ).read_text()
    assert "PlatformDefaults" in perlmutter
    assert "PlatformDefaults" in aurora
    assert "core.invertPropagator" not in perlmutter
    assert "core.invertPropagator" not in aurora
    assert not (REPO_ROOT / "application/nucleon_TMD_CG").exists()
    assert not (
        REPO_ROOT
        / "application/nucleon_TMD/Aurora/pyquda_nucleon_TMD_GI.py"
    ).exists()


def test_proton_qtmd_measurement_keeps_only_c2_state():
    measurement = proton_TMD(
        {
            "eta": [1],
            "b_z": 2,
            "b_T": 1,
            "p_2pt": [[0, 0, 0, 0]],
            "width": 2.0,
            "boost_out": [0, 0, 0],
        }
    )
    for field in (
        "eta",
        "b_z",
        "b_T",
        "pf",
        "plist",
        "qlist",
        "pol_list",
        "t_separations",
    ):
        assert not hasattr(measurement, field)


def test_proton_emt_requires_explicit_nonempty_t_separations():
    source = (
        REPO_ROOT
        / "pyquda_measurement_utils/proton_EMT_vibe_develop.py"
    ).read_text()
    assert "self.config_num" not in source
    assert 'parameters["t_separations"]' in source
    assert "t_separations must contain at least one sink time" in source

    c2_wrapper = (
        REPO_ROOT
        / "application/EMT_disconnected_1pt/perlmutter/"
        "Pyquda_EMT_disconnected_proton_2pt.py"
    ).read_text()
    assert '"t_separations": t_separations' in c2_wrapper
    assert 'parser.add_argument("--t_separations"' in c2_wrapper


def test_shared_host_conversion_accepts_numpy():
    value = np.arange(6).reshape(2, 3)
    converted = array_to_numpy(value)
    assert np.array_equal(converted, value)
