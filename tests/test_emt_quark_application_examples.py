"""CPU-only wiring checks for the quark-loop application entry points."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "application/EMT_disconnected_1pt"


def _run_driver(platform, tmp_path, env=None, rank=0):
    calls = {}

    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        return result

    info = types.SimpleNamespace(mpi_rank=rank)
    gauge = types.SimpleNamespace(
        latt_info=info, hypSmear=lambda *args: calls.update(hyp=args)
    )

    def read_gauge(path, **kwargs):
        calls["gauge"] = path
        return gauge

    class Measurement:
        def __init__(self, parameters):
            calls["parameters"] = parameters

        def flowed_fermionic_1pt(self, *args, **kwargs):
            calls["measurement"] = (args, kwargs)

    def aggregate(*args):
        assert calls["barrier"]
        calls["aggregate"] = args
        return "mock.bases.h5"

    comm = types.SimpleNamespace(Barrier=lambda: calls.update(barrier=True))
    package = "pyquda_measurement_utils"
    modules = {
        "pyquda": module(
            "pyquda", init=lambda *a, **kw: calls.update(init=(a, kw)),
            getMPIComm=lambda: comm,
        ),
        "pyquda_utils": module(
            "pyquda_utils", io=types.SimpleNamespace(readNERSCGauge=read_gauge)
        ),
        package + ".Disconnected_1pt_EMT_vibe_develop": module(
            package + ".Disconnected_1pt_EMT_vibe_develop",
            EMTDisconnectedQuark1pt=Measurement,
            parse_multigrid_blocks=lambda value: value,
            aggregate_emt_quark_1pt_bases=aggregate,
        ),
        package + ".Disconnected_utils_vibe_develop": module(
            package + ".Disconnected_utils_vibe_develop",
            NESTED_HP_ORDERING="nested_xyzt_checkerboard_v1",
            disconnected_sample_log_path=lambda data, tag: str(data) + "/sample_log",
            effective_n_inversions=lambda n, scheme, hp, **kw: n * hp * 12,
        ),
        package + ".flowed_fermion_bilinear_vibe_develop": module(
            package + ".flowed_fermion_bilinear_vibe_develop",
            parse_optional_multigrid_blocks=lambda value: value,
        ),
        package + ".io_corr": module(
            package + ".io_corr",
            get_emt_quark_loop_file_tag=lambda data, lat, cfg, src, tag: str(data) + "/" + tag,
        ),
        package + ".tools": module(
            package + ".tools", mpi_print=lambda *a, **kw: None,
            mpi_timer_print=lambda *a, **kw: None,
        ),
    }
    script = APP / platform / "Pyquda_EMT_disconnected_quark_1pt.py"
    argv = [str(script), "--config_num", "1050"]
    if platform == "Aurora":
        argv += ["--data-dir", str(tmp_path / "data"),
                 "--resource-path", str(tmp_path / "cache")]
    environment = {"EMT_1PT_DATA_DIR": str(tmp_path / "data"), **(env or {})}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", argv):
        runpy.run_path(str(script), run_name="__main__")
    calls["boundary"] = info.t_boundary
    return calls


def test_perlmutter_default_and_hp512_spin_color_wiring(tmp_path):
    defaults = _run_driver("perlmutter", tmp_path)
    assert defaults["parameters"]["spin_color_dilution"] == "none"
    assert defaults["measurement"][0][2] == [1, 4, 0]
    assert defaults["measurement"][1]["save_raw_per_vector"] is False
    assert "aggregate" not in defaults

    calls = _run_driver("perlmutter", tmp_path, {
        "EMT_1PT_NOISE_SCHEME": "hierarchical_probing",
        "EMT_1PT_HP_NUM_VECTORS": "512",
        "EMT_1PT_SPIN_COLOR_DILUTION": "point",
        "EMT_1PT_BLOCK_INTERVAL_SOLVES": "768",
        "EMT_1PT_N_VEC": "2",
        "EMT_1PT_BASE_START": "1",
        "EMT_1PT_BASE_STOP": "2",
    })
    assert calls["parameters"]["spin_color_dilution"] == "point"
    assert calls["parameters"]["hp_num_vectors"] == 512
    assert "hp_ordering" not in calls["parameters"]
    args, kwargs = calls["measurement"]
    assert args[2] == [2, 4, 0]
    assert (kwargs["base_start"], kwargs["base_stop"]) == (1, 2)
    assert kwargs["block_interval_solves"] == 768
    assert kwargs["flow_batch_size"] == 1


def test_aurora_benchmark_settings_and_rank_zero_finalization(tmp_path):
    calls = _run_driver("Aurora", tmp_path)
    init_args, init_kwargs = calls["init"]
    assert init_args == ([2, 2, 4, 4],)
    assert init_kwargs["backend"] == "dpnp"
    assert init_kwargs["backend_target"] == "sycl"
    assert calls["boundary"] == -1
    params = calls["parameters"]
    assert params["noise_scheme"] == "hierarchical_probing"
    assert params["hp_num_vectors"] == 512
    assert params["spin_color_dilution"] == "point"
    assert params["hp_ordering"] == "nested_xyzt_checkerboard_v1"
    assert len(params["qext"]) == 125
    assert params["flow_epsilon"] == 0.09
    assert params["flow_steps"] == 10
    assert params["multigrid"] == "4.4.4.4;4.4.2.2"
    args, kwargs = calls["measurement"]
    assert args[1] == [-0.015, 1.0372, 1e-10, 5000]
    assert args[2] == [1, 4, 0]
    assert kwargs["block_interval_solves"] == 768
    assert kwargs["flow_batch_size"] == 24
    assert kwargs["save_raw_per_vector"] is False
    assert calls["aggregate"] == (kwargs["shard_dir"], kwargs["tag"], 1)
    nonroot = _run_driver("Aurora", tmp_path, rank=1)
    assert nonroot["barrier"]
    assert "aggregate" not in nonroot


def test_perlmutter_shell_defaults_and_checkout_selection(tmp_path):
    # Mock activation and Python, but execute the real shell argument/default logic.
    harness = '''
source() { printf 'ACTIVATION=%s\\n' "$1"; }
python3() { env; printf 'PYTHON_ARG=%s\\n' "$@"; }
export -f source python3
bash "$@"
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith("EMT_1PT_")}
    env.pop("MEASUREMENT_ROOT", None)
    env.update({
        "PYQUDA_ROOT": "/wrong/original-checkout",
        "EMT_1PT_DATA_DIR": str(tmp_path / "data"),
        "QUDA_RESOURCE_PATH": str(tmp_path / "quda"),
        "CUPY_CACHE_DIR": str(tmp_path / "cupy"),
    })
    for wrapper in ("run_quark_1pt.sh", "run_finalize_quark_1pt.sh"):
        result = subprocess.run(
            ["bash", "-c", harness, "_", str(APP / "perlmutter" / wrapper),
             "--config_num", "1050"],
            env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert f"PYQUDA_ROOT={ROOT}\n" in result.stdout
        assert f"ACTIVATION={ROOT}/systems/perlmutter/activate-venv-quda.sh" in result.stdout
        if wrapper == "run_quark_1pt.sh":
            assert "EMT_1PT_N_VEC=1\n" in result.stdout
            assert "EMT_1PT_BASE_STOP=1\n" in result.stdout
            assert "EMT_1PT_SPIN_COLOR_DILUTION=none\n" in result.stdout
            assert f"EMT_1PT_GAUGE_PATH={ROOT}/test_gauge/" in result.stdout


def test_aurora_launcher_rejects_missing_allocation_before_writes(tmp_path):
    env = dict(os.environ)
    env.pop("PBS_NODEFILE", None)
    result = subprocess.run(
        ["bash", str(APP / "Aurora/run_quark_1pt.sh")],
        env=env, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "8-node PBS allocation" in result.stderr
    source = (APP / "Aurora/run_quark_1pt.sh").read_text()
    assert "flow_batch_size=24\n" in source
    assert '--flow-batch-size "$flow_batch_size"' in source
    assert 'export PYQUDA_ROOT="$measurement_root"' in source
    assert '"$benchmark_dir/tile_flat_affinity.sh"' in source
    assert "/run/l64c64a076/" not in source
