import os
import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest


TEST_REQUIRES = "gpu"


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _run(cmd, env):
    subprocess.run(cmd, cwd=_repo_root(), env=env, check=True, timeout=900)


def test_optional_pion_soft_factor_tiny_gauge_smoke():
    if os.environ.get("PYQUDA_RUN_TINY_GAUGE_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_TINY_GAUGE_SMOKE=1 on a GPU node to run tiny-gauge smoke workflows")

    repo = _repo_root()
    data_dir = Path(os.environ.get("PYQUDA_TINY_GAUGE_SMOKE_DIR", "/tmp/pyquda_measurement_tiny_smoke"))
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    env_prop = os.environ.copy()
    env_prop.update(
        {
            "PION_SOFT_DATA_DIR": str(data_dir / "pion_soft_factor"),
            "PION_SOFT_GAUGE_PATH": str(repo / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
            "PION_SOFT_T_START": "0",
            "PION_SOFT_T_COUNT": "3",
            "PION_SOFT_QUARK_MOM_Z": "0",
            "PION_SOFT_BT_DIR": "0",
            "PION_SOFT_BT_LENGTH": "0",
            "PION_SOFT_BZ_LENGTH": "0",
            "PION_SOFT_TSEP_LIST": "2",
            "PION_SOFT_MAXITER": "300",
            "QUDA_ENABLE_TUNING": "0",
            "QUDA_RESOURCE_PATH": str(data_dir / ".quda-cache" / "pion_soft_factor"),
            "CUPY_CACHE_DIR": str(data_dir / ".cupy-cache" / "pion_soft_factor"),
        }
    )
    env_contract = env_prop.copy()
    env_contract["PION_SOFT_T_COUNT"] = "1"

    _run(["bash", "application/pion_soft_factor/perlmutter/run_pion_soft_factor_prop.sh"], env_prop)
    _run(["bash", "application/pion_soft_factor/perlmutter/run_pion_soft_factor_contract.sh"], env_contract)

    assert list((data_dir / "pion_soft_factor" / "pion_soft_factor").glob("*.h5"))
    assert list((data_dir / "pion_soft_factor" / "pion_soft_factor_c2pt").glob("*.h5"))
    assert list((data_dir / "pion_soft_factor" / "pion_soft_factor_qTMDWF").glob("*.h5"))


def test_optional_pion_emff_tiny_gauge_smoke():
    if os.environ.get("PYQUDA_RUN_TINY_GAUGE_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_TINY_GAUGE_SMOKE=1 on a GPU node to run tiny-gauge smoke workflows")
    if os.environ.get("PYQUDA_RUN_EMFF_TINY_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_EMFF_TINY_SMOKE=1 to include the heavier EMFF inversion smoke")

    repo = _repo_root()
    data_dir = Path(os.environ.get("PYQUDA_TINY_GAUGE_SMOKE_DIR", "/tmp/pyquda_measurement_tiny_smoke")) / "pion_emff"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PION_EMFF_DATA_DIR": str(data_dir),
            "PION_EMFF_GAUGE_PATH": str(repo / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
            "PION_EMFF_NUM_SRC": "1",
            "PION_EMFF_QMAX": "0",
            "PION_EMFF_PF": "0.0.0",
            "PION_EMFF_WIDTH": "1.0",
            "PION_EMFF_POS_BOOST_SRC": "0.0.0",
            "PION_EMFF_POS_BOOST_SINK": "0.0.0",
            "PION_EMFF_NEG_BOOST_SRC": "0.0.0",
            "PION_EMFF_NEG_BOOST_SINK": "0.0.0",
            "PION_EMFF_MAXITER": "300",
            "QUDA_ENABLE_TUNING": "0",
            "QUDA_RESOURCE_PATH": str(data_dir / ".quda-cache"),
            "CUPY_CACHE_DIR": str(data_dir / ".cupy-cache"),
        }
    )

    _run(["bash", "application/EMFF_pion/perlmutter/run_pion_EMFF.sh"], env)

    assert list((data_dir / "pion_EMFF").glob("*.h5"))
    assert list((data_dir / "c2pt").glob("*.h5"))


def test_optional_pion_emff_background_response_tiny_gauge_smoke():
    if os.environ.get("PYQUDA_RUN_TINY_GAUGE_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_TINY_GAUGE_SMOKE=1 on a GPU node to run tiny-gauge smoke workflows")
    if os.environ.get("PYQUDA_RUN_EMFF_BG_TINY_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_EMFF_BG_TINY_SMOKE=1 to include the EMFF background-response smoke")

    repo = _repo_root()
    data_dir = Path(os.environ.get("PYQUDA_TINY_GAUGE_SMOKE_DIR", "/tmp/pyquda_measurement_tiny_smoke")) / "pion_emff_background_response"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PION_EMFF_BG_DATA_DIR": str(data_dir),
            "PION_EMFF_BG_GAUGE_PATH": str(repo / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
            "PION_EMFF_BG_PF": "0.0.0",
            "PION_EMFF_BG_QEXT_LIST": "0.0.0;0.0.1",
            "PION_EMFF_BG_TSEP_LIST": "2.4",
            "PION_EMFF_BG_CURRENT_GAMMAS": "T",
            "PION_EMFF_BG_TAU_WINDOW": "restricted",
            "PION_EMFF_BG_TAU_MIN": "1",
            "PION_EMFF_BG_WIDTH": "1.0",
            "PION_EMFF_BG_MAXITER": "300",
            "QUDA_ENABLE_TUNING": "0",
            "QUDA_RESOURCE_PATH": str(data_dir / ".quda-cache"),
            "CUPY_CACHE_DIR": str(data_dir / ".cupy-cache"),
        }
    )

    _run(["bash", "application/EMFF_pion_background_response/perlmutter/run_pion_EMFF_background_response.sh"], env)

    assert list((data_dir / "background_response").glob("*.h5"))


def test_optional_pion_current_current_response_tiny_gauge_smoke():
    if os.environ.get("PYQUDA_RUN_TINY_GAUGE_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_TINY_GAUGE_SMOKE=1 on a GPU node to run tiny-gauge smoke workflows")
    if os.environ.get("PYQUDA_RUN_CURRENT_CURRENT_TINY_SMOKE") != "1":
        raise SkipTest("set PYQUDA_RUN_CURRENT_CURRENT_TINY_SMOKE=1 to include current-current response smoke")

    repo = _repo_root()
    data_dir = Path(os.environ.get("PYQUDA_TINY_GAUGE_SMOKE_DIR", "/tmp/pyquda_measurement_tiny_smoke")) / "pion_current_current_response"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PION_CC_RESPONSE_DATA_DIR": str(data_dir),
            "PION_CC_RESPONSE_GAUGE_PATH": str(repo / "test_gauge" / "S8T32_wilson_b6.cg.1e-08.0"),
            "PION_CC_RESPONSE_PF": "0.0.0",
            "PION_CC_RESPONSE_FIRST_QEXT": "0.0.1",
            "PION_CC_RESPONSE_SECOND_QEXT": "0.0.-1",
            "PION_CC_RESPONSE_TSEP": "2",
            "PION_CC_RESPONSE_FIRST_GAMMA": "T",
            "PION_CC_RESPONSE_SECOND_GAMMA": "T",
            "PION_CC_RESPONSE_FIRST_TAU_WINDOW": "restricted",
            "PION_CC_RESPONSE_SECOND_TAU_WINDOW": "restricted",
            "PION_CC_RESPONSE_FIRST_TAU_MIN": "1",
            "PION_CC_RESPONSE_SECOND_TAU_MIN": "1",
            "PION_CC_RESPONSE_WIDTH": "1.0",
            "PION_CC_RESPONSE_MAXITER": "300",
            "QUDA_ENABLE_TUNING": "0",
            "QUDA_RESOURCE_PATH": str(data_dir / ".quda-cache"),
            "CUPY_CACHE_DIR": str(data_dir / ".cupy-cache"),
        }
    )

    _run(["bash", "application/pion_current_current_response/perlmutter/run_pion_current_current_response.sh"], env)

    assert list((data_dir / "current_current_response").glob("*.h5"))
