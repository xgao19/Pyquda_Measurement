"""Aurora example: deterministic, source-independent topological charge loop."""

from pathlib import Path
from pyquda import init
from pyquda_utils import io
from pyquda_measurement_utils.topological_charge_1pt import TopologicalCharge1pt
from pyquda_measurement_utils.io_corr import get_topological_charge_loop_file_tag


# Edit all production choices here; no inversion or source-count parameters.
ROOT = Path("/lus/flare/projects/StructNGB/xgao")
ENSEMBLE = "l80c80a050"
STREAM = "b"
CONFIGURATION = 600
GRID = [1, 5, 4, 5]
FLOW_TYPE = "wilson"
FLOW_OUTPUT_INTERVAL = 0.1
FLOW_INTERVALS = 4
FLOW_SUBSTEPS = 10
QMAX = 1
GAUGE_PATH = ROOT / "ensembles/s8080b7596" / f"l8080f21b7596m00101m0202{STREAM}.coulomb.1e-14" / f"l8080f21b7596m00101m0202{STREAM}.coulomb.1e-14.{CONFIGURATION}"
OUTPUT_DIR = ROOT / "run/l80c80a050/develop/topological_charge_1pt/example/data"
CACHE_DIR = ROOT / "run/l80c80a050/develop/topological_charge_1pt/example/cache"
CODE_VERSION = "record-exact-commit-and-working-tree-before-production"
SETUP = f"clover.noHYP.{FLOW_TYPE}.dt{FLOW_OUTPUT_INTERVAL:g}.n{FLOW_INTERVALS}.sub{FLOW_SUBSTEPS}.qmax{QMAX}"


def main():
    init(GRID, enable_mps=True, backend="dpnp", backend_target="sycl", resource_path=str(CACHE_DIR))
    gauge = io.readNERSCGauge(str(GAUGE_PATH), checksum=False, link_trace=False, plaquette=False)
    parameters = dict(
        ensemble=ENSEMBLE, stream=STREAM, config_num=CONFIGURATION, setup=SETUP,
        gauge_path=str(GAUGE_PATH), gauge_preprocessing="Coulomb gauge fixed input; no extra HYP",
        code_version=CODE_VERSION, flow_type=FLOW_TYPE, flow_epsilon=FLOW_OUTPUT_INTERVAL,
        flow_steps=FLOW_INTERVALS, flow_substeps_per_interval=FLOW_SUBSTEPS,
        qext=[[x, y, z, 0] for x in range(-QMAX, QMAX + 1)
              for y in range(-QMAX, QMAX + 1) for z in range(-QMAX, QMAX + 1)],
    )
    tag = get_topological_charge_loop_file_tag(OUTPUT_DIR, ENSEMBLE, STREAM, CONFIGURATION, SETUP)
    result = TopologicalCharge1pt(parameters).flowed_1pt(gauge, tag)
    if result is not None:
        print("OUTPUT", tag + ".h5", "Q_total", result["Q_total"], flush=True)


if __name__ == "__main__":
    main()
