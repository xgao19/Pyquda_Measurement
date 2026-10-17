"""Run algebra/bookkeeping tests with explicit CPU-only PyQUDA import stubs.

This runner never validates GPU kernels or MPI. Unmocked GPU calls fail loudly.
The gamma matrices are loaded from the user's PyQUDA source, not reimplemented.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def install_stubs(pyquda_root):
    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU/MPI operation unavailable in CPU-only test runner")

    def module(name, **attrs):
        obj = types.ModuleType(name)
        obj.__dict__.update(attrs)
        sys.modules[name] = obj
        return obj

    class Fermion:
        def __init__(self, info):
            self.latt_info = info
            self.data = np.zeros(np.asarray(info.coordinate(0)).shape + (4, 3), complex)

        def copy(self):
            result = Fermion(self.latt_info)
            result.data[:] = self.data
            return result

    class DeviceOnly:
        def __init__(self, *args, **kwargs):
            unavailable()

    module("pyquda", getMPIComm=unavailable, getGridSize=unavailable,
           getGridCoord=unavailable, MPI=types.SimpleNamespace(IN_PLACE=object(), SUM=object()))
    module("pyquda.field", LatticeFermion=Fermion, LatticeGauge=DeviceOnly,
           LatticeInfo=DeviceOnly, LatticeComplex=DeviceOnly,
           LatticeLink=DeviceOnly, LatticePropagator=DeviceOnly, MultiLatticeFermion=DeviceOnly)
    module("pyquda_comm", getArrayBackend=lambda: "numpy",
           getMPIRank=unavailable, getCoordFromRank=unavailable, getRankFromCoord=unavailable)
    module("pyquda_comm.array", arrayAsArray=lambda value, *args, **kwargs: np.asarray(value),
           arrayIdentity=unavailable, arrayZeros=unavailable)
    gamma_path = pyquda_root / "pyquda_core/pyquda/gamma.py"
    spec = importlib.util.spec_from_file_location("pyquda.gamma", gamma_path)
    gamma = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gamma
    spec.loader.exec_module(gamma)
    core = module("pyquda_utils.core", getDirac=unavailable, gatherLattice=unavailable)
    phase = module("pyquda_utils.phase", MomentumPhase=unavailable)
    convert = module("pyquda_utils.convert", multiField=unavailable,
                     fermionToLink=unavailable, linkToFermion=unavailable)
    gamma_api = module("pyquda_utils.gamma", gamma=gamma.DeGrandRossiMatrix.matrix)
    module("pyquda_utils.fft", fft=unavailable, ifft=unavailable)
    source = module("pyquda_utils.source", sequential12=unavailable)
    module("pyquda_utils", core=core, phase=phase, convert=convert, gamma=gamma_api, source=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--pyquda-root", type=Path, default=repo.parent / "PyQUDA-develop")
    args, pytest_args = parser.parse_known_args()
    sys.path.insert(0, str(repo))
    install_stubs(args.pyquda_root)
    import pytest

    print("CPU MOCK TESTS ONLY: no QUDA, GPU or MPI verification")
    return pytest.main(pytest_args or [
        str(repo / "tests/test_disconnected_probing_math.py"),
        str(repo / "tests/test_disconnected_probing_workflow.py"),
        str(repo / "tests/test_disconnected_noise_bookkeeping.py"),
        str(repo / "tests/test_disconnected_emt_flow_bookkeeping.py"),
        str(repo / "tests/test_emt_projected_allreduce.py"),
        str(repo / "tests/test_projected_reduce.py"),
        str(repo / "tests/test_pion_soft_factor_optimized.py"),
        str(repo / "tests/test_tmdwf_components9.py"),
        str(repo / "tests/test_disconnected_shard_utils.py"), "-q",
        str(repo / "tests/test_flowed_quark_ringed_norm.py"),
        str(repo / "tests/test_emt_disconnected_analysis.py"),
        str(repo / "tests/test_emt_ringed_stochastic_comparison.py"),
        str(repo / "tests/test_emt_quark_1pt_convergence.py"),
        str(repo / "tests/test_emt_proton_t44_analysis.py"),
        str(repo / "tests/test_fermion_bilinear_basis.py"),
        str(repo / "tests/test_disconnected_qtmd_trace_direction.py"),
        str(repo / "tests/test_application_parameter_bookkeeping.py"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
