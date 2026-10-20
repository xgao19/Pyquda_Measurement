"""Hadron-independent flowed clover topological-density loops.

The output is finite-flow, spatially averaged and time resolved, not an
already extrapolated RGI form factor. See docs/topological_charge/README.md.
"""

import operator
import numpy as np
from opt_einsum import contract
from pyquda import getGridSize
from pyquda_utils import phase

from .flowed_gauge_utils import all_clover_traceless, advance_gauge_flow
from .io_corr import save_topological_charge_1pt_hdf5
from .tools import array_to_numpy, _reduce_spatial_projected


def topological_density(fields):
    """Compute a^4 q using epsilon_xyzt=+1 and Hermitian clover fields."""
    result = contract("...ab,...ba->...", fields[0][1], fields[2][3])
    result -= contract("...ab,...ba->...", fields[0][2], fields[1][3])
    result += contract("...ab,...ba->...", fields[0][3], fields[1][2])
    return result.real / (4.0 * np.pi**2)


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


class TopologicalCharge1pt:
    """Measure all requested flow times, including zero, on a gauge copy.

    ``flow_epsilon`` is the output interval in lattice units. Every interval
    uses ``flow_substeps_per_interval`` steps, including the first interval.
    ``qext`` contains integer (nx, ny, nz, 0) modes and exactly one zero mode.
    """

    def __init__(self, parameters):
        self.flow_type = str(parameters["flow_type"]).strip().lower()
        if self.flow_type not in ("wilson", "symanzik"):
            raise ValueError("flow_type must be wilson or symanzik")
        self.flow_epsilon = float(parameters["flow_epsilon"])
        if not np.isfinite(self.flow_epsilon) or self.flow_epsilon <= 0:
            raise ValueError("flow_epsilon must be finite and positive")
        self.flow_steps = _integer(parameters["flow_steps"], "flow_steps", 0)
        self.substeps = _integer(parameters["flow_substeps_per_interval"], "flow_substeps_per_interval", 1)
        q = np.asarray(parameters["qext"])
        if (q.ndim != 2 or q.shape[1] != 4 or not np.issubdtype(q.dtype, np.integer)
                or np.any(q[:, 3] != 0) or len(np.unique(q, axis=0)) != len(q)):
            raise ValueError("qext must contain unique integer (nx, ny, nz, 0) modes")
        zero = np.flatnonzero(np.all(q == 0, axis=1))
        if len(zero) != 1:
            raise ValueError("qext must include exactly one zero momentum")
        self.qlist = q.astype(np.int32)
        self.zero_index = int(zero[0])
        self.identity = {key: parameters[key] for key in (
            "ensemble", "stream", "config_num", "setup", "gauge_path",
            "gauge_preprocessing", "code_version",
        )}

    def flowed_1pt(self, gauge, tag=None):
        """Return the result dictionary on rank zero and None elsewhere.

        A nonempty tag writes ``tag.h5`` on root only. Input gauge links are
        unchanged; QUDA resident state is not guaranteed to remain unchanged.
        """
        info = gauge.latt_info
        volume = int(np.prod(info.global_size[:3]))
        root = info.mpi_rank == 0
        times = np.arange(self.flow_steps + 1, dtype=np.float64) * self.flow_epsilon
        loops = np.empty((len(self.qlist), len(times), info.global_size[3]),
                         dtype=np.complex128) if root else None
        flowed = gauge.copy()
        flowed.toDevice()
        phases = phase.MomentumPhase(info).getPhases(self.qlist[:, :3].tolist(), [0, 0, 0, 0])
        for step in range(len(times)):
            fields = all_clover_traceless(flowed)
            density = topological_density(fields)
            projected = array_to_numpy(contract("qwtzyx,wtzyx->qt", phases, density))
            reduced = _reduce_spatial_projected(projected)
            if root:
                loops[:, step, :] = reduced / volume
            del fields, density, projected, reduced
            if step < self.flow_steps:
                advance_gauge_flow(flowed, self.flow_type, self.substeps,
                                   self.flow_epsilon / self.substeps)
        if not root:
            return None
        attrs = {
            **self.identity,
            "schema": "topological_charge_loop_v1",
            "axis_order": "momentum,flow_time,absolute_time",
            "global_lattice_size": np.asarray(info.global_size, dtype=np.int32),
            "mpi_grid": np.asarray(getGridSize(), dtype=np.int32),
            "flow_type": self.flow_type, "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps, "flow_substeps_per_interval": self.substeps,
            "integration_epsilon": self.flow_epsilon / self.substeps,
            "momentum_phase_origin": np.zeros(4, dtype=np.int32),
            "spatial_momentum_phase_convention": "exp(+2pi*i*sum_j q_j*(x_j-origin_j)/L_j)",
            "loop_time_convention": "absolute_lattice_time",
            "volume_norm": volume,
            "operator_normalization": "ReTr(H01H23-H02H13+H03H12)/(4*pi^2)",
            "field_strength_convention": "H=-i*(C-Cdag)_traceless/8; xyzt; epsilon0123=+1",
            "density_units": "a^4*q_physical",
            "renormalization_stage": "finite_flow; continuum_then_zero_flow_limit_required",
            "renormalization_applied": False,
        }
        result = {"loop": loops, "momentum_list": self.qlist.copy(),
                  "flow_times_lattice": times,
                  "Q_total": volume * loops[self.zero_index].real.sum(axis=-1),
                  "attrs": attrs}
        if tag:
            save_topological_charge_1pt_hdf5(tag, result)
        return result
