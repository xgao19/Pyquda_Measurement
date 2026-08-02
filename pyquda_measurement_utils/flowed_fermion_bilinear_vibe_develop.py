"""Shared flowed-fermion bilinear primitives for EMT measurements.

This module intentionally contains no stochastic-noise, shard, resume, or
HDF5 workflow code.  It is the small common base used by connected pion and
proton EMT and by disconnected quark EMT.
"""

import numpy as np
from opt_einsum import contract

from pyquda.field import LatticeGauge, LatticePropagator, MultiLatticeFermion
from pyquda_utils import convert

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    VECTOR_GAMMA_POSITIONS,
    gamma_stack,
)
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


_VALID_FLOW_TYPES = {"wilson", "symanzik"}
EMT_OPERATOR_SCHEMA_VERSION = 5
my_gammas = list(GAMMA_LABELS)


def _gamma_matrix(gamma_like):
    return gamma_like.matrix if hasattr(gamma_like, "matrix") else gamma_like


def _array_on_backend(value, reference_array):
    xp = _get_xp_from_array(reference_array)
    if type(value).__module__.split(".")[0] == xp.__name__:
        return value
    if hasattr(value, "get"):
        value = value.get()
    return _asarray_on_queue(value, xp, reference_array)


def normalize_flow_type(flow_type):
    flow = str(flow_type).strip().lower()
    if flow not in _VALID_FLOW_TYPES:
        raise ValueError(
            f"flow_type should be one of {_VALID_FLOW_TYPES}, got {flow_type!r}"
        )
    return flow


def flow_times(flow_epsilon, flow_steps):
    return np.arange(flow_steps + 1, dtype=np.float64) * float(flow_epsilon)


def parse_multigrid_blocks(value):
    """Parse dot-separated QUDA blocks and semicolon-separated MG levels."""
    if isinstance(value, str):
        level_text = [item.strip() for item in value.split(";") if item.strip()]
        if not level_text:
            raise ValueError("multigrid block specification is empty")
        blocks = []
        for item in level_text:
            try:
                block = [int(entry) for entry in item.split(".")]
            except ValueError as error:
                raise ValueError(
                    f"invalid multigrid block {item!r}; expected X.Y.Z.T"
                ) from error
            blocks.append(block)
    else:
        blocks = [[int(entry) for entry in block] for block in value]
    if not blocks or any(len(block) != 4 for block in blocks):
        raise ValueError("each multigrid level must contain four integers")
    if any(entry <= 0 for block in blocks for entry in block):
        raise ValueError("multigrid block entries must be positive")
    return blocks


def parse_optional_multigrid_blocks(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return parse_multigrid_blocks(value)


class FlowedFermionBilinearKernel:
    """Lightweight Gamma, derivative, and two-propagator flow primitives."""

    def __init__(self, flow_type):
        self.flow_type = normalize_flow_type(flow_type)
        self._emt_gamma_cache = {}

    @staticmethod
    def _gamma_cache_key(ref_arr):
        xp = _get_xp_from_array(ref_arr)
        queue = getattr(ref_arr, "sycl_queue", None)
        if queue is not None:
            location = ("sycl_queue", id(queue))
        else:
            device = getattr(ref_arr, "device", None)
            device_id = getattr(device, "id", device)
            location = (
                ("device", str(device_id))
                if device is not None
                else ("host", None)
            )
        return (xp.__name__, str(getattr(ref_arr, "dtype", None)), location)

    def _gamma_cache_entry(self, ref_arr):
        key = self._gamma_cache_key(ref_arr)
        entry = self._emt_gamma_cache.get(key)
        if entry is None:
            entry = {"stack": gamma_stack(ref_arr), "matrices": {}}
            self._emt_gamma_cache[key] = entry
        return entry

    def _gamma5_for(self, ref_arr):
        return self._gamma_cache_entry(ref_arr)["stack"][GAMMA_LABELS.index("5")]

    def _gamma_stack_for(self, ref_arr):
        return self._gamma_cache_entry(ref_arr)["stack"]

    def _vector_gamma_stack_for(self, ref_arr):
        entry = self._gamma_cache_entry(ref_arr)
        if "vector_stack" not in entry:
            xp = _get_xp_from_array(ref_arr)
            entry["vector_stack"] = xp.stack(
                [entry["stack"][position] for position in VECTOR_GAMMA_POSITIONS]
            )
        return entry["vector_stack"]

    def _cached_backend_matrix(self, name, matrix, ref_arr):
        entry = self._gamma_cache_entry(ref_arr)
        if name not in entry["matrices"]:
            entry["matrices"][name] = _array_on_backend(
                _gamma_matrix(matrix), ref_arr
            )
        return entry["matrices"][name]

    def _get_interpolator_gamma_for(self, interpolator, ref_arr):
        if interpolator not in my_gammas:
            raise ValueError(
                f"Unsupported interpolator {interpolator!r}. "
                f"Expected one of {my_gammas}."
            )
        return self._gamma_stack_for(ref_arr)[my_gammas.index(interpolator)]

    @staticmethod
    def _covdev_sym_prop(gauge_dirac, prop: LatticePropagator, mu: int):
        mf = convert.propagatorToMultiFermion(prop)
        mf_covdev = convert.propagatorToMultiFermion(prop)
        for spin in range(4):
            for color in range(3):
                idx = spin * 3 + color
                forward = gauge_dirac.covDev(mf[idx], mu)
                backward = gauge_dirac.covDev(mf[idx], mu + 4)
                mf_covdev[idx] = 0.5 * (forward - backward)
        return convert.multiFermionToPropagator(mf_covdev)

    def _make_dst2(self, prop: LatticePropagator):
        gamma5 = self._gamma5_for(prop.data)
        return contract(
            "ab,wtzyxbcij,cd->wtzyxadij",
            gamma5,
            prop.data.conj().transpose(0, 1, 2, 3, 4, 6, 5, 8, 7),
            gamma5,
        )

    def _left_covdev_dst2_from_prop(
        self, gauge_dirac, prop: LatticePropagator, mu: int
    ):
        derivative = self._covdev_sym_prop(gauge_dirac, prop, mu)
        derivative_dagger = derivative.data.conj().transpose(
            0, 1, 2, 3, 4, 6, 5, 8, 7
        )
        gamma5 = self._gamma5_for(prop.data)
        return contract(
            "ab,wtzyxbcij,cd->wtzyxadij",
            gamma5,
            derivative_dagger,
            gamma5,
        )

    @staticmethod
    def _flow_two_props_pyquda(
        gauge: LatticeGauge,
        prop_a: LatticePropagator,
        prop_b: LatticePropagator,
        stepsize: float,
        nsteps: int,
        flow_type: str = "wilson",
    ):
        mf_a = convert.propagatorToMultiFermion(prop_a)
        mf_b = convert.propagatorToMultiFermion(prop_b)
        if mf_a.L5 != mf_b.L5:
            raise ValueError("The two propagators must have matching L5")
        l5 = mf_a.L5
        packed = convert.multiField(
            [mf_a[idx] for idx in range(l5)]
            + [mf_b[idx] for idx in range(l5)]
        )
        del mf_a, mf_b
        packed_flow = gauge.gradientFlow(packed, flow_type, nsteps, stepsize)
        del packed
        mf_a_flow = MultiLatticeFermion(gauge.latt_info, l5, packed_flow.data[:l5])
        mf_b_flow = MultiLatticeFermion(
            gauge.latt_info, l5, packed_flow.data[l5 : 2 * l5]
        )
        prop_a_flow = convert.multiFermionToPropagator(mf_a_flow)
        prop_b_flow = convert.multiFermionToPropagator(mf_b_flow)
        prop_a_flow._packed_flow_owner = packed_flow
        prop_b_flow._packed_flow_owner = packed_flow
        return prop_a_flow, prop_b_flow

    def _advance_flowed_props(
        self, gauge, prop_a, prop_b, step, stepsize, nsteps, substeps_per_interval=1
    ):
        if nsteps > 0 and step == 0:
            return self._flow_two_props_pyquda(
                gauge,
                prop_a,
                prop_b,
                stepsize / 10,
                nsteps=10,
                flow_type=self.flow_type,
            )
        if nsteps > 0 and step < nsteps:
            return self._flow_two_props_pyquda(
                gauge,
                prop_a,
                prop_b,
                stepsize / substeps_per_interval,
                nsteps=substeps_per_interval,
                flow_type=self.flow_type,
            )
        return prop_a, prop_b


__all__ = [
    "EMT_OPERATOR_SCHEMA_VERSION",
    "FlowedFermionBilinearKernel",
    "flow_times",
    "my_gammas",
    "normalize_flow_type",
    "parse_multigrid_blocks",
    "parse_optional_multigrid_blocks",
]
