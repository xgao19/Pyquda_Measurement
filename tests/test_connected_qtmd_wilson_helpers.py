import pytest

from pyquda_measurement_utils.qtmd_operator_utils import (
    apply_gi_qtmd_staple_to_propagator,
    create_cg_qtmd_wilsonline_index_lists,
    shift_propagator_pdf_gi,
    shift_qtmd_cg,
)


class FakeFermion:
    def __init__(self, path=None):
        self.path = list(path or [])

    def copy(self):
        return FakeFermion(self.path)


class FakePureGauge:
    def covDev(self, fermion, direction):
        return FakeFermion(fermion.path + [direction])


class FakeGauge:
    pure_gauge = FakePureGauge()


class FakePropagator:
    def __init__(self):
        self.values = {(spin, color): FakeFermion() for spin in range(4) for color in range(3)}

    def copy(self):
        copied = FakePropagator()
        copied.values = {key: val.copy() for key, val in self.values.items()}
        return copied

    def getFermion(self, spin, color):
        return self.values[(spin, color)]

    def setFermion(self, fermion, spin, color):
        self.values[(spin, color)] = fermion


class FakeShiftField:
    def __init__(self, shifts=None):
        self.shifts = list(shifts or [])

    def shift(self, steps, direction):
        return FakeShiftField(self.shifts + [(steps, direction)])


def _all_paths(prop):
    return {tuple(value.path) for value in prop.values.values()}


def test_connected_pdf_gi_one_step_helper_supports_both_directions():
    gauge = FakeGauge()
    positive = shift_propagator_pdf_gi(
        gauge, FakePropagator(), [0, 1, 0, 0], [0, 0, 0, 0]
    )
    negative = shift_propagator_pdf_gi(
        gauge, FakePropagator(), [0, -1, 0, 0], [0, 0, 0, 0]
    )
    assert _all_paths(positive) == {(2,)}
    assert _all_paths(negative) == {(6,)}


def test_connected_pdf_gi_helpers_reject_non_incremental_jumps():
    gauge = FakeGauge()
    with pytest.raises(ValueError):
        shift_propagator_pdf_gi(
            gauge,
            FakePropagator(),
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        )


def test_connected_gi_qtmd_production_requires_link_cache():
    w_index = [2, 4, 3, 0]
    with pytest.raises(TypeError):
        apply_gi_qtmd_staple_to_propagator(
            FakePropagator(), w_index
        )


def test_cg_qtmd_incremental_shift_order_is_unchanged():
    shifted = shift_qtmd_cg(
        FakeShiftField(),
        [3, -2, 0, 1],
        [1, 1, 0, 1],
    )
    assert shifted.shifts == [(2, 1), (-3, 2)]


def test_cg_qtmd_greedy_execution_order_is_unchanged():
    direction0, direction1 = create_cg_qtmd_wilsonline_index_lists(2, 2)
    expected = [
        [0, -2, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, -1, 0, 0],
        [1, -2, 0, 0],
        [2, -2, 0, 0],
        [2, -1, 0, 0],
        [2, 0, 0, 0],
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 2, 0, 0],
    ]
    assert direction0 == expected
    assert direction1 == [
        [b_T, b_z, eta, 1] for b_T, b_z, eta, _direction in expected
    ]
import pytest
