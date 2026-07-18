"""Slow direct-covDev GI qTMD reference used only by tests."""

from pyquda_measurement_utils.qtmd_operator_utils import (
    gi_qtmd_staple_segments,
)


def apply_signed_covariant_shift(gauge, fermion, direction, steps):
    shifted = fermion
    covdev_direction = direction if steps >= 0 else direction + 4
    for _ in range(abs(int(steps))):
        shifted = gauge.pure_gauge.covDev(shifted, covdev_direction)
    return shifted


def create_fermion_TMD_GI(gauge, fermion, W_index):
    shifted = fermion.copy()
    for direction, steps in reversed(gi_qtmd_staple_segments(W_index)):
        shifted = apply_signed_covariant_shift(gauge, shifted, direction, steps)
    return shifted
