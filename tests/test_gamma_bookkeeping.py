import sys
from pathlib import Path

import numpy as np

from pyquda_utils import gamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS, PYQUDA_GAMMA_IDS
from pyquda_measurement_utils.pion_utils_vibe_develop import my_gammas as pion_gammas
from pyquda_measurement_utils.pion_utils_vibe_develop import pyquda_gammas_order as pion_ids
from pyquda_measurement_utils.proton_qTMD_pyquda import my_gammas as proton_gammas
from pyquda_measurement_utils.proton_qTMD_pyquda import pyquda_gammas_order as proton_ids


EXPECTED_GAMMAS = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
EXPECTED_PYQUDA_ORDER = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]


def test_pion_and_proton_gamma_label_order_match():
    assert list(GAMMA_LABELS) == EXPECTED_GAMMAS
    assert pion_gammas == EXPECTED_GAMMAS
    assert proton_gammas == EXPECTED_GAMMAS


def test_pion_and_proton_pyquda_gamma_order_match_labels():
    assert list(PYQUDA_GAMMA_IDS) == EXPECTED_PYQUDA_ORDER
    assert pion_ids == EXPECTED_PYQUDA_ORDER
    assert proton_ids == EXPECTED_PYQUDA_ORDER
    assert len(EXPECTED_GAMMAS) == len(EXPECTED_PYQUDA_ORDER)


def _matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return np.asarray(gamma_like)


def test_pyquda_gamma_basic_trace_identities():
    identity = _matrix(gamma.gamma(0))
    gamma5 = _matrix(gamma.gamma(15))

    np.testing.assert_allclose(gamma5 @ gamma5, identity)
    assert np.isclose(np.trace(identity), 4)
    assert np.isclose(np.trace(gamma5), 0)

    for gamma_id in [1, 2, 4, 8]:
        gamma_mu = _matrix(gamma.gamma(gamma_id))
        np.testing.assert_allclose(gamma_mu @ gamma_mu, identity)
        np.testing.assert_allclose(gamma5 @ gamma_mu @ gamma5, -gamma_mu)


def test_pyquda_gamma_label_products_match_expected_composites():
    gamma_x = _matrix(gamma.gamma(1))
    gamma_y = _matrix(gamma.gamma(2))
    gamma_z = _matrix(gamma.gamma(4))
    gamma_t = _matrix(gamma.gamma(8))
    gamma5 = _matrix(gamma.gamma(15))

    np.testing.assert_allclose(_matrix(gamma.gamma(14)), gamma_x @ gamma5)
    np.testing.assert_allclose(_matrix(gamma.gamma(13)), -gamma_y @ gamma5)
    np.testing.assert_allclose(_matrix(gamma.gamma(11)), gamma_z @ gamma5)
    np.testing.assert_allclose(_matrix(gamma.gamma(7)), -gamma_t @ gamma5)
