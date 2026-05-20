import numpy as np

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    _flow_times,
    _normalize_flow_type,
)


def test_flow_times_include_zero_flow_and_regular_steps():
    np.testing.assert_allclose(_flow_times(0.02, 4), [0.0, 0.02, 0.04, 0.06, 0.08])


def test_flow_type_normalization_accepts_supported_cases_only():
    assert _normalize_flow_type(" Wilson ") == "wilson"
    assert _normalize_flow_type("SYMANZIK") == "symanzik"

    try:
        _normalize_flow_type("zeuthen")
    except ValueError:
        return
    raise AssertionError("_normalize_flow_type should reject unsupported flow types")
