import numpy as np

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _flow_times,
    _normalize_flow_type,
    _unique_zero_momentum_index,
    ringed_kinetic_pervec_from_emt,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import kinetic_spacetime_from_raw


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


def test_ringed_kinetic_is_extracted_from_zero_momentum_diagonal_trace():
    tmunu = np.zeros((2, 4, 4, 2, 3, 5), dtype=np.complex128)
    for source in range(2):
        for mu in range(4):
            tmunu[source, mu, mu, 1] = (source + 1) * (mu + 2)

    kinetic = ringed_kinetic_pervec_from_emt(tmunu, 1, spatial_volume=8)

    assert kinetic.shape == (2, 3, 5)
    np.testing.assert_allclose(kinetic[0], -2.0 * sum(range(2, 6)) / 8.0)
    np.testing.assert_allclose(kinetic[1], -4.0 * sum(range(2, 6)) / 8.0)
    np.testing.assert_allclose(
        kinetic_spacetime_from_raw(kinetic),
        np.mean(kinetic, axis=(0, -1)),
    )


def test_ringed_zero_momentum_should_be_unique():
    assert _unique_zero_momentum_index([[1, 0, 0, 0], [0, 0, 0, 0]]) == 1

    for qext in (
        [[1, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    ):
        try:
            _unique_zero_momentum_index(qext)
        except ValueError as err:
            assert "exactly one zero momentum" in str(err)
        else:
            raise AssertionError("ringed output should reject missing or duplicate zero momentum")


def test_ringed_zero_momentum_validation_precedes_inverter_setup():
    class FakeLatticeInfo:
        global_size = [2, 2, 2, 2]

    class FakeGauge:
        latt_info = FakeLatticeInfo()

    measurement = EMTDisconnectedQuark1pt({
        "qext": [[1, 0, 0, 0]],
        "pf": [0, 0, 0, 0],
        "p_2pt": [[1, 0, 0, 0]],
        "pos_boost": [0, 0, 0],
        "neg_boost": [0, 0, 0],
        "width": 1.0,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
    })

    try:
        measurement.flowed_fermionic_1pt(
            FakeGauge(),
            [0.1, 1.0, 1e-10, 10],
            [1, 4, 0],
            tag="emt",
            ringed_tag="ringed",
        )
    except ValueError as err:
        assert "exactly one zero momentum" in str(err)
    else:
        raise AssertionError("missing zero momentum should fail before inverter setup")
