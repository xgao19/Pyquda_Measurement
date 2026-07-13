import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt_module

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _flow_times,
    _normalize_flow_type,
    _unique_zero_momentum_index,
    ringed_kinetic_pervec_from_emt,
    validate_quark_gluon_loop_axes,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import kinetic_spacetime_from_raw
from pyquda_measurement_utils.disconnected_shards import (
    append_completed_base,
    prepare_sample_log,
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


def test_counter_configuration_validation_precedes_inverter_setup():
    class FakeLatticeInfo:
        global_size = [2, 2, 2, 2]
        t_boundary = -1

    class FakeGauge:
        latt_info = FakeLatticeInfo()

    measurement = EMTDisconnectedQuark1pt({
        "qext": [[0, 0, 0, 0]],
        "pf": [0, 0, 0, 0],
        "p_2pt": [[0, 0, 0, 0]],
        "pos_boost": [0, 0, 0],
        "neg_boost": [0, 0, 0],
        "width": 1.0,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
    })

    with pytest.raises(ValueError, match="config_num is required"):
        measurement.flowed_fermionic_1pt(
            FakeGauge(), [0.1, 1.0, 1e-10, 10], [1, 4, 0],
            tag="emt", ringed_tag="ringed",
        )


def test_logged_base_skips_before_inverter_and_without_hdf5(tmp_path, monkeypatch):
    class FakeLatticeInfo:
        global_size = [2, 2, 2, 2]
        t_boundary = -1
        mpi_rank = 0

    class FakeGauge:
        latt_info = FakeLatticeInfo()

    class FakeComm:
        @staticmethod
        def bcast(value, root=0):
            return value

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 17,
        "qext": [[0, 0, 0, 0]],
        "pf": [0, 0, 0, 0],
        "p_2pt": [[0, 0, 0, 0]],
        "pos_boost": [0, 0, 0],
        "neg_boost": [0, 0, 0],
        "width": 1.0,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
    })
    inv = [0.1, 1.0, 1e-10, 10]
    rand = [1, 4, 0]
    tag = str(tmp_path / "lat.EMTc.17.0.sm")
    log = tmp_path / "sample.log"
    attrs = measurement._measurement_attrs(
        FakeLatticeInfo(), inv, rand, 17, 0, 1, 8
    )
    common = {
        key: value for key, value in attrs.items()
        if key not in {"n_vec", "n_base_noise", "effective_n_inversions"}
    }
    common["output_kind"] = "emt_quark_1pt"
    common["block_interval_solves"] = 64
    prepare_sample_log(log, tag, common)
    append_completed_base(log, tag, common, 0)

    monkeypatch.setattr(emt_module, "getMPIComm", lambda: FakeComm())
    monkeypatch.setattr(
        emt_module.core, "getDirac",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inverter should not initialize")),
    )
    assert measurement.flowed_fermionic_1pt(
        FakeGauge(), inv, rand, tag=tag, sample_log_file=log
    ) == (None, None)
    assert not list(tmp_path.rglob("*.h5"))


def test_quark_gluon_axes_must_match_before_analysis():
    qext = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32)
    flow_times = np.asarray([0.0, 0.207936])
    validate_quark_gluon_loop_axes(qext, qext.copy(), flow_times, flow_times.copy())

    with pytest.raises(ValueError, match="matching qext"):
        validate_quark_gluon_loop_axes(qext, qext[:1], flow_times, flow_times)
    with pytest.raises(ValueError, match="matching flow_times"):
        validate_quark_gluon_loop_axes(qext, qext, flow_times, [0.0, 0.1])
