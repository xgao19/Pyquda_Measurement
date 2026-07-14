import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt_module

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _flow_times,
    _normalize_flow_type,
    _unique_zero_momentum_index,
    emt_tensor_from_derivative_bilinear,
    parse_multigrid_blocks,
    ringed_kinetic_pervec_from_derivative,
    validate_quark_gluon_loop_axes,
)
from pyquda_measurement_utils.flowed_quark_ringed_norm import kinetic_spacetime_from_raw
from pyquda_measurement_utils.fermion_bilinear_basis import gamma_matrices_numpy
from pyquda_measurement_utils.disconnected_shards import (
    append_completed_base,
    prepare_sample_log,
    sample_log_fingerprint,
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


def test_multigrid_block_parser_supports_one_and_two_coarsening_levels():
    assert parse_multigrid_blocks("8.8.4.4") == [[8, 8, 4, 4]]
    assert parse_multigrid_blocks("4.4.4.4;4.4.4.4") == [
        [4, 4, 4, 4], [4, 4, 4, 4]
    ]
    assert parse_multigrid_blocks([[8, 8, 4, 4]]) == [[8, 8, 4, 4]]
    for invalid in ("", "4.4.4", "4.4.x.4", "4.4.0.4"):
        with pytest.raises(ValueError):
            parse_multigrid_blocks(invalid)


def test_multigrid_blocks_are_measurement_identity():
    common = {
        "config_num": 7,
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
    }
    one = EMTDisconnectedQuark1pt({**common, "multigrid": "4.4.4.4"})
    two = EMTDisconnectedQuark1pt({
        **common, "multigrid": "4.4.4.4;4.4.4.4"
    })

    class Info:
        t_boundary = -1

    inv = [-0.015, 1.0372, 1e-10, 1000]
    rand = [8, 4, 0]
    attrs_one = one._measurement_attrs(Info(), inv, rand, 7, 0, 8, 64**3)
    attrs_two = two._measurement_attrs(Info(), inv, rand, 7, 0, 8, 64**3)
    np.testing.assert_array_equal(attrs_one["multigrid_blocks"], [[4, 4, 4, 4]])
    np.testing.assert_array_equal(
        attrs_two["multigrid_blocks"], [[4, 4, 4, 4], [4, 4, 4, 4]]
    )
    assert sample_log_fingerprint(attrs_one) != sample_log_fingerprint(attrs_two)


def test_ringed_kinetic_is_extracted_from_vector_derivative_diagonal():
    derivative = np.zeros((2, 16, 4, 2, 3, 5), dtype=np.complex128)
    vector_positions = [3, 5, 7, 1]
    for source in range(2):
        for mu, gamma_position in enumerate(vector_positions):
            derivative[source, gamma_position, mu, 1] = (source + 1) * (mu + 2)

    kinetic = ringed_kinetic_pervec_from_derivative(derivative, 1, spatial_volume=8)

    assert kinetic.shape == (2, 3, 5)
    np.testing.assert_allclose(kinetic[0], -2.0 * sum(range(2, 6)) / 8.0)
    np.testing.assert_allclose(kinetic[1], -4.0 * sum(range(2, 6)) / 8.0)
    np.testing.assert_allclose(
        kinetic_spacetime_from_raw(kinetic),
        np.mean(kinetic, axis=(0, -1)),
    )

    tmunu = emt_tensor_from_derivative_bilinear(derivative)
    for mu in range(4):
        np.testing.assert_array_equal(
            tmunu[:, mu, mu], derivative[:, vector_positions[mu], mu]
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


def test_emt_default_hp_ordering_is_isotropic_four_dimensional():
    measurement = EMTDisconnectedQuark1pt({
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 1,
    })
    assert measurement.hp_ordering == "interleaved_xyzt_binary_projected_to_evenodd"


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
            tag="emt",
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


def test_complete_gamma_basis_does_not_add_covdev_calls(monkeypatch):
    class FakeField:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __sub__(self, other):
            return FakeField(self.data - other.data)

    class FakePureGauge:
        def __init__(self):
            self.calls = []

        def covDev(self, field, direction):
            self.calls.append(direction)
            return FakeField((direction + 1) * field.data)

    class FakeGaugeDirac:
        def loadGauge(self, gauge):
            return None

    class FakeLatticeInfo:
        global_size = [1, 1, 1, 1]

    class FakeGauge:
        latt_info = FakeLatticeInfo()
        pure_gauge = FakePureGauge()
        gauge_dirac = FakeGaugeDirac()

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 1,
        "qext": [[0, 0, 0, 0]],
        "pf": [0, 0, 0, 0],
        "p_2pt": [[0, 0, 0, 0]],
        "pos_boost": [0, 0, 0],
        "neg_boost": [0, 0, 0],
        "width": 1.0,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 0,
    })
    monkeypatch.setattr(measurement, "_gamma_stack_for", lambda _ref: gamma_matrices_numpy())
    monkeypatch.setattr(
        measurement,
        "_project_gamma_fields",
        lambda fields, _phases: np.zeros((fields.shape[0], 1, 1), dtype=np.complex128),
    )
    monkeypatch.setattr(
        measurement,
        "_impose_P_Breit_slice",
        lambda _field, _phases: np.zeros((1, 1), dtype=np.complex128),
    )
    data = np.ones((1, 1, 1, 1, 1, 4, 3), dtype=np.complex128)
    local, derivative, flowed_noise_norm = measurement._get_primitive_bilinears_P_Breit_slice(
        FakeGauge(), FakeField(data), FakeField(2 * data), [None]
    )
    assert local.shape == (16, 1, 1)
    assert derivative.shape == (16, 4, 1, 1)
    assert flowed_noise_norm.shape == (1, 1)
    assert FakeGauge.pure_gauge.calls == [0, 4, 1, 5, 2, 6, 3, 7]
