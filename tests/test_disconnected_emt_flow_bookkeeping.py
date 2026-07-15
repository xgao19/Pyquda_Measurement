import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt_module

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _flow_times,
    _interval_batches,
    _normalize_flow_type,
    _positive_flow_batch_size,
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


def test_flow_batch_size_and_interval_validation():
    assert _positive_flow_batch_size(1) == 1
    assert _positive_flow_batch_size(np.int64(8)) == 8
    assert _interval_batches(0, 5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert _interval_batches(3, 16, 6) == [(3, 9), (9, 15), (15, 16)]
    for invalid in (0, -1, 1.5, "2", True):
        with pytest.raises(ValueError, match="positive integer"):
            _positive_flow_batch_size(invalid)


def test_hp_part_batches_never_cross_part_boundaries():
    hp16 = [
        _interval_batches(start, stop, 6)
        for _, start, stop in emt_module.base_part_ranges(16, 64)
    ]
    assert hp16 == [[(0, 6), (6, 12), (12, 16)]]
    hp256 = [
        _interval_batches(start, stop, 20)
        for _, start, stop in emt_module.base_part_ranges(256, 64)
    ]
    assert hp256 == [
        [(0, 20), (20, 40), (40, 60), (60, 64)],
        [(64, 84), (84, 104), (104, 124), (124, 128)],
        [(128, 148), (148, 168), (168, 188), (188, 192)],
        [(192, 212), (212, 232), (232, 252), (252, 256)],
    ]


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
    assert "flow_batch_size" not in attrs_one
    assert "flow_batch_size" not in attrs_two
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

    class FakeLatticeInfo:
        global_size = [1, 1, 1, 1]

    class FakeGauge:
        def __init__(self):
            self.latt_info = FakeLatticeInfo()
            self.pure_gauge = FakePureGauge()
            self.context_entries = 0
            self.context_exits = 0

        def use(self):
            gauge = self

            class GaugeContext:
                def __enter__(self):
                    gauge.context_entries += 1
                    return gauge.pure_gauge

                def __exit__(self, exc_type, exc_value, traceback):
                    gauge.context_exits += 1

            return GaugeContext()

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
        lambda fields, _phases: np.asarray(fields).reshape(fields.shape[0], 1, 1),
    )
    monkeypatch.setattr(
        measurement,
        "_impose_P_Breit_slice",
        lambda field, _phases: np.asarray(field).reshape(1, 1),
    )
    data = (
        np.arange(12, dtype=np.float64).reshape(1, 1, 1, 1, 1, 4, 3)
        + 1j * np.arange(12, 24, dtype=np.float64).reshape(1, 1, 1, 1, 1, 4, 3)
    )
    gauge = FakeGauge()
    local, derivative, flowed_noise_norm = measurement._get_primitive_bilinears_P_Breit_slice(
        gauge, FakeField(data), FakeField(2 * data), [None]
    )
    gamma = gamma_matrices_numpy()
    expected_local = np.einsum(
        "wtzyxia,gij,wtzyxja->gwtzyx", data.conj(), gamma, 2 * data
    ).reshape(16, 1, 1)
    expected_derivative = np.empty((16, 4, 1, 1), dtype=np.complex128)
    for mu in range(4):
        symmetric_difference = ((mu + 1) - (mu + 5)) * 2 * data
        expected_derivative[:, mu] = -0.5 * np.einsum(
            "wtzyxia,gij,wtzyxja->gwtzyx",
            data.conj(), gamma, symmetric_difference,
        ).reshape(16, 1, 1)
    expected_norm = np.einsum(
        "wtzyxia,wtzyxia->wtzyx", data.conj(), data
    ).reshape(1, 1)
    assert local.shape == (16, 1, 1)
    assert derivative.shape == (16, 4, 1, 1)
    assert flowed_noise_norm.shape == (1, 1)
    np.testing.assert_allclose(local, expected_local, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        flowed_noise_norm, expected_norm, rtol=1e-13, atol=1e-13
    )
    assert gauge.context_entries == 1
    assert gauge.context_exits == 1
    assert gauge.pure_gauge.calls == [0, 4, 1, 5, 2, 6, 3, 7]


def test_flowed_batch_uses_one_context_and_one_multifield_flow_per_step(monkeypatch):
    class FakeInfo:
        global_size = [1, 1, 1, 1]
        mpi_rank = 0

    class FakeField:
        def __init__(self, identity):
            self.identity = identity

    class FakeFlowedOwner:
        def __init__(self, fields):
            self.fields = [FakeField(("flowed", field.identity)) for field in fields]

        def __getitem__(self, index):
            return self.fields[index]

    class FakeGauge:
        def __init__(self, shared=None):
            self.latt_info = FakeInfo()
            self.shared = shared or {
                "contexts": 0,
                "flows": [],
                "antiperiodic": 0,
            }

        def copy(self):
            return FakeGauge(self.shared)

        def setAntiPeriodicT(self):
            self.shared["antiperiodic"] += 1

        def use(self):
            shared = self.shared

            class Context:
                def __enter__(self):
                    shared["contexts"] += 1
                    return object()

                def __exit__(self, *_args):
                    return None

            return Context()

        def gradientFlow(self, fields, flow_type, n_steps, step_size):
            self.shared["flows"].append(
                (len(fields), flow_type, n_steps, step_size)
            )
            return FakeFlowedOwner(fields)

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 1,
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.2,
        "flow_steps": 2,
    })
    primitive_calls = []

    def fake_primitive(_gauge, xi, eta, _phases, gauge_dirac=None):
        assert gauge_dirac is not None
        primitive_calls.append((xi.identity, eta.identity, gauge_dirac))
        return (
            np.full((16, 1, 1), len(primitive_calls), dtype=np.complex128),
            np.full((16, 4, 1, 1), len(primitive_calls), dtype=np.complex128),
            np.full((1, 1), len(primitive_calls), dtype=np.complex128),
        )

    monkeypatch.setattr(
        measurement, "_get_primitive_bilinears_P_Breit_slice", fake_primitive
    )
    monkeypatch.setattr(emt_module.convert, "multiField", lambda fields: fields)
    gauge = FakeGauge()
    xis = [FakeField(("xi", index)) for index in range(3)]
    etas = [FakeField(("eta", index)) for index in range(3)]
    local, derivative, norm = measurement._measure_flowed_batch(
        gauge, xis, etas, [None]
    )

    assert local.shape == (3, 16, 1, 3, 1)
    assert derivative.shape == (3, 16, 4, 1, 3, 1)
    assert norm.shape == (3, 1, 3, 1)
    assert gauge.shared["contexts"] == 3
    assert gauge.shared["antiperiodic"] == 1
    assert gauge.shared["flows"] == [
        (6, "wilson", 10, 0.02),
        (6, "wilson", 1, 0.2),
    ]
    assert len(primitive_calls) == 9


def test_plain_noise_batches_across_bases_and_logs_only_successful_batches(
    tmp_path, monkeypatch
):
    class FakeInfo:
        global_size = [1, 1, 1, 1]
        mpi_rank = 0

    class FakeGauge:
        latt_info = FakeInfo()

    class FakeComm:
        @staticmethod
        def Barrier():
            return None

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 3,
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 0,
    })
    batches = []
    writes = []
    completed = []

    monkeypatch.setattr(emt_module, "getMPIComm", lambda: FakeComm())
    monkeypatch.setattr(
        emt_module,
        "iter_noise_base_hp_interval",
        lambda _info, base, hp_start, hp_stop, *_args, **_kwargs: iter(
            [(base, base, hp_start, object())]
        ),
    )

    def fake_measure(_gauge, _dirac, records, _phases):
        bases = [record[1] for record in records]
        batches.append(bases)
        if bases == [2, 3]:
            raise RuntimeError("synthetic batch failure")
        count = len(records)
        return (
            np.zeros((count, 16, 1, 1, 1), dtype=np.complex128),
            np.zeros((count, 16, 4, 1, 1, 1), dtype=np.complex128),
            np.zeros((count, 1, 1, 1), dtype=np.complex128),
        )

    monkeypatch.setattr(measurement, "_invert_and_measure_batch", fake_measure)
    monkeypatch.setattr(
        emt_module,
        "write_raw_part_hdf5",
        lambda path, *_args, **_kwargs: writes.append(path.name),
    )
    monkeypatch.setattr(
        emt_module,
        "append_completed_base",
        lambda _log, _tag, _attrs, base: completed.append(base),
    )
    attrs = {
        "config_num": 3,
        "noise_stream": 0,
        "n_vec": 5,
        "n_base_noise": 5,
        "effective_n_inversions": 5,
    }

    with pytest.raises(RuntimeError, match="synthetic batch failure"):
        measurement._measure_base_shards(
            FakeGauge(), object(), [0.1, 1.0, 1e-10, 10], [5, 4, 0],
            str(tmp_path / "test.h5"), [None], attrs,
            tmp_path / "shards", tmp_path / "sample.log", 0, 5, 64,
            set(), 2,
        )
    assert batches == [[0, 1], [2, 3]]
    assert completed == [0, 1]
    assert len(writes) == 2


def test_hp_batching_stays_within_one_base_and_part(tmp_path, monkeypatch):
    class FakeInfo:
        global_size = [1, 1, 1, 1]
        mpi_rank = 0

    class FakeGauge:
        latt_info = FakeInfo()

    class FakeComm:
        @staticmethod
        def Barrier():
            return None

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 4,
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 0,
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 16,
    })
    batches = []
    completed = []
    writes = []
    monkeypatch.setattr(emt_module, "getMPIComm", lambda: FakeComm())
    monkeypatch.setattr(
        emt_module,
        "iter_noise_base_hp_interval",
        lambda _info, base, hp_start, hp_stop, *_args, **_kwargs: iter(
            [(base * 16 + hp, base, hp, object()) for hp in range(hp_start, hp_stop)]
        ),
    )

    def fake_measure(_gauge, _dirac, records, _phases):
        records = list(records)
        batches.append([(record[1], record[2]) for record in records])
        count = len(records)
        return (
            np.zeros((count, 16, 1, 1, 1), dtype=np.complex128),
            np.zeros((count, 16, 4, 1, 1, 1), dtype=np.complex128),
            np.zeros((count, 1, 1, 1), dtype=np.complex128),
        )

    monkeypatch.setattr(measurement, "_invert_and_measure_batch", fake_measure)
    monkeypatch.setattr(
        emt_module, "write_raw_part_hdf5",
        lambda path, *_args, **_kwargs: writes.append(path.name),
    )
    monkeypatch.setattr(
        emt_module, "append_completed_base",
        lambda _log, _tag, _attrs, base: completed.append(base),
    )
    attrs = {
        "config_num": 4,
        "noise_stream": 0,
        "n_vec": 1,
        "n_base_noise": 1,
        "effective_n_inversions": 16,
    }
    measurement._measure_base_shards(
        FakeGauge(), object(), [0.1, 1.0, 1e-10, 10], [1, 4, 0],
        str(tmp_path / "test.h5"), [None], attrs,
        tmp_path / "shards", tmp_path / "sample.log", 0, 1, 64,
        set(), 6,
    )
    assert [[hp for _, hp in batch] for batch in batches] == [
        list(range(0, 6)), list(range(6, 12)), list(range(12, 16))
    ]
    assert all({base for base, _ in batch} == {0} for batch in batches)
    assert completed == [0]
    assert len(writes) == 1


def test_source_loop_uses_one_full_mg_setup_then_thin_restores(tmp_path, monkeypatch):
    class FakeLatticeInfo:
        global_size = [1, 1, 1, 1]
        t_boundary = -1
        mpi_rank = 0

    class FakeGauge:
        latt_info = FakeLatticeInfo()

    class FakeComm:
        @staticmethod
        def bcast(value, root=0):
            return value

        @staticmethod
        def Barrier():
            return None

    class FakeDirac:
        def __init__(self):
            self.load_calls = []

        def loadGauge(self, gauge, thin_update_only=False):
            self.load_calls.append(bool(thin_update_only))

        @staticmethod
        def invert(source):
            return source

    class FakeMomentumPhase:
        def __init__(self, latt_info):
            pass

        @staticmethod
        def getPhases(momentum, source_position):
            return [None]

    measurement = EMTDisconnectedQuark1pt({
        "config_num": 19,
        "qext": [[0, 0, 0, 0]],
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 0,
    })
    fake_dirac = FakeDirac()
    monkeypatch.setattr(emt_module, "getMPIComm", lambda: FakeComm())
    monkeypatch.setattr(emt_module.core, "getDirac", lambda *args: fake_dirac)
    monkeypatch.setattr(emt_module.phase, "MomentumPhase", FakeMomentumPhase)
    monkeypatch.setattr(
        emt_module,
        "iter_noise_base_hp_interval",
        lambda _info, base, hp_start, hp_stop, *_args, **_kwargs: iter(
            [(base, base, hp_start, object())]
        ),
    )
    monkeypatch.setattr(emt_module, "write_raw_part_hdf5", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        measurement,
        "_measure_flowed_batch",
        lambda _gauge, xis, _etas, _phases: (
            np.zeros((len(xis), 16, 1, 1, 1), dtype=np.complex128),
            np.zeros((len(xis), 16, 4, 1, 1, 1), dtype=np.complex128),
            np.zeros((len(xis), 1, 1, 1), dtype=np.complex128),
        ),
    )

    measurement.flowed_fermionic_1pt(
        FakeGauge(),
        [0.1, 1.0, 1e-10, 10],
        [2, 4, 7],
        tag=str(tmp_path / "S1T1.EMTc.19.0.test.h5"),
        shard_dir=tmp_path / "shards",
        sample_log_file=tmp_path / "sample.log",
        base_start=0,
        base_stop=2,
    )
    assert fake_dirac.load_calls == [False, True, True]
