from unittest import SkipTest

import numpy as np
import pytest

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for flowed-quark ringed tests") from err

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    ringed_kinetic_pervec_from_derivative,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    SHARD_SCHEMA,
    part_source_bookkeeping,
    shard_part_attrs,
    shard_part_path,
    write_raw_part_hdf5,
)
import pyquda_measurement_utils.flowed_quark_ringed_norm as ringed_module
from pyquda_measurement_utils.flowed_quark_ringed_norm import (
    RingedQuark1pt,
    finalize_ringed_quark_1pt_shards,
)


def _parameters(**overrides):
    values = {
        "config_num": 9,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 2,
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 2,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "gauge_preprocessing": "test",
    }
    values.update(overrides)
    return values


def test_ringed_is_kinetic_only_emt_runner_subclass():
    measurement = RingedQuark1pt(_parameters())
    assert isinstance(measurement, EMTDisconnectedQuark1pt)

    class Info:
        global_size = [2, 2, 2, 4]

    assert measurement._raw_step_tail_shapes(Info()) == {
        "kinetic_pervec": (4,)
    }
    assert measurement._metadata_datasets() == {}
    assert not hasattr(ringed_module, "FlowedQuarkRingedNorm")
    assert not hasattr(ringed_module, "flowed_kinetic_norm")
    assert not hasattr(ringed_module, "analyze_ringed_ensemble")
    assert not hasattr(ringed_module, "compute_ringed_factors")


def test_ringed_requires_configuration_and_unique_zero_momentum():
    with pytest.raises(ValueError, match="config_num is required"):
        RingedQuark1pt(_parameters(config_num=None))
    with pytest.raises(ValueError, match="only the unique zero momentum"):
        RingedQuark1pt(_parameters(qext=[[1, 0, 0, 0]]))


def test_ringed_contraction_matches_emt_vector_diagonal(monkeypatch):
    class Field:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.complex128)

        def __sub__(self, other):
            return Field(self.data - other.data)

    class Info:
        global_size = [1, 1, 1, 1]

    class Gauge:
        latt_info = Info()

        @staticmethod
        def covDev(field, direction):
            return Field((direction + 1) * field.data)

    rng = np.random.default_rng(17)
    shape = (1, 1, 1, 1, 1, 4, 3)
    xi = Field(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    eta = Field(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    emt = EMTDisconnectedQuark1pt({
        **_parameters(), "qext": [[0, 0, 0, 0]],
    })
    ringed = RingedQuark1pt(_parameters())

    def project_gamma(fields, _phases):
        return np.asarray(fields).reshape(16, -1).sum(axis=1)[:, None, None]

    def project_scalar(field, _phases):
        return np.asarray([[np.asarray(field).sum()]])

    monkeypatch.setattr(emt, "_project_gamma_fields", project_gamma)
    monkeypatch.setattr(emt, "_impose_P_Breit_slice", project_scalar)
    monkeypatch.setattr(ringed, "_impose_P_Breit_slice", project_scalar)
    _, derivative, _ = emt._get_primitive_bilinears_P_Breit_slice(
        Gauge(), Gauge(), xi, eta, [None]
    )
    direct = ringed._contract_flowed_source(
        Gauge(), Gauge(), xi, eta, [None]
    )["kinetic_pervec"]
    derived = ringed_kinetic_pervec_from_derivative(
        derivative[None, :, :, :, None, :], 0, spatial_volume=1
    )[0, 0]
    np.testing.assert_allclose(direct, derived, rtol=1e-13, atol=1e-13)


def _ringed_attrs(config_num):
    return {
        "measurement": "flowed_quark_ringed_norm",
        "output_kind": "flowed_quark_ringed_norm",
        "block_interval_solves": 64,
        "content": "kinetic_only",
        "producer": "standalone_ringed_shared_emt_runner",
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 2,
        "flow_times": np.asarray([0.0, 0.1, 0.2]),
        "qext": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "mass": 0.1,
        "csw": 1.0,
        "gauge_preprocessing": "test",
        "t_boundary": -1,
        "flavor_convention": "single_flavor_trace_for_this_dirac_operator",
        "derivative_convention": "gamma_mu*(Dplus_mu-Dminus_mu)",
        "kinetic_relation_to_emt": (
            "K=-2*sum_mu(L_D[gamma_mu,mu,q0])/spatial_volume"
        ),
        "n_zn": 4,
        "config_num": config_num,
        "noise_stream": 7,
        "noise_generator": "splitmix64_global_coordinate_v1",
        "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
        "noise_scheme": "hierarchical_probing",
        "hp_num_vectors": 2,
        "hp_ordering": "interleaved_xyzt_binary_projected_to_evenodd",
        "volume_norm": 8,
        "ringed_factors_stored": False,
    }


def _write_ringed_base(shard_dir, tag, config_num, base_idx):
    path = shard_part_path(shard_dir, tag, base_idx, 0, 0, 2)
    attrs = shard_part_attrs(_ringed_attrs(config_num), base_idx, 0, 0, 2, 2)
    bookkeeping = part_source_bookkeeping(base_idx, 0, 2, 2)
    kinetic = np.full((2, 3, 4), -(config_num + 1), dtype=np.complex128)
    write_raw_part_hdf5(
        path, {"kinetic_pervec": kinetic}, attrs, bookkeeping
    )


def test_ringed_finalize_stores_only_kinetic_and_base_hp_bookkeeping(tmp_path):
    tag = str(tmp_path / "FlowedQuarkRinged" / "lat.FlowedQuarkRinged.9.0.sm")
    shard_dir = tmp_path / "FlowedQuarkRinged" / "shards"
    _write_ringed_base(shard_dir, tag, 9, 0)
    finalize_ringed_quark_1pt_shards(shard_dir, tag, 1)

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["content"] == "kinetic_only"
        assert set(h5) == {"flow_times", "raw", "avg"}
        assert set(h5["raw"]) == {
            "kinetic_pervec", "base_noise_index", "hp_index"
        }
        assert set(h5["avg"]) == {"kinetic_spacetime"}
        assert h5["raw/kinetic_pervec"].shape == (2, 3, 4)
        np.testing.assert_array_equal(h5["raw/base_noise_index"], [0, 0])
        np.testing.assert_allclose(h5["avg/kinetic_spacetime"], -10.0)
        assert "spin_color_dilution" not in h5.attrs
        assert "spin_index" not in h5["raw"]
        assert "color_index" not in h5["raw"]


def test_timer_default_is_off(monkeypatch):
    monkeypatch.delenv("PYQUDA_MEASUREMENT_TIMERS", raising=False)
    from pyquda_measurement_utils.tools import timing_enabled

    assert not timing_enabled()
    monkeypatch.setenv("PYQUDA_MEASUREMENT_TIMERS", "1")
    assert timing_enabled()
