"""Round-trip dense soft-factor data and preserve independent ordered pairs."""

import numpy as np
import h5py
import pytest

from pyquda_measurement_utils.io_corr import (
    PION_SOFT_FACTOR_STORAGE_SCHEMA,
    save_pion_soft_factor_c2pt_hdf5_noRoll,
    save_pion_soft_factor_hdf5_noRoll,
    save_pion_soft_factor_qTMDWF_hdf5_noRoll,
)
from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS, gamma_matrices_numpy
from pyquda_measurement_utils.pion_soft_factor_vibe_develop import (
    _raw_gamma_by_label, soft_factor_gamma_channel_pairs,
)


def inputs():
    pairs = np.asarray([[[0, 0, fw], [0, 0, bw]] for fw in (4, 5) for bw in (4, 5)])
    raw = gamma_matrices_numpy()
    source = (raw[GAMMA_LABELS.index("Z")] - raw[GAMMA_LABELS.index("X")]) @ raw[0]
    plus = (raw[GAMMA_LABELS.index("Z")] + raw[GAMMA_LABELS.index("X")]) @ raw[0]
    pion = {"Z5-X5__Z5-X5": (source, source)}
    ss = {**pion, "Z5-X5__Z5+X5": (source, plus)}
    return pairs, source, pion, ss


def values(shape):
    count = int(np.prod(shape))
    return (np.arange(count) + 1j * np.arange(count)[::-1]).reshape(shape)


def test_soft_factor_default_gamma_pairs_use_canonical_raw_basis():
    assert list(soft_factor_gamma_channel_pairs) == ["5", "I", "X", "Y", "X5", "Y5"]
    raw = gamma_matrices_numpy()
    for label, (g1, g2) in soft_factor_gamma_channel_pairs.items():
        assert label == g1 == g2
        matrix = _raw_gamma_by_label[label]
        np.testing.assert_array_equal(np.asarray(getattr(matrix, "matrix", matrix)), raw[GAMMA_LABELS.index(label)])


def test_dense_c4_round_trip_axes_metadata_and_both_pz9_pairs(tmp_path):
    pairs, source, pion, _ = inputs()
    vectors = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
    data = values((4, 6, 3, 8))
    tag = str(tmp_path / "C4")
    save_pion_soft_factor_hdf5_noRoll(data, tag, pairs, pion, soft_factor_gamma_channel_pairs,
                                      vectors, 6, attrs={"source_time": 3, "lattice": [8] * 4})
    with h5py.File(tag + ".h5", "r") as handle:
        assert handle.attrs["soft_factor_schema"] == "full_spin_color_v3"
        assert handle.attrs["soft_factor_storage_schema"] == PION_SOFT_FACTOR_STORAGE_SCHEMA
        assert handle.attrs["dataset_axes"] == "quark_pair,gamma_pair,displacement,time"
        assert handle.attrs["t_sep"] == 6
        assert handle.attrs["time_axis"] == "source_relative"
        assert handle["C4"].dtype == np.dtype("complex128")
        np.testing.assert_array_equal(handle["C4"][()], data)
        np.testing.assert_array_equal(handle["quark_momentum_pairs"][()], pairs)
        np.testing.assert_array_equal(handle["meson_momentum_list"][:, 2], [8, 9, 9, 10])
        assert not np.array_equal(handle["C4"][1], handle["C4"][2])
        np.testing.assert_array_equal(handle["displacement_vectors"][()], vectors)
        np.testing.assert_array_equal(handle["time"][()], np.arange(8))
        np.testing.assert_array_equal(handle["pion_source_matrices"][()], [source])
        np.testing.assert_array_equal(handle["pion_sink_matrices"][()], [source])
        assert list(handle["gamma_pair_labels"].asstr()[()]) == list(soft_factor_gamma_channel_pairs)
        assert all(isinstance(item, h5py.Dataset) for item in handle.values())
        assert "physical_from_pyquda" in handle


def test_dense_ss_and_sp_round_trip_without_implicit_time_roll(tmp_path):
    pairs, source, _, pion_pairs = inputs()
    ss, sp = values((4, 2, 8)), values((4, 16, 3, 8))
    vectors = [[0, 0, 0], [1, 0, 0], [1, 0, 2]]
    save_pion_soft_factor_c2pt_hdf5_noRoll(ss, str(tmp_path / "SS"), pairs, pion_pairs)
    save_pion_soft_factor_qTMDWF_hdf5_noRoll(sp, str(tmp_path / "SP"), pairs,
                                            "Z5-X5", source, vectors)
    for kind, expected in [("SS", ss), ("SP", sp)]:
        with h5py.File(tmp_path / f"{kind}.h5", "r") as handle:
            np.testing.assert_array_equal(handle[kind][()], expected)
            np.testing.assert_array_equal(handle["quark_momentum_pairs"][()], pairs)
            assert "t_sep" not in handle.attrs
            if kind == "SS":
                assert list(handle["pion_pair_labels"].asstr()[()]) == list(pion_pairs)
                assert handle["pion_sink_matrices"].shape == (2, 4, 4)
            else:
                assert list(handle["sink_gamma_labels"].asstr()[()]) == list(GAMMA_LABELS)
                np.testing.assert_array_equal(handle["pion_source_matrix"][()], source)
                np.testing.assert_array_equal(handle["displacement_vectors"][()], vectors)


@pytest.mark.parametrize("problem", ["duplicate_pairs", "wrong_pair_shape", "wrong_corr_shape", "nonfinite", "bad_vectors", "unknown_gamma"])
def test_dense_writer_rejects_invalid_inputs_before_opening_output(tmp_path, problem):
    pairs, _, pion, _ = inputs()
    data = values((4, 6, 1, 8))
    vectors = [[0, 0, 0]]
    channels = dict(soft_factor_gamma_channel_pairs)
    if problem == "duplicate_pairs":
        pairs[2] = pairs[1]
    elif problem == "wrong_pair_shape":
        pairs = pairs[:, 0]
    elif problem == "wrong_corr_shape":
        data = data[:, :, 0]
    elif problem == "nonfinite":
        data[0, 0, 0, 0] = np.nan
    elif problem == "bad_vectors":
        vectors = [[0.5, 0, 0]]
    else:
        channels["5"] = ("unknown", "5")
    tag = str(tmp_path / "invalid")
    with pytest.raises(ValueError):
        save_pion_soft_factor_hdf5_noRoll(data, tag, pairs, pion, channels, vectors, 6)
    assert not (tmp_path / "invalid.h5").exists()
