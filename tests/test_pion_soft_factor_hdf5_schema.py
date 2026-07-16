import sys
from pathlib import Path
from unittest import SkipTest

import numpy as np

try:
    import h5py
except ModuleNotFoundError as err:
    raise SkipTest("h5py is required for pion soft-factor HDF5 schema tests") from err

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.io_corr import (
    save_pion_soft_factor_c2pt_hdf5_noRoll,
    save_pion_soft_factor_hdf5_noRoll,
    save_pion_soft_factor_qTMDWF_hdf5_noRoll,
)
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    gamma_matrices_numpy,
)
from pyquda_measurement_utils.pion_soft_factor_vibe_develop import (
    _gamma_matrix,
    _raw_gamma_by_label,
    soft_factor_gamma_channel_pairs,
)


class DummyLatticeInfo:
    mpi_rank = 0


def test_soft_factor_default_gamma_pairs_use_canonical_raw_basis():
    assert list(soft_factor_gamma_channel_pairs) == ["5", "I", "X", "Y", "X5", "Y5"]
    raw = gamma_matrices_numpy()
    for pair_label, (gamma1_label, gamma2_label) in soft_factor_gamma_channel_pairs.items():
        assert pair_label == gamma1_label == gamma2_label
        gamma_idx = GAMMA_LABELS.index(pair_label)
        np.testing.assert_array_equal(_gamma_matrix(_raw_gamma_by_label[pair_label]), raw[gamma_idx])


def test_pion_soft_factor_4pt_hdf5_schema(tmp_path):
    tag = str(tmp_path / "pion_soft_factor" / "schema")
    pion_channel_pairs = {
        "Z5-X5__Z5-X5": (np.eye(4), np.eye(4)),
    }
    gamma_channel_pairs = {
        "5__5": ("5", "5"),
        "I__I": ("I", "I"),
    }
    bT_dir = [0, 1]
    bT_length = 2
    tseplist = [2, 4]
    corr = np.arange(
        len(tseplist) * len(pion_channel_pairs) * len(gamma_channel_pairs) * len(bT_dir) * (bT_length + 1) * 4,
        dtype=np.float64,
    )
    corr = corr.reshape(len(tseplist), len(pion_channel_pairs), len(gamma_channel_pairs), len(bT_dir), bT_length + 1, 4)

    save_pion_soft_factor_hdf5_noRoll(
        corr,
        tag,
        pion_channel_pairs,
        gamma_channel_pairs,
        bT_dir,
        bT_length,
        tseplist,
        DummyLatticeInfo(),
    )

    with h5py.File(tag + ".h5", "r") as h5:
        assert h5.attrs["soft_factor_schema"] == "paired_channels_v2"
        assert h5.attrs["gamma_convention"] == "canonical_raw_pyquda"
        path = "pion_pair/Z5-X5__Z5-X5/gamma_pair/5__5/bX_0/ts2"
        assert path in h5
        assert h5[path].shape == (4,)
        np.testing.assert_array_equal(h5[path][...], corr[0, 0, 0, 0, 0])
        assert "pion_pair/Z5-X5__Z5-X5/gamma_pair/I__I/bY_2/ts4" in h5
        assert list(h5["gamma1_labels"].asstr()[...]) == ["5", "I"]
        assert list(h5["gamma2_labels"].asstr()[...]) == ["5", "I"]
        assert "physical_from_pyquda" in h5


def test_pion_soft_factor_diagnostic_c2pt_and_qtmdwf_schema(tmp_path):
    c2_tag = str(tmp_path / "pion_soft_factor_c2pt" / "schema")
    qtmdwf_tag = str(tmp_path / "pion_soft_factor_qTMDWF" / "schema")
    momentum = [0, 0, 2]

    c2 = np.arange(8, dtype=np.float64).astype(np.complex128)
    pair_label = "Z5-X5__Z5-X5"
    save_pion_soft_factor_c2pt_hdf5_noRoll(c2, c2_tag, pair_label, momentum, DummyLatticeInfo())

    with h5py.File(c2_tag + ".h5", "r") as h5:
        path = f"SS/pion_pair/{pair_label}/PX0PY0PZ2"
        assert path in h5
        np.testing.assert_array_equal(h5[path][...], c2)

    bT_dir = [0, 1]
    bT_length = 1
    bz_length = 2
    corr_count = len(bT_dir) * (bT_length + 1) * (bz_length + 1)
    qtmdwf = np.arange(corr_count * 8, dtype=np.float64).reshape(corr_count, 8).astype(np.complex128)
    save_pion_soft_factor_qTMDWF_hdf5_noRoll(qtmdwf, qtmdwf_tag, pair_label, momentum, bT_dir, bT_length, bz_length, DummyLatticeInfo())

    with h5py.File(qtmdwf_tag + ".h5", "r") as h5:
        prefix = f"SP/pion_pair/{pair_label}/PX0PY0PZ2"
        assert f"{prefix}/b_X/bT0/bz0" in h5
        assert f"{prefix}/b_Y/bT1/bz2" in h5
        np.testing.assert_array_equal(h5[f"{prefix}/b_X/bT0/bz0"][...], qtmdwf[0])
