from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

APPLICATION_ROOT = Path(__file__).resolve().parents[1] / "application"
if str(APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(APPLICATION_ROOT))

import analysis_helper.emt_disconnected_analysis as analysis
from analysis_helper.emt_disconnected_analysis import (
    read_gluon_loop,
    read_quark_loop,
    source_relative_loops,
)
from pyquda_measurement_utils.fermion_bilinear_basis import VECTOR_GAMMA_POSITIONS


def _write_quark(path, derivative, volume=2):
    with h5py.File(path, "w") as h5:
        h5.attrs["volume_norm"] = volume
        h5.attrs["effective_n_inversions"] = derivative.shape[0]
        h5.attrs["qext"] = [[0, 0, 0, 0]]
        h5.attrs["flow_times"] = [0.0]
        h5.attrs["noise_scheme"] = "zn"
        h5.attrs["hp_num_vectors"] = 1
        h5.attrs["source_bookkeeping_schema"] = "base_hp_v1"
        _write_provenance(h5, [1, 1, 1, derivative.shape[-1]])
        raw = h5.require_group("raw")
        raw.create_dataset("derivative_bilinear_pervec", data=derivative)
        raw.create_dataset("base_noise_index", data=np.arange(derivative.shape[0]))
        raw.create_dataset("hp_index", data=np.zeros(derivative.shape[0], dtype=np.int32))


def _write_provenance(h5, lattice_size):
    h5.attrs["loop_provenance_schema"] = "emt_disconnected_loop_provenance_v1"
    h5.attrs["global_lattice_size"] = lattice_size
    h5.attrs["momentum_phase_origin"] = [0, 0, 0, 0]
    h5.attrs["spatial_momentum_phase_convention"] = (
        "exp(-2pi*i*sum_j q_j*(x_j-origin_j)/L_j)"
    )
    h5.attrs["loop_time_convention"] = "absolute_lattice_time"


def test_quark_reader_chunks_vector_channels_and_rolls_to_relative_time(tmp_path):
    derivative = np.zeros((5, 16, 4, 1, 1, 6), dtype=np.complex128)
    for source in range(5):
        for nu, gamma_position in enumerate(VECTOR_GAMMA_POSITIONS):
            for mu in range(4):
                derivative[source, gamma_position, mu, 0, 0] = (
                    100 * source + 10 * nu + mu + np.arange(6)
                )
    path = tmp_path / "quark.h5"
    _write_quark(path, derivative)

    for chunk_size in (1, 3, 8):
        cumulative, counts, _, _, _ = read_quark_loop(
            path, [0, 0, 0, 2], source_chunk_size=chunk_size
        )
        b = derivative[:, VECTOR_GAMMA_POSITIONS]
        expected = 0.5 * (b + np.swapaxes(b, 1, 2))
        expected = np.roll(expected, -2, axis=-1)
        expected = np.cumsum(expected, axis=0) / counts[:, None, None, None, None, None] / 2
        np.testing.assert_allclose(cumulative, expected)


def test_gluon_reader_rolls_absolute_time(tmp_path):
    path = tmp_path / "gluon.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["qext"] = [[0, 0, 0, 0]]
        h5.attrs["flow_times"] = [0.0]
        _write_provenance(h5, [1, 1, 1, 6])
        group = h5.require_group("Tmunu")
        for mu in range(4):
            for nu in range(mu, 4):
                group.create_dataset(
                    f"T{mu + 1}{nu + 1}",
                    data=(10 * mu + nu + np.arange(6))[None, None],
                )
    loop, _, _ = read_gluon_loop(path, [0, 0, 0, 3])
    np.testing.assert_array_equal(loop[1, 3, 0, 0], np.roll(13 + np.arange(6), -3))
    np.testing.assert_array_equal(loop[3, 1], loop[1, 3])


def test_quark_reader_never_requests_nonvector_gamma(monkeypatch):
    data = np.zeros((2, 16, 4, 1, 1, 2), dtype=np.complex128)
    requested = []

    class Dataset:
        shape = data.shape
        ndim = data.ndim
        dtype = data.dtype

        def __getitem__(self, key):
            requested.append(key[1])
            return data[key]

    class File:
        attrs = {
            "volume_norm": 1,
            "effective_n_inversions": 2,
            "qext": [[0, 0, 0, 0]],
            "flow_times": [0.0],
            "noise_scheme": "zn",
            "hp_num_vectors": 1,
            "source_bookkeeping_schema": "base_hp_v1",
            "loop_provenance_schema": "emt_disconnected_loop_provenance_v1",
            "global_lattice_size": [1, 1, 1, 2],
            "momentum_phase_origin": [0, 0, 0, 0],
            "spatial_momentum_phase_convention": "exp(-2pi*i*sum_j q_j*(x_j-origin_j)/L_j)",
            "loop_time_convention": "absolute_lattice_time",
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __contains__(self, key):
            return key == "raw/derivative_bilinear_pervec"

        def __getitem__(self, key):
            if key == "raw/derivative_bilinear_pervec":
                return Dataset()
            if key == "raw/hp_index":
                return np.zeros(2, dtype=np.int32)
            return np.asarray([0, 1], dtype=np.int32)

    monkeypatch.setattr(analysis.h5py, "File", lambda *_args, **_kwargs: File())
    read_quark_loop("unused", [0, 0, 0, 0], source_chunk_size=1)
    assert set(requested) == set(VECTOR_GAMMA_POSITIONS)


def test_source_relative_conversion_applies_spatial_phase_and_periodic_time():
    loop = np.arange(2 * 2 * 4, dtype=np.complex128).reshape(2, 2, 4)
    qext = np.asarray([[0, 0, 0, 0], [1, -1, 0, 0]])
    source = [2, 1, 0, 3]
    converted = source_relative_loops(
        loop, qext, [source], [8, 8, 8, 4], [0, 0, 0, 0], q_axis=1
    )[0]
    phase = np.exp(-2j * np.pi * (2 - 1) / 8)
    np.testing.assert_allclose(converted[:, 0], np.roll(loop[:, 0], -3, axis=-1))
    np.testing.assert_allclose(
        converted[:, 1], phase * np.roll(loop[:, 1], -3, axis=-1)
    )


def test_old_loop_without_fourier_provenance_is_rejected(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["qext"] = [[0, 0, 0, 0]]
        h5.attrs["flow_times"] = [0.0]
        h5.attrs["volume_norm"] = 1
        h5.attrs["effective_n_inversions"] = 1
        raw = h5.require_group("raw")
        raw.create_dataset(
            "derivative_bilinear_pervec",
            data=np.zeros((1, 16, 4, 1, 1, 2), np.complex128),
        )
    with pytest.raises(KeyError, match="strict loop provenance"):
        read_quark_loop(path, [0, 0, 0, 0])


def test_inconsistent_fourier_origin_is_rejected(tmp_path):
    derivative = np.zeros((1, 16, 4, 1, 1, 2), dtype=np.complex128)
    path = tmp_path / "invalid-origin.h5"
    _write_quark(path, derivative)
    with h5py.File(path, "r+") as h5:
        h5.attrs["momentum_phase_origin"] = [1, 0, 0, 0]
    with pytest.raises(ValueError, match="invalid momentum_phase_origin"):
        read_quark_loop(path, [0, 0, 0, 0])
