from pathlib import Path
import sys

import h5py
import numpy as np

APPLICATION_ROOT = Path(__file__).resolve().parents[1] / "application"
if str(APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(APPLICATION_ROOT))

import analysis_helper.emt_disconnected_analysis as analysis
from analysis_helper.emt_disconnected_analysis import (
    read_gluon_loop,
    read_quark_loop,
)
from pyquda_measurement_utils.fermion_bilinear_basis import VECTOR_GAMMA_POSITIONS


def _write_quark(path, derivative, volume=2):
    with h5py.File(path, "w") as h5:
        h5.attrs["volume_norm"] = volume
        h5.attrs["effective_n_inversions"] = derivative.shape[0]
        h5.attrs["qext"] = [[0, 0, 0, 0]]
        h5.attrs["flow_times"] = [0.0]
        raw = h5.require_group("raw")
        raw.create_dataset("derivative_bilinear_pervec", data=derivative)
        raw.create_dataset("source_index", data=np.arange(derivative.shape[0]))
        raw.create_dataset("base_noise_index", data=np.arange(derivative.shape[0]))
        raw.create_dataset("hp_index", data=np.zeros(derivative.shape[0], dtype=np.int32))


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
            path, source_t=2, source_chunk_size=chunk_size
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
        group = h5.require_group("Tmunu")
        for mu in range(4):
            for nu in range(mu, 4):
                group.create_dataset(
                    f"T{mu + 1}{nu + 1}",
                    data=(10 * mu + nu + np.arange(6))[None, None],
                )
    loop, _, _ = read_gluon_loop(path, source_t=3)
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
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            if key == "raw/derivative_bilinear_pervec":
                return Dataset()
            return np.asarray([0, 1], dtype=np.int32)

    monkeypatch.setattr(analysis.h5py, "File", lambda *_args, **_kwargs: File())
    read_quark_loop("unused", source_t=0, source_chunk_size=1)
    assert set(requested) == set(VECTOR_GAMMA_POSITIONS)
