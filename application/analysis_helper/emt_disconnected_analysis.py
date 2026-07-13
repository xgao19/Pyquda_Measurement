"""Memory-bounded readers for source-relative disconnected EMT analysis."""

import h5py
import numpy as np

from pyquda_measurement_utils.fermion_bilinear_basis import VECTOR_GAMMA_POSITIONS


def read_quark_loop(path, source_t, source_chunk_size=8):
    """Stream vector-Gamma channels and build cumulative source-relative EMT."""
    source_chunk_size = int(source_chunk_size)
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size should be positive")
    with h5py.File(path, "r") as h5:
        derivative = h5["raw/derivative_bilinear_pervec"]
        if derivative.ndim != 6 or derivative.shape[1:3] != (16, 4):
            raise ValueError(
                f"{path} has invalid derivative primitive shape {derivative.shape}"
            )
        volume_norm = h5.attrs.get("volume_norm")
        if volume_norm is None:
            raise KeyError(f"{path} is missing attrs/volume_norm")
        n_source, _, _, n_q, n_flow, n_t = derivative.shape
        n_eff_attr = int(h5.attrs.get("effective_n_inversions", n_source))
        if n_eff_attr != n_source:
            raise ValueError(
                f"{path} has effective_n_inversions={n_eff_attr}, "
                f"raw source axis={n_source}"
            )
        cumulative = np.empty(
            (n_source, 4, 4, n_q, n_flow, n_t), dtype=derivative.dtype
        )
        running = np.zeros(cumulative.shape[1:], dtype=derivative.dtype)
        for start in range(0, n_source, source_chunk_size):
            stop = min(start + source_chunk_size, n_source)
            b_tensor = np.empty(
                (stop - start, 4, 4, n_q, n_flow, n_t),
                dtype=derivative.dtype,
            )
            for nu, gamma_position in enumerate(VECTOR_GAMMA_POSITIONS):
                b_tensor[:, nu] = derivative[start:stop, gamma_position]
            tensor = 0.5 * (b_tensor + np.swapaxes(b_tensor, 1, 2))
            tensor = np.roll(tensor, -int(source_t), axis=-1)
            for local_idx, source_idx in enumerate(range(start, stop)):
                running += tensor[local_idx]
                cumulative[source_idx] = (
                    running / (source_idx + 1) / float(volume_norm)
                )
            del b_tensor, tensor

        bookkeeping = {
            name: h5[f"raw/{name}"][...]
            for name in ("source_index", "base_noise_index", "hp_index")
        }
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)

    counts = np.arange(1, n_source + 1, dtype=np.int32)
    return cumulative, counts, bookkeeping, qext, flow_times


def read_gluon_loop(path, source_t):
    """Read the ten stored symmetric components and make time source-relative."""
    with h5py.File(path, "r") as h5:
        sample = h5["Tmunu/T11"][...]
        loop = np.zeros((4, 4) + sample.shape, dtype=sample.dtype)
        for mu in range(4):
            for nu in range(mu, 4):
                data = h5[f"Tmunu/T{mu + 1}{nu + 1}"][...]
                loop[mu, nu] = data
                loop[nu, mu] = data
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)
    return np.roll(loop, -int(source_t), axis=-1), qext, flow_times


__all__ = ["read_quark_loop", "read_gluon_loop"]

