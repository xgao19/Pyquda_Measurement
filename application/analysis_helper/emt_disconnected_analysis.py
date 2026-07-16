"""Memory-bounded readers for source-relative disconnected EMT analysis."""

import h5py
import numpy as np

from pyquda_measurement_utils.fermion_bilinear_basis import VECTOR_GAMMA_POSITIONS


LOOP_PROVENANCE_SCHEMA = "emt_disconnected_loop_provenance_v1"
SPATIAL_PHASE_CONVENTION = "exp(-2pi*i*sum_j q_j*(x_j-origin_j)/L_j)"
ABSOLUTE_TIME_CONVENTION = "absolute_lattice_time"


def _text_attr(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_loop_provenance(h5, path):
    """Return ``(lattice_size, origin)`` or reject an old/inconsistent loop."""
    required = (
        "loop_provenance_schema",
        "global_lattice_size",
        "momentum_phase_origin",
        "spatial_momentum_phase_convention",
        "loop_time_convention",
    )
    missing = [name for name in required if name not in h5.attrs]
    if missing:
        raise KeyError(f"{path} is missing strict loop provenance attrs {missing}")
    if _text_attr(h5.attrs["loop_provenance_schema"]) != LOOP_PROVENANCE_SCHEMA:
        raise ValueError(f"{path} has an unsupported loop provenance schema")
    if _text_attr(h5.attrs["spatial_momentum_phase_convention"]) != SPATIAL_PHASE_CONVENTION:
        raise ValueError(f"{path} has an inconsistent spatial momentum phase")
    if _text_attr(h5.attrs["loop_time_convention"]) != ABSOLUTE_TIME_CONVENTION:
        raise ValueError(f"{path} is not stored in absolute lattice time")
    lattice_size = np.asarray(h5.attrs["global_lattice_size"], dtype=np.int64)
    origin = np.asarray(h5.attrs["momentum_phase_origin"], dtype=np.int64)
    if lattice_size.shape != (4,) or np.any(lattice_size <= 0):
        raise ValueError(f"{path} has invalid global_lattice_size={lattice_size}")
    if (
        origin.shape != (4,)
        or np.any(origin < 0)
        or np.any(origin >= lattice_size)
        or origin[3] != 0
    ):
        raise ValueError(f"{path} has invalid momentum_phase_origin={origin}")
    return lattice_size, origin


def source_relative_loops(
    loop,
    qext,
    source_positions,
    lattice_size,
    momentum_phase_origin=(0, 0, 0, 0),
    *,
    q_axis=-2,
):
    """Rephase origin-based absolute-time loops for one or more sources."""
    loop = np.asarray(loop, dtype=np.complex128)
    qext = np.asarray(qext, dtype=np.int64)
    sources = np.asarray(source_positions, dtype=np.int64)
    lattice_size = np.asarray(lattice_size, dtype=np.int64)
    origin = np.asarray(momentum_phase_origin, dtype=np.int64)
    if sources.ndim == 1:
        sources = sources[None, :]
    if sources.ndim != 2 or sources.shape[1] != 4:
        raise ValueError("source_positions should have shape [Nsource,4]")
    if lattice_size.shape != (4,) or origin.shape != (4,):
        raise ValueError("lattice_size and momentum_phase_origin should have length 4")
    if qext.ndim != 2 or qext.shape[1] != 4:
        raise ValueError("qext should have shape [Nq,4]")
    if np.any(sources < 0) or np.any(sources >= lattice_size):
        raise ValueError("source_positions should lie inside the global lattice")
    q_axis = int(q_axis)
    if q_axis < 0:
        q_axis += loop.ndim
    if not 0 <= q_axis < loop.ndim:
        raise np.AxisError(q_axis, ndim=loop.ndim)
    if loop.shape[q_axis] != len(qext) or loop.shape[-1] != lattice_size[3]:
        raise ValueError("loop momentum/time axes do not match qext/global lattice")
    spatial_offsets = (sources[:, :3] - origin[:3]) / lattice_size[:3]
    phases = np.exp(-2j * np.pi * np.einsum("qi,si->sq", qext[:, :3], spatial_offsets))
    relative = np.stack(
        [np.roll(loop, -int(source[3]), axis=-1) for source in sources], axis=0
    )
    phase_shape = [1] * relative.ndim
    phase_shape[0] = len(sources)
    phase_shape[q_axis + 1] = len(qext)
    return relative * phases.reshape(phase_shape)


def read_quark_loop(path, source_position, source_chunk_size=8):
    """Stream vector-Gamma channels and build cumulative source-relative EMT."""
    source_chunk_size = int(source_chunk_size)
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size should be positive")
    with h5py.File(path, "r") as h5:
        lattice_size, phase_origin = validate_loop_provenance(h5, path)
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        derivative = h5["raw/derivative_bilinear_pervec"]
        if derivative.ndim != 6 or derivative.shape[1:3] != (16, 4):
            raise ValueError(
                f"{path} has invalid derivative primitive shape {derivative.shape}"
            )
        volume_norm = h5.attrs.get("volume_norm")
        if volume_norm is None:
            raise KeyError(f"{path} is missing attrs/volume_norm")
        n_source, _, _, n_q, n_flow, n_t = derivative.shape
        if n_t != lattice_size[3] or n_q != len(qext):
            raise ValueError(f"{path} loop axes do not match provenance")
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
            tensor = source_relative_loops(
                tensor,
                qext,
                source_position,
                lattice_size,
                phase_origin,
                q_axis=3,
            )[0]
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
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)

    counts = np.arange(1, n_source + 1, dtype=np.int32)
    return cumulative, counts, bookkeeping, qext, flow_times


def read_gluon_loop(path, source_position):
    """Read the ten stored symmetric components and make time source-relative."""
    with h5py.File(path, "r") as h5:
        lattice_size, phase_origin = validate_loop_provenance(h5, path)
        sample = h5["Tmunu/T11"][...]
        loop = np.zeros((4, 4) + sample.shape, dtype=sample.dtype)
        for mu in range(4):
            for nu in range(mu, 4):
                data = h5[f"Tmunu/T{mu + 1}{nu + 1}"][...]
                loop[mu, nu] = data
                loop[nu, mu] = data
        qext = np.asarray(h5.attrs["qext"], dtype=np.int32)
        flow_times = np.asarray(h5.attrs["flow_times"], dtype=np.float64)
    relative = source_relative_loops(
        loop, qext, source_position, lattice_size, phase_origin, q_axis=2
    )[0]
    return relative, qext, flow_times


__all__ = [
    "ABSOLUTE_TIME_CONVENTION",
    "LOOP_PROVENANCE_SCHEMA",
    "SPATIAL_PHASE_CONVENTION",
    "read_gluon_loop",
    "read_quark_loop",
    "source_relative_loops",
    "validate_loop_provenance",
]
