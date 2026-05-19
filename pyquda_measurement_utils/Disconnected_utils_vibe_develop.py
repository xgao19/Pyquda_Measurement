"""Shared helpers for disconnected one-point measurements."""

import numpy as np

from pyquda.field import LatticeFermion

from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


VALID_NOISE_SCHEMES = {"zn", "hierarchical_probing"}
VALID_HP_ORDERINGS = {
    "global_xyzt_gray_projected_to_evenodd",
    "spatial_xyz_then_t_gray_projected_to_evenodd",
}


def array_to_numpy(arr):
    """Convert backend arrays to NumPy arrays."""
    if hasattr(arr, "get"):
        return arr.get()
    if type(arr).__module__.split(".")[0] == "cupy":
        return arr.get()
    if type(arr).__module__.split(".")[0] == "dpnp":
        import dpnp

        return dpnp.asnumpy(arr)
    return np.asarray(arr)


def normalize_noise_scheme(noise_scheme: str) -> str:
    """Normalize and validate the stochastic noise scheme name."""
    scheme = str(noise_scheme).strip().lower()
    if scheme not in VALID_NOISE_SCHEMES:
        raise ValueError(f"noise_scheme should be one of {VALID_NOISE_SCHEMES}, got {noise_scheme!r}")
    return scheme


def is_power_of_two(value: int) -> bool:
    """Return whether value is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)) for positive integers, with ceil_log2(1)=0."""
    if value <= 1:
        return 0
    return int(value - 1).bit_length()


def validate_hierarchical_probing_options(hp_num_vectors: int, hp_ordering: str):
    """Validate common hierarchical probing options."""
    if hp_ordering not in VALID_HP_ORDERINGS:
        raise ValueError(f"hp_ordering should be one of {VALID_HP_ORDERINGS}, got {hp_ordering!r}")
    if not is_power_of_two(hp_num_vectors):
        raise ValueError(f"hp_num_vectors should be a positive power of two, got {hp_num_vectors}")


def effective_n_inversions(n_vec: int, noise_scheme: str, hp_num_vectors: int) -> int:
    """Return the number of effective stochastic inversions."""
    return n_vec * hp_num_vectors if noise_scheme == "hierarchical_probing" else n_vec


def source_bookkeeping_arrays(n_eff: int):
    """Create common source bookkeeping arrays."""
    return {
        "source_index": np.arange(n_eff, dtype=np.int32),
        "base_noise_index": np.zeros(n_eff, dtype=np.int32),
        "hp_index": np.zeros(n_eff, dtype=np.int32),
    }


def make_zn_noise_fermion(latt_info, n: int = 2) -> LatticeFermion:
    """Create one stochastic fermion source with Z_n phases."""
    xi = LatticeFermion(latt_info)
    xp = _get_xp_from_array(xi.data)
    r = xp.random.randint(0, n, size=xi.data.shape)
    xi.data[:] = xp.exp(2j * xp.pi * r / n).astype(xi.data.dtype)
    return xi


def hierarchical_gray_index(latt_info, hp_ordering: str):
    """Return the Gray-code site index for a hierarchical probing ordering."""
    coords = latt_info.coordinate()
    x, y, z, t = [np.asarray(coords[mu], dtype=np.int64) for mu in range(4)]
    Gx, Gy, Gz, _Gt = latt_info.global_size

    if hp_ordering == "global_xyzt_gray_projected_to_evenodd":
        site_id = x + Gx * (y + Gy * (z + Gz * t))
        return site_id ^ (site_id >> 1)

    if hp_ordering == "spatial_xyz_then_t_gray_projected_to_evenodd":
        spatial_id = x + Gx * (y + Gy * z)
        spatial_gray = spatial_id ^ (spatial_id >> 1)
        time_gray = t ^ (t >> 1)
        spatial_bits = ceil_log2(Gx * Gy * Gz)
        return spatial_gray | (time_gray << spatial_bits)

    raise ValueError(f"Unsupported hp_ordering {hp_ordering!r}")


def hierarchical_probe_pattern(latt_info, hp_idx: int, hp_ordering: str):
    """Build a site-only Rademacher probing vector in even-odd layout."""
    if hp_idx == 0:
        return np.ones_like(latt_info.coordinate(0), dtype=np.float64)

    gray_id = hierarchical_gray_index(latt_info, hp_ordering)
    parity = np.zeros_like(gray_id, dtype=bool)
    mask = int(hp_idx)
    bit = 1
    while bit <= mask:
        if mask & bit:
            parity ^= (gray_id & bit) != 0
        bit <<= 1
    return np.where(parity, -1.0, 1.0)


def apply_hierarchical_probe(xi: LatticeFermion, hp_idx: int, hp_ordering: str) -> LatticeFermion:
    """Multiply a base stochastic source by one hierarchical probing vector."""
    if hp_idx == 0:
        return xi.copy()

    probed = xi.copy()
    xp = _get_xp_from_array(xi.data)
    pattern = hierarchical_probe_pattern(xi.latt_info, hp_idx, hp_ordering)
    pattern = _asarray_on_queue(pattern, xp, xi.data)
    probed.data[:] *= pattern[..., None, None]
    return probed


def iter_noise_sources(latt_info, n_vec: int, n_zn: int, noise_scheme: str, hp_num_vectors: int, hp_ordering: str):
    """Yield effective stochastic sources with optional hierarchical probing."""
    if noise_scheme == "zn":
        for base_idx in range(n_vec):
            yield base_idx, base_idx, 0, make_zn_noise_fermion(latt_info, n=n_zn)
        return

    for base_idx in range(n_vec):
        base_noise = make_zn_noise_fermion(latt_info, n=n_zn)
        for hp_idx in range(hp_num_vectors):
            effective_idx = base_idx * hp_num_vectors + hp_idx
            yield effective_idx, base_idx, hp_idx, apply_hierarchical_probe(base_noise, hp_idx, hp_ordering)
