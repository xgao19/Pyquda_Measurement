"""Shared helpers for disconnected one-point measurements."""

import numpy as np

from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array


VALID_NOISE_SCHEMES = {"zn", "hierarchical_probing"}
VALID_SPIN_COLOR_DILUTIONS = {"none", "point"}
SPIN_COLOR_POINT_FACTOR = 12
COUNTER_NOISE_ALGORITHM = "splitmix64_global_coordinate_v1"
VALID_HP_ORDERINGS = {
    "global_xyzt_gray_projected_to_evenodd",
    "interleaved_xyzt_binary_projected_to_evenodd",
    "interleaved_xyz_binary_projected_to_evenodd",
    "spatial_xyz_then_t_gray_projected_to_evenodd",
}

_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
_CONFIG_SALT = np.uint64(0xD1B54A32D192ED03)
_BASE_NOISE_SALT = np.uint64(0x8CB92BA72F3D8DD7)
_STREAM_SALT = np.uint64(0xDB4F0B9175AE2165)


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


def normalize_spin_color_dilution(spin_color_dilution: str) -> str:
    """Normalize and validate the spin-color dilution mode."""
    mode = str(spin_color_dilution).strip().lower()
    if mode not in VALID_SPIN_COLOR_DILUTIONS:
        raise ValueError(
            f"spin_color_dilution should be one of {VALID_SPIN_COLOR_DILUTIONS}, "
            f"got {spin_color_dilution!r}"
        )
    return mode


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


def spin_color_dilution_factor(spin_color_dilution: str = "none") -> int:
    """Return the solve multiplier for the spin-color dilution mode."""
    mode = normalize_spin_color_dilution(spin_color_dilution)
    return SPIN_COLOR_POINT_FACTOR if mode == "point" else 1


def effective_n_inversions(n_vec: int, noise_scheme: str, hp_num_vectors: int, spin_color_dilution: str = "none") -> int:
    """Return the number of effective stochastic inversions."""
    hp_factor = hp_num_vectors if noise_scheme == "hierarchical_probing" else 1
    return n_vec * hp_factor * spin_color_dilution_factor(spin_color_dilution)


def source_bookkeeping_arrays(n_eff: int, include_spin_color: bool = False):
    """Create common source bookkeeping arrays."""
    bookkeeping = {
        "source_index": np.arange(n_eff, dtype=np.int32),
        "base_noise_index": np.zeros(n_eff, dtype=np.int32),
        "hp_index": np.zeros(n_eff, dtype=np.int32),
    }
    if include_spin_color:
        bookkeeping["spin_index"] = np.full(n_eff, -1, dtype=np.int32)
        bookkeeping["color_index"] = np.full(n_eff, -1, dtype=np.int32)
    return bookkeeping


def create_gi_qtmd_wilsonline_index_lists(eta_list, max_b_z: int, max_b_T: int):
    """Create fixed-length GI qTMD Wilson-index lists for transverse x/y."""
    index_list_trans0 = []
    index_list_trans1 = []
    for eta in eta_list:
        eta = int(eta)
        for current_bz in range(0, int(max_b_z) + 1, 2):
            if eta < current_bz // 2:
                continue
            for current_b_T in range(0, int(max_b_T) + 1):
                index_list_trans0.append([current_b_T, current_bz, eta, 0])
                index_list_trans1.append([current_b_T, current_bz, eta, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, eta, 0])
                    index_list_trans1.append([current_b_T, -current_bz, eta, 1])
    return index_list_trans0, index_list_trans1


def make_zn_noise_fermion(latt_info, n: int = 2):
    """Create one stochastic fermion source with Z_n phases."""
    from pyquda.field import LatticeFermion

    xi = LatticeFermion(latt_info)
    xp = _get_xp_from_array(xi.data)
    r = xp.random.randint(0, n, size=xi.data.shape)
    phases = xp.exp(2j * xp.pi * r / n).astype(xi.data.dtype)
    if xp.__name__ == "dpnp" and hasattr(xi.data, "sycl_queue"):
        import dpnp

        phases = dpnp.asarray(dpnp.asnumpy(phases), sycl_queue=xi.data.sycl_queue)
    xi.data[:] = phases
    return xi


def _splitmix64(values):
    """Apply the fixed SplitMix64 finalizer with unsigned wraparound."""
    z = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z + _SPLITMIX_GAMMA) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2) & _UINT64_MASK
    return z ^ (z >> np.uint64(31))


def counter_zn_phase_indices(
    latt_info,
    config_num: int,
    base_noise_index: int,
    stream_seed: int = 0,
    n: int = 4,
    spin_count: int = 4,
    color_count: int = 3,
):
    """Build decomposition-independent Z_n phase indices from global coordinates."""
    config_num = int(config_num)
    base_noise_index = int(base_noise_index)
    stream_seed = int(stream_seed)
    n = int(n)
    if config_num < 0 or base_noise_index < 0 or stream_seed < 0:
        raise ValueError("counter-noise configuration, base index, and stream seed must be non-negative")
    if n <= 0:
        raise ValueError(f"n should be positive, got {n}")

    coords = latt_info.coordinate()
    x, y, z, t = [np.asarray(coords[mu], dtype=np.uint64) for mu in range(4)]
    Gx, Gy, Gz, _ = [int(extent) for extent in latt_info.global_size]
    site_id = x + np.uint64(Gx) * (y + np.uint64(Gy) * (z + np.uint64(Gz) * t))
    config_key = _splitmix64(np.uint64(config_num) ^ _CONFIG_SALT)
    base_key = _splitmix64(np.uint64(base_noise_index) ^ _BASE_NOISE_SALT)
    stream_key = _splitmix64(np.uint64(stream_seed) ^ _STREAM_SALT)
    common_key = config_key ^ base_key ^ stream_key
    phase_dtype = np.min_scalar_type(n - 1)
    phase_indices = np.empty(site_id.shape + (spin_count, color_count), dtype=phase_dtype)
    for spin_idx in range(spin_count):
        for color_idx in range(color_count):
            spin_color_idx = spin_idx * color_count + color_idx
            counter = site_id * np.uint64(spin_count * color_count) + np.uint64(spin_color_idx)
            hashed = _splitmix64(counter ^ common_key)
            phase_indices[..., spin_idx, color_idx] = hashed % np.uint64(n)
    return phase_indices


def make_counter_zn_noise_fermion(
    latt_info,
    config_num: int,
    base_noise_index: int,
    stream_seed: int = 0,
    n: int = 4,
):
    """Create full-volume counter-based Z_n noise invariant under MPI decomposition."""
    from pyquda.field import LatticeFermion

    xi = LatticeFermion(latt_info)
    xp = _get_xp_from_array(xi.data)
    phase_indices = counter_zn_phase_indices(
        latt_info,
        config_num,
        base_noise_index,
        stream_seed=stream_seed,
        n=n,
        spin_count=xi.data.shape[-2],
        color_count=xi.data.shape[-1],
    )
    if int(n) == 4:
        phase_table = np.asarray([1.0, 1.0j, -1.0, -1.0j], dtype=np.dtype(xi.data.dtype))
    elif int(n) == 2:
        phase_table = np.asarray([1.0, -1.0], dtype=np.dtype(xi.data.dtype))
    else:
        phase_table = np.exp(2j * np.pi * np.arange(int(n)) / int(n)).astype(np.dtype(xi.data.dtype))
    phases = phase_table[phase_indices]
    xi.data[:] = _asarray_on_queue(phases, xp, xi.data)
    return xi


def make_site_zn_noise_fermion(latt_info, n: int = 2):
    """Create one stochastic fermion source with site-only Z_n phases."""
    from pyquda.field import LatticeFermion

    xi = LatticeFermion(latt_info)
    xp = _get_xp_from_array(xi.data)
    r = xp.random.randint(0, n, size=xi.data.shape[:-2])
    phases = xp.exp(2j * xp.pi * r / n).astype(xi.data.dtype)
    if xp.__name__ == "dpnp" and hasattr(xi.data, "sycl_queue"):
        import dpnp

        phases = dpnp.asarray(dpnp.asnumpy(phases), sycl_queue=xi.data.sycl_queue)
    xi.data[:] = phases[..., None, None]
    return xi


def apply_spin_color_point_dilution(xi, spin_idx: int, color_idx: int):
    """Keep only one spin-color channel of a site-noise source."""
    diluted = xi.copy()
    if not (0 <= int(spin_idx) < diluted.data.shape[-2]):
        raise ValueError(f"spin_idx {spin_idx} outside spin dimension {diluted.data.shape[-2]}")
    if not (0 <= int(color_idx) < diluted.data.shape[-1]):
        raise ValueError(f"color_idx {color_idx} outside color dimension {diluted.data.shape[-1]}")
    diluted.data[:] = 0
    diluted.data[..., int(spin_idx), int(color_idx)] = xi.data[..., int(spin_idx), int(color_idx)]
    return diluted


def hierarchical_gray_index(latt_info, hp_ordering: str):
    """Return the Gray-code site index for a hierarchical probing ordering."""
    coords = latt_info.coordinate()
    x, y, z, t = [np.asarray(coords[mu], dtype=np.int64) for mu in range(4)]
    Gx, Gy, Gz, Gt = latt_info.global_size

    if hp_ordering == "global_xyzt_gray_projected_to_evenodd":
        site_id = x + Gx * (y + Gy * (z + Gz * t))
        return site_id ^ (site_id >> 1)

    if hp_ordering == "interleaved_xyzt_binary_projected_to_evenodd":
        site_id = np.zeros_like(x, dtype=np.int64)
        out_bit = 0
        for bit in range(max(ceil_log2(Gx), ceil_log2(Gy), ceil_log2(Gz), ceil_log2(Gt))):
            for coord, extent in ((x, Gx), (y, Gy), (z, Gz), (t, Gt)):
                if bit < ceil_log2(extent):
                    site_id |= ((coord >> bit) & 1) << out_bit
                    out_bit += 1
        return site_id

    if hp_ordering == "interleaved_xyz_binary_projected_to_evenodd":
        site_id = np.zeros_like(x, dtype=np.int64)
        out_bit = 0
        for bit in range(max(ceil_log2(Gx), ceil_log2(Gy), ceil_log2(Gz))):
            for coord, extent in ((x, Gx), (y, Gy), (z, Gz)):
                if bit < ceil_log2(extent):
                    site_id |= ((coord >> bit) & 1) << out_bit
                    out_bit += 1
        return site_id

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


def apply_hierarchical_probe(xi, hp_idx: int, hp_ordering: str):
    """Multiply a base stochastic source by one hierarchical probing vector."""
    if hp_idx == 0:
        return xi.copy()

    probed = xi.copy()
    xp = _get_xp_from_array(xi.data)
    pattern = hierarchical_probe_pattern(xi.latt_info, hp_idx, hp_ordering)
    pattern = _asarray_on_queue(pattern, xp, xi.data)
    probed.data[:] *= pattern[..., None, None]
    return probed


def _seed_backend_random(latt_info, seed: int):
    """Seed the active array backend random generator for this lattice."""
    from pyquda.field import LatticeFermion

    probe = LatticeFermion(latt_info)
    xp = _get_xp_from_array(probe.data)
    xp.random.seed(int(seed))
    del probe


def iter_noise_sources(
    latt_info,
    n_vec: int,
    n_zn: int,
    noise_scheme: str,
    hp_num_vectors: int,
    hp_ordering: str,
    spin_color_dilution: str = "none",
    include_spin_color: bool = False,
    skip_base_indices=None,
    base_seed_start=None,
    base_seed_fn=None,
    counter_noise_config=None,
    counter_noise_stream: int = 0,
):
    """Yield effective stochastic sources with optional hierarchical probing."""
    spin_color_dilution = normalize_spin_color_dilution(spin_color_dilution)
    skip_base_indices = set() if skip_base_indices is None else {int(idx) for idx in skip_base_indices}

    def prepare_base_noise(base_idx):
        if int(base_idx) in skip_base_indices:
            return None
        if counter_noise_config is not None:
            if spin_color_dilution == "point":
                raise ValueError("counter-based noise is not implemented for site-only spin-color dilution")
            return make_counter_zn_noise_fermion(
                latt_info,
                int(counter_noise_config),
                int(base_idx),
                stream_seed=int(counter_noise_stream),
                n=n_zn,
            )
        if base_seed_fn is not None:
            _seed_backend_random(latt_info, int(base_seed_fn(base_idx)))
        elif base_seed_start is not None:
            _seed_backend_random(latt_info, int(base_seed_start) + int(base_idx))
        if spin_color_dilution == "point":
            return make_site_zn_noise_fermion(latt_info, n=n_zn)
        return make_zn_noise_fermion(latt_info, n=n_zn)

    def make_output(effective_idx, base_idx, hp_idx, spin_idx, color_idx, source):
        if include_spin_color:
            return effective_idx, base_idx, hp_idx, spin_idx, color_idx, source
        return effective_idx, base_idx, hp_idx, source

    def iter_spin_color_sources(effective_base_idx, base_idx, hp_idx, source):
        if spin_color_dilution == "none":
            yield make_output(effective_base_idx, base_idx, hp_idx, -1, -1, source)
            return
        for spin_idx in range(source.data.shape[-2]):
            for color_idx in range(source.data.shape[-1]):
                effective_idx = effective_base_idx * SPIN_COLOR_POINT_FACTOR + spin_idx * source.data.shape[-1] + color_idx
                yield make_output(
                    effective_idx,
                    base_idx,
                    hp_idx,
                    spin_idx,
                    color_idx,
                    apply_spin_color_point_dilution(source, spin_idx, color_idx),
                )

    if noise_scheme == "zn":
        for base_idx in range(n_vec):
            source = prepare_base_noise(base_idx)
            if source is None:
                continue
            yield from iter_spin_color_sources(base_idx, base_idx, 0, source)
        return

    for base_idx in range(n_vec):
        base_noise = prepare_base_noise(base_idx)
        if base_noise is None:
            continue
        for hp_idx in range(hp_num_vectors):
            effective_base_idx = base_idx * hp_num_vectors + hp_idx
            yield from iter_spin_color_sources(
                effective_base_idx,
                base_idx,
                hp_idx,
                apply_hierarchical_probe(base_noise, hp_idx, hp_ordering),
            )
