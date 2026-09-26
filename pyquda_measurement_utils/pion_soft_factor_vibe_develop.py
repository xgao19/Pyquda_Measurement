"""
Pion soft-factor four-point workflow in PyQUDA.

This module ports the legacy GPT/PyQUDA mixed workflow
``PyQUDA_qTMD_ff_4pt_einsum.py`` to a PyQUDA-native structure.  The calculation
is intentionally split into two stages:

1. Generate and save Coulomb-gauge wall-source propagators for every source
   time slice and for every requested quark momentum.
2. Read those wall propagators back and contract the pion soft-factor
   four-point functions.

Wall-source propagators
-----------------------
For a wall source at time ``t0`` and quark momentum ``k`` the source is

    eta_k(x, t) = delta_{t,t0} exp(+i k . x).

The saved propagator is

    G_k(x; t0) = D^{-1} eta_k.

The soft-factor contraction needs both ``+k`` and ``-k`` wall propagators on all
time slices, matching the legacy convention where the backward antiquark line is
formed by gamma5 hermiticity from the ``-k`` propagator.

Two-point and TMDWF checks
-------------------------
The source-time pair ``G_fw = G_k(t0)`` and ``G_bw = G_-kb(t0)`` can be used for
the same wall-source pion diagnostics as the legacy code.  The antiquark line is

    Gbar_bw(x; t0) = gamma5 G_bw(x; t0)^dagger gamma5.

For a source interpolator ``Gamma_src`` and sink interpolator ``Gamma_sink``,

    C2(t) = sum_x Tr[
        Gamma_src Gbar_bw(x; t0) Gamma_sink G_fw(x; t0)
    ].

The TMDWF-like check additionally shifts the backward line by ``bT`` and ``bz``
before the same trace.

Soft-factor four-point contraction
----------------------------------
For each source time ``t0`` and sink separation ``tsep`` the contraction uses
four wall propagators:

    Gw                = G_kfw(t0)
    Gw_bperp_dagger   = G_-kbw(t0)
    Gw_dagger         = G_-kfw(t0 + tsep)
    Gw_bperp          = G_kbw(t0 + tsep)

The sink-side ``Gw_bperp`` is multiplied by the legacy momentum-transfer phase
``exp[-2 i P . x]`` where ``P = kfw - (-kbw)`` is the pion momentum.  The
transverse separation is applied by ordinary coordinate-gauge shifts, with no
explicit gauge link.

For each transverse displacement ``b`` the two closed spin-color blocks are

    A_b(x) = Gw(x) Gamma_src gamma5 Gw_bperp_dagger(x+b)^dagger gamma5,
    B_b(x) = Gw_bperp(x+b) Gamma_sink gamma5 Gw_dagger(x)^dagger gamma5.

The soft-factor four-point correlator is then

    C4(t; tsep, b, Gamma1, Gamma2) =
        sum_x Tr[ A_b(x) Gamma2 B_b(x) Gamma1 ].

The gamma lists and default pion interpolators are kept close to the legacy
script so that output can be compared directly before further refactoring.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import time

import h5py
import numpy as np
from pyquda import getMPIComm
from pyquda_utils import core, gamma, phase, source

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
)
from pyquda_measurement_utils.io_corr import ensure_parent_dir
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    matrix_on_backend,
    matrix_stack_on_backend,
    zeros_on_backend,
)
from pyquda_measurement_utils.tools import (
    _get_xp_from_array,
    array_to_numpy,
    mpi_print,
)


soft_factor_gammas = ["5", "I", "X", "Y", "X5", "Y5"]
_raw_gamma_by_label = {
    label: gamma.gamma(gamma_id)
    for label, gamma_id in zip(GAMMA_LABELS, PYQUDA_GAMMA_IDS)
}
soft_factor_gamma_channel_pairs = {
    label: (label, label) for label in soft_factor_gammas
}
_z5_minus_x5 = gamma.gamma(4) @ gamma.gamma(15) - gamma.gamma(1) @ gamma.gamma(15)
soft_factor_pion_channel_pairs = {
    "Z5-X5__Z5-X5": (_z5_minus_x5, _z5_minus_x5),
}
G5 = gamma.gamma(15)


def _soft_factor_gpu_profile_enabled():
    value = os.environ.get("PION_SOFT_PROFILE_GPU", "0").strip().lower()
    return value not in {"0", "false", "off", "no", ""}


class _SoftFactorGpuProfiler:
    """Collect opt-in CUDA timings and device-memory high-water marks."""

    def __init__(self, latt_info, xp):
        self.latt_info = latt_info
        self.enabled = False
        self.timings = {}
        self.peak_device_used = 0
        self.peak_pool_used = 0
        self.peak_label = "unavailable"
        self.device_total = 0
        if not _soft_factor_gpu_profile_enabled():
            return
        try:
            import cupy

            if xp is not cupy:
                mpi_print(latt_info, "SOFT_FACTOR_PROFILE disabled: backend is not CuPy")
                return
            self.cupy = cupy
            self.enabled = True
            self._sample_memory("start")
        except Exception as exc:
            self.enabled = False
            mpi_print(latt_info, f"SOFT_FACTOR_PROFILE disabled: {type(exc).__name__}: {exc}")

    def _sample_memory(self, label):
        if not self.enabled:
            return
        free, total = self.cupy.cuda.runtime.memGetInfo()
        device_used = int(total) - int(free)
        pool_used = int(self.cupy.get_default_memory_pool().used_bytes())
        self.device_total = int(total)
        if device_used > self.peak_device_used:
            self.peak_device_used = device_used
            self.peak_label = label
        self.peak_pool_used = max(self.peak_pool_used, pool_used)

    @contextmanager
    def measure(self, label):
        if not self.enabled:
            yield
            return
        start = self.cupy.cuda.Event()
        stop = self.cupy.cuda.Event()
        start.record()
        wall_start = time.perf_counter()
        try:
            yield
        finally:
            stop.record()
            stop.synchronize()
            cuda_ms = float(self.cupy.cuda.get_elapsed_time(start, stop))
            wall_ms = 1000.0 * (time.perf_counter() - wall_start)
            entry = self.timings.setdefault(label, [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += cuda_ms
            entry[2] += wall_ms
            self._sample_memory(label)

    def report(self):
        if not self.enabled:
            return
        self._sample_memory("end")
        payload = {
            "timings": self.timings,
            "peak_device_used": self.peak_device_used,
            "peak_pool_used": self.peak_pool_used,
            "peak_label": self.peak_label,
            "device_total": self.device_total,
        }
        gathered = getMPIComm().gather(payload, root=0)
        if self.latt_info.mpi_rank != 0:
            return
        labels = sorted({label for item in gathered for label in item["timings"]})
        for label in labels:
            entries = [item["timings"].get(label, [0, 0.0, 0.0]) for item in gathered]
            mpi_print(
                self.latt_info,
                f"SOFT_FACTOR_PROFILE label={label} "
                f"calls_max={max(entry[0] for entry in entries)} "
                f"cuda_ms_max={max(entry[1] for entry in entries):.3f} "
                f"wall_ms_max={max(entry[2] for entry in entries):.3f}",
            )
        peak_rank, peak = max(
            enumerate(gathered),
            key=lambda item: item[1]["peak_device_used"],
        )
        gib = float(1 << 30)
        mpi_print(
            self.latt_info,
            f"SOFT_FACTOR_MEMORY peak_rank={peak_rank} "
            f"peak_label={peak['peak_label']} "
            f"device_used_gib={peak['peak_device_used'] / gib:.3f} "
            f"pool_used_gib={max(item['peak_pool_used'] for item in gathered) / gib:.3f} "
            f"device_total_gib={peak['device_total'] / gib:.3f}",
        )


def _source_block_cache_limit_bytes():
    """Keep at most half of the free device memory, or 8 GiB off-device."""
    try:
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free) // 2
    except Exception:
        return 8 << 30


def _spin_color_matrix(xp, prop):
    """Pack (t,z,y,x,spin,spin,color,color) into (volume, 12, 12)."""
    return xp.transpose(prop, (0, 1, 2, 3, 4, 6, 5, 7)).reshape(-1, 12, 12)


def _soft_factor_spin_matrices(xp, gamma_ls, gamma5, dtype):
    eye3 = xp.eye(3, dtype=dtype)
    spin = xp.matmul(gamma_ls, gamma5)
    spin12 = xp.einsum("sil,ba->sialb", spin, eye3).reshape(gamma_ls.shape[0], 12, 12)
    gamma5_12 = xp.einsum("mn,ba->mbna", gamma5, eye3).reshape(12, 12)
    return spin12, gamma5_12


def _unpack_soft_factor_block(xp, out_m, n_src, spatial):
    out = out_m.reshape(n_src, *spatial, 4, 3, 4, 3)
    return xp.transpose(out, (0, 1, 2, 3, 4, 5, 7, 6, 8))


def prepare_soft_factor_left(left, gamma_ls, gamma5):
    """Precompute ``left @ (Gamma gamma5)`` for changing right fields."""
    xp = _get_xp_from_array(left)
    spin12, gamma5_12 = _soft_factor_spin_matrices(
        xp,
        gamma_ls,
        gamma5,
        left.dtype,
    )
    left_m = _spin_color_matrix(xp, left)
    left_factor = xp.matmul(left_m[None], spin12[:, None])
    return left_factor, gamma5_12, left.shape[:4]


def soft_factor_block_from_left(prepared_left, right):
    """Finish a block whose left field and pion matrix were precomputed."""
    left_factor, gamma5_12, spatial = prepared_left
    xp = _get_xp_from_array(right)
    right_m = _spin_color_matrix(xp, right)
    right_t = xp.swapaxes(xp.matmul(gamma5_12, right_m), -1, -2)
    out_m = xp.matmul(left_factor, right_t[None])
    return _unpack_soft_factor_block(xp, out_m, left_factor.shape[0], spatial)


def prepare_soft_factor_right(gamma_ls, gamma5, right):
    """Precompute ``(Gamma gamma5) @ right.T`` for changing left fields."""
    xp = _get_xp_from_array(right)
    spin12, gamma5_12 = _soft_factor_spin_matrices(
        xp,
        gamma_ls,
        gamma5,
        right.dtype,
    )
    right_m = _spin_color_matrix(xp, right)
    right_t = xp.swapaxes(xp.matmul(gamma5_12, right_m), -1, -2)
    right_factor = xp.matmul(spin12[:, None], right_t[None])
    return right_factor, right.shape[:4]


def soft_factor_block_from_right(left, prepared_right):
    """Finish a block whose right field and pion matrix were precomputed."""
    right_factor, spatial = prepared_right
    xp = _get_xp_from_array(left)
    left_m = _spin_color_matrix(xp, left)
    out_m = xp.matmul(left_m[None], right_factor)
    return _unpack_soft_factor_block(xp, out_m, right_factor.shape[0], spatial)


def soft_factor_block(left, gamma_ls, gamma5, right):
    """Build A or B as a batched 12x12 product.

    ``left`` and ``right`` are lexicographic propagators with shape
    ``(t,z,y,x,spin,spin,color,color)``. ``gamma_ls`` is a stack of spin
    matrices. The result matches

        einsum("tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc",
               left, gamma_ls, gamma5, right, gamma5).
    """
    return soft_factor_block_from_left(
        prepare_soft_factor_left(left, gamma_ls, gamma5),
        right,
    )


# Lattice mu = x,y,z,t maps onto lexicographic axes t,z,y,x.
_LEXICO_AXIS_OF_MU = (3, 2, 1, 0)
# Lexicographic axis t,z,y,x maps onto grid indices x,y,z,t.
_GRID_INDEX_OF_LEXICO_AXIS = (3, 2, 1, 0)


def _tmdwf_components(backward, forward, src_gamma, gamma5):
    """Site tensors whose dot product is the TMDWF trace.

    Two spin indices are traced at the site, leaving 4*3*3 components.
    A circular shift of ``backward`` is the same shift of the first tensor.
    """
    xp = _get_xp_from_array(backward)
    backward_bar = xp.einsum(
        "ij,tzyxmlca,kl->tzyxkjca", gamma5, backward.conj(), gamma5, optimize=True
    )
    left = xp.einsum("ij,tzyxjlca->tzyxilca", src_gamma, backward_bar, optimize=True)
    left_c = xp.einsum("tzyxjiab->tzyxiab", left, optimize=True)
    forward_c = xp.swapaxes(xp.einsum("tzyxilba->tzyxiba", forward, optimize=True), -1, -2)
    return left_c, forward_c


def _tmdwf_fft_axes(bT_dir, bz_length):
    axes = [_LEXICO_AXIS_OF_MU[int(bT_dir)]]
    if bz_length and 1 not in axes:
        axes.append(1)
    return axes


def _tmdwf_plane_from_components(left, forward, bT_dir, bT_length, bz_length):
    """All separations from one circular correlation.

    ``left`` and ``forward`` are component fields on a volume that is periodic
    and complete along the shifted axes. Positive separations match ``np.roll``.
    """
    xp = _get_xp_from_array(left)
    b_axis = _LEXICO_AXIS_OF_MU[int(bT_dir)]
    z_axis = 1
    axes = tuple(_tmdwf_fft_axes(bT_dir, bz_length))
    spectrum = xp.fft.fftn(left, axes=axes) * xp.conj(
        xp.fft.fftn(xp.conj(forward), axes=axes)
    )
    correlated = xp.fft.ifftn(spectrum, axes=axes)
    spatial = sorted(axes)
    drop = tuple(axis for axis in range(1, correlated.ndim) if axis not in spatial)
    plane = correlated.sum(axis=drop)
    n_b = bT_length + 1
    n_z = bz_length + 1
    out = xp.empty((n_b, n_z, plane.shape[0]), dtype=plane.dtype)
    axis_pos = {axis: 1 + i for i, axis in enumerate(spatial)}
    for bT in range(n_b):
        for bz in range(n_z):
            index = [slice(None)]
            for axis in spatial:
                shift = 0
                if axis == b_axis:
                    shift += bT
                if axis == z_axis:
                    shift += bz
                index.append((-shift) % plane.shape[axis_pos[axis]])
            out[bT, bz] = plane[tuple(index)]
    return out


def tmdwf_separation_plane(backward, forward, src_gamma, gamma5, bT_dir, bT_length, bz_length):
    """Correlate one backward/forward pair over ``bT`` and ``bz``."""
    left, forward_c = _tmdwf_components(backward, forward, src_gamma, gamma5)
    return _tmdwf_plane_from_components(left, forward_c, bT_dir, bT_length, bz_length)


def _stitch_lexico(blocks, coords, grid, gather_dims):
    """Place local lexicographic blocks into the gathered axes."""
    local = blocks[0].shape
    out_size = []
    for axis in range(4):
        grid_index = _GRID_INDEX_OF_LEXICO_AXIS[axis]
        factor = grid[grid_index] if grid_index in gather_dims else 1
        out_size.append(local[axis] * factor)
    out = np.empty((*out_size, *local[4:]), dtype=blocks[0].dtype)
    for block, coord in zip(blocks, coords):
        slices = []
        for axis in range(4):
            grid_index = _GRID_INDEX_OF_LEXICO_AXIS[axis]
            if grid_index in gather_dims:
                start = coord[grid_index] * local[axis]
                slices.append(slice(start, start + local[axis]))
            else:
                slices.append(slice(None))
        out[tuple(slices)] = block
    return out


def rank_keeps_gathered_plane(coord, grid, gather_dims):
    """One rank per gathered subvolume may enter the final spatial sum.

    After an allgather, every rank in that subcommunicator holds the same
    plane. ``gatherLattice`` then sums over the whole grid, so the other
    copies must contribute zero or the correlator grows by the gathered
    grid volume.
    """
    replicated = [dim for dim in gather_dims if int(grid[dim]) > 1]
    if not replicated:
        return True
    return all(int(coord[dim]) == 0 for dim in replicated)


def _local_rank_keeps_plane(axes):
    from pyquda_comm import getGridCoord, getGridSize

    gather_dims = {_GRID_INDEX_OF_LEXICO_AXIS[axis] for axis in axes}
    return rank_keeps_gathered_plane(getGridCoord(), getGridSize(), gather_dims)


def _gather_subcomm_color_key(coord, grid, gather_dims):
    """Return the subgroup color and gathered-coordinate key for one rank."""
    color = 0
    color_stride = 1
    key = 0
    key_stride = 1
    for dim in range(4):
        if dim in gather_dims:
            key += int(coord[dim]) * key_stride
            key_stride *= int(grid[dim])
        else:
            color += int(coord[dim]) * color_stride
            color_stride *= int(grid[dim])
    return color, key


def _gather_lexico_axes(field, axes):
    """Gather shifted axes onto one rank of each unaffected-coordinate group."""
    from pyquda_comm import getGridCoord, getGridSize

    grid = tuple(int(value) for value in getGridSize())
    coord = tuple(int(value) for value in getGridCoord())
    gather_dims = {_GRID_INDEX_OF_LEXICO_AXIS[axis] for axis in axes}
    if all(grid[dim] == 1 for dim in gather_dims):
        return field
    host = np.ascontiguousarray(array_to_numpy(field))
    color, key = _gather_subcomm_color_key(coord, grid, gather_dims)
    comm = getMPIComm()
    sub = comm.Split(int(color), int(key))
    try:
        blocks = sub.gather(host, root=0)
        coords = sub.gather(coord, root=0)
    finally:
        sub.Free()
    if key != 0:
        return None
    stitched = _stitch_lexico(blocks, coords, grid, gather_dims)
    xp = _get_xp_from_array(field)
    return xp.asarray(stitched)


def momentum_tag(momentum):
    return "qx" + str(momentum[0]) + "qy" + str(momentum[1]) + "qz" + str(momentum[2])


def as_momentum_3(momentum):
    if len(momentum) == 4:
        return [int(momentum[0]), int(momentum[1]), int(momentum[2])]
    return [int(momentum[0]), int(momentum[1]), int(momentum[2])]


class pion_soft_factor:
    def __init__(self, parameters):
        self.quark_mom = [as_momentum_3(mom) for mom in parameters["quark_mom"]]
        self.bT_dir = parameters["bT_dir"]
        self.bT_length = parameters["bT_length"]
        self.bz_length = parameters.get("bz_length", 0)
        self.tsep_list = parameters["tsep_list"]
        self.pion_channel_pairs = parameters.get(
            "pion_channel_pairs", soft_factor_pion_channel_pairs
        )
        self.gamma_channel_pairs = parameters.get(
            "gamma_channel_pairs", soft_factor_gamma_channel_pairs
        )
        if not self.pion_channel_pairs or not self.gamma_channel_pairs:
            raise ValueError("soft-factor channel-pair mappings must not be empty")
        for pair_label, gamma_labels in self.gamma_channel_pairs.items():
            if len(gamma_labels) != 2 or any(label not in _raw_gamma_by_label for label in gamma_labels):
                raise ValueError(
                    f"Invalid Gamma pair {pair_label!r}: expected two canonical raw labels"
                )

    def create_wall_src(self, latt_info, tslice, momentum):
        source_phase = phase.MomentumPhase(latt_info).getPhase(as_momentum_3(momentum))
        return source.propagator(latt_info, "wall", int(tslice), source_phase)

    def create_wall_propagator(self, dirac, latt_info, tslice, momentum):
        wall_src = self.create_wall_src(latt_info, tslice, momentum)
        return core.invertPropagator(dirac, wall_src, 1, 0)

    def save_wall_propagator(self, prop, tag, attrs=None):
        save_h5 = tag + ".h5"
        ensure_parent_dir(save_h5)
        prop.saveH5(save_h5, "propagator")
        if attrs:
            # saveH5 is collective MPI-IO. Attribute create/write on that
            # handle must also be collective, so wait until it has closed
            # and let only rank 0 open the file serially.
            comm = getMPIComm()
            comm.Barrier()
            attr_error = None
            root_exception = None
            if comm.Get_rank() == 0:
                try:
                    with h5py.File(save_h5, "a") as f:
                        for key, value in attrs.items():
                            f.attrs[key] = value
                except Exception as exc:
                    root_exception = exc
                    attr_error = (type(exc).__name__, str(exc))
            attr_error = comm.bcast(attr_error, root=0)
            if attr_error is not None:
                error = RuntimeError(
                    f"Failed to write wall-propagator attributes: "
                    f"{attr_error[0]}: {attr_error[1]}"
                )
                if root_exception is not None:
                    raise error from root_exception
                raise error

    def load_wall_propagator(self, tag):
        return core.LatticePropagator.loadH5(tag + ".h5", "propagator")

    def apply_phase(self, prop, momentum, sign=1, x0=None):
        x0 = [0, 0, 0, 0] if x0 is None else x0
        xp = _get_xp_from_array(prop.data)
        mom_phase = phase.MomentumPhase(prop.latt_info).getPhase(as_momentum_3(momentum), x0=x0)
        if sign == -1:
            mom_phase = mom_phase.conj()
        mom_phase = matrix_on_backend(mom_phase, prop.data)
        phased = prop.copy()
        phased.data[:] = phased.data * mom_phase[:, :, :, :, :, None, None, None, None]
        return phased

    def contract_wall_2pt(self, latt_info, prop_fw, prop_bw, pion_mom, pion_pair_label):
        xp = _get_xp_from_array(prop_fw.data)
        src_matrix, sink_matrix = self.pion_channel_pairs[pion_pair_label]
        src_gamma = matrix_on_backend(src_matrix, prop_fw.data)
        sink_gamma = matrix_on_backend(sink_matrix, prop_fw.data)
        gamma5 = matrix_on_backend(G5, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        prop_fw_t = prop_fw_phase.lexico(False)
        prop_bw_bar = xp.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, prop_bw.lexico(False).conj(), gamma5, optimize=True)
        prop_bw_src_sink = xp.einsum("ik,tzyxklca,ln->tzyxinca", src_gamma, prop_bw_bar, sink_gamma, optimize=True)
        corr_local = xp.einsum("tzyxjiab,tzyxilba->tzyx", prop_bw_src_sink, prop_fw_t, optimize=True)
        corr_t = xp.einsum("tzyx->t", corr_local, optimize=True)
        return core.gatherLattice(array_to_numpy(corr_t), [0, -1, -1, -1])

    def contract_tmdwf_check(self, latt_info, prop_fw, prop_bw, pion_mom, pion_pair_label):
        src_matrix, _ = self.pion_channel_pairs[pion_pair_label]
        src_gamma = matrix_on_backend(src_matrix, prop_fw.data)
        gamma5 = matrix_on_backend(G5, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        backward = prop_bw.lexico(False)
        forward = prop_fw_phase.lexico(False)
        left, forward_c = _tmdwf_components(backward, forward, src_gamma, gamma5)
        corr_list = []
        for bT_dir in self.bT_dir:
            axes = _tmdwf_fft_axes(bT_dir, self.bz_length)
            keep_plane = _local_rank_keeps_plane(axes)
            gathered_left = _gather_lexico_axes(left, axes)
            gathered_forward = _gather_lexico_axes(forward_c, axes)
            plane = (
                _tmdwf_plane_from_components(
                    gathered_left,
                    gathered_forward,
                    bT_dir,
                    self.bT_length,
                    self.bz_length,
                )
                if keep_plane
                else None
            )
            for bT in range(self.bT_length + 1):
                for bz in range(self.bz_length + 1):
                    corr_t = (
                        np.asarray(array_to_numpy(plane[bT, bz]), dtype=np.complex128)
                        if keep_plane
                        else np.zeros(latt_info.size[3], dtype=np.complex128)
                    )
                    corr_list.append(core.gatherLattice(corr_t, [0, -1, -1, -1]))
        return np.asarray(corr_list)

    def _cached_source_block(self, prop_fw, prop_bw_src, bT_dir, bT, build):
        """Reuse the source-side block across sink separations.

        ``tmp_1`` depends only on the two source propagators and ``b``.
        A new source pair drops the previous cache. If the next block would
        exceed the memory limit, it is computed and not stored.
        """
        cache_id = (id(prop_fw), id(prop_bw_src), tuple(self.pion_channel_pairs))
        if getattr(self, "_source_block_cache_id", None) != cache_id:
            self._source_block_cache = {}
            self._source_block_cache_bytes = 0
            self._source_block_cache_id = cache_id
            self._source_block_hits = 0
            self._source_block_misses = 0
        key = (int(bT_dir), int(bT))
        cached = self._source_block_cache.get(key)
        if cached is not None:
            self._source_block_hits += 1
            return cached
        value = build()
        nbytes = int(getattr(value, "nbytes", 0))
        limit = _source_block_cache_limit_bytes()
        if self._source_block_cache_bytes + nbytes <= limit:
            self._source_block_cache[key] = value
            self._source_block_cache_bytes += nbytes
        self._source_block_misses += 1
        return value

    def _source_block_cache_complete(self, prop_fw, prop_bw_src, bT_dir):
        """Return whether every requested source block is already resident."""
        cache_id = (id(prop_fw), id(prop_bw_src), tuple(self.pion_channel_pairs))
        if getattr(self, "_source_block_cache_id", None) != cache_id:
            self._source_block_cache = {}
            self._source_block_cache_bytes = 0
            self._source_block_cache_id = cache_id
            self._source_block_hits = 0
            self._source_block_misses = 0
        return all(
            (int(bT_dir), bT) in self._source_block_cache
            for bT in range(self.bT_length + 1)
        )

    def contract_soft_factor(self, latt_info, prop_fw, prop_bw_src, prop_sink_bw, prop_sink_fw, pion_mom):
        xp = _get_xp_from_array(prop_fw.data)
        profiler = _SoftFactorGpuProfiler(latt_info, xp)
        gamma5 = matrix_on_backend(G5, prop_fw.data)
        pion_pair_labels = list(self.pion_channel_pairs)
        gamma_pair_labels = list(self.gamma_channel_pairs)
        pion_src_matrices = {
            label: matrices[0] for label, matrices in self.pion_channel_pairs.items()
        }
        pion_sink_matrices = {
            label: matrices[1] for label, matrices in self.pion_channel_pairs.items()
        }
        gamma1_matrices = {
            pair_label: _raw_gamma_by_label[labels[0]]
            for pair_label, labels in self.gamma_channel_pairs.items()
        }
        gamma2_matrices = {
            pair_label: _raw_gamma_by_label[labels[1]]
            for pair_label, labels in self.gamma_channel_pairs.items()
        }
        with profiler.measure("operator_setup"):
            src_ls = matrix_stack_on_backend(
                [pion_src_matrices[key] for key in pion_pair_labels], prop_fw.data
            )
            sink_ls = matrix_stack_on_backend(
                [pion_sink_matrices[key] for key in pion_pair_labels], prop_fw.data
            )
            gamma1_ls = matrix_stack_on_backend(
                [gamma1_matrices[key] for key in gamma_pair_labels], prop_fw.data
            )
            gamma2_ls = matrix_stack_on_backend(
                [gamma2_matrices[key] for key in gamma_pair_labels], prop_fw.data
            )

        with profiler.measure("prepare_sink_fixed"):
            Gw_dagger = prop_sink_fw.lexico(False)
            phased_sink_backward = self.apply_phase(
                prop_sink_bw,
                [-2 * pion_mom[0], -2 * pion_mom[1], -2 * pion_mom[2]],
                1,
            )
            Gw_dagger_conj = Gw_dagger.conj()
            sink_right = prepare_soft_factor_right(sink_ls, gamma5, Gw_dagger_conj)
        source_left = None

        local_shape = (
            len(pion_pair_labels),
            len(gamma_pair_labels),
            len(self.bT_dir),
            self.bT_length + 1,
            latt_info.size[3],
        )
        corr_local_collect = zeros_on_backend(
            local_shape,
            prop_fw.data.dtype,
            xp,
            prop_fw.data,
        )
        for idir, bT_dir in enumerate(self.bT_dir):
            shifted_sink_backward = phased_sink_backward
            source_cache_complete = self._source_block_cache_complete(
                prop_fw,
                prop_bw_src,
                bT_dir,
            )
            if not source_cache_complete and source_left is None:
                with profiler.measure("prepare_source_fixed"):
                    source_left = prepare_soft_factor_left(
                        prop_fw.lexico(False),
                        src_ls,
                        gamma5,
                    )
            shifted_source_backward = None if source_cache_complete else prop_bw_src
            for bT in range(self.bT_length + 1):
                if bT != 0:
                    with profiler.measure("sink_shift"):
                        shifted_sink_backward = shifted_sink_backward.shift(1, bT_dir)
                    if not source_cache_complete:
                        with profiler.measure("source_shift"):
                            shifted_source_backward = shifted_source_backward.shift(1, bT_dir)
                with profiler.measure("sink_lexico"):
                    Gw_bperp_shift = shifted_sink_backward.lexico(False)
                with profiler.measure("source_block"):
                    tmp_1 = self._cached_source_block(
                        prop_fw,
                        prop_bw_src,
                        bT_dir,
                        bT,
                        lambda shifted=shifted_source_backward: soft_factor_block_from_left(
                            source_left,
                            shifted.lexico(False).conj(),
                        ),
                    )
                with profiler.measure("sink_block"):
                    tmp_2 = soft_factor_block_from_right(Gw_bperp_shift, sink_right)
                for isrc in range(len(pion_pair_labels)):
                    # One spatial reduction serves every Gamma pair:
                    # M[t,j,i,k,l] = sum_{zyx,ba} A[tzyxjiba] B[tzyxklba].
                    with profiler.measure("spatial_reduce"):
                        color_spin = xp.einsum(
                            "tzyxjiba,tzyxklba->tjikl",
                            tmp_1[isrc],
                            tmp_2[isrc],
                            optimize=True,
                        )
                    with profiler.measure("gamma_contract"):
                        corr_by_gamma = xp.einsum(
                            "tjikl,gik,glj->gt",
                            color_spin,
                            gamma2_ls,
                            gamma1_ls,
                            optimize=True,
                        )
                    corr_local_collect[isrc, :, idir, bT] = corr_by_gamma
                    mpi_print(
                        latt_info,
                        f"Contract pion soft factor bT={bT} dir={bT_dir} "
                        f"pion_pair={pion_pair_labels[isrc]} gamma_pairs={len(gamma_pair_labels)}",
                    )
                del (
                    Gw_bperp_shift,
                    tmp_1,
                    tmp_2,
                )
        with profiler.measure("result_publication"):
            corr_collect = core.gatherLattice(
                array_to_numpy(corr_local_collect),
                [4, -1, -1, -1],
            )
        if latt_info.mpi_rank == 0:
            corr_collect = np.asarray(corr_collect, dtype=np.complex128)
        profiler.report()
        return corr_collect, pion_pair_labels, gamma_pair_labels
