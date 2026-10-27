"""
Pion soft-factor four-point workflow in PyQUDA.

This module implements the native GPT matrix-trace definitions in PyQUDA.
The calculation is intentionally split into two stages:

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

The SS two-point diagnostic projects each wall line separately. For forward
quark momentum ``kf`` and backward quark momentum ``kb``, define

    F(t) = sum_x exp(-i kf.x) G_fw(x; t0),
    B(t) = sum_x exp(+i kb.x) G_bw(x; t0),
    C2_SS(t) = Tr[Gamma_src gamma5 B(t)^dagger gamma5 Gamma_sink F(t)].

The SP TMDWF diagnostic retains the full canonical raw 16-Gamma sink basis:

    C_SP(g, b, t) = sum_x exp[-i (kf+kb).x]
        Tr[Gamma_src gamma5 G_bw(x+b)^dagger gamma5 Gamma_g G_fw(x)].

The positive shift reads x+b, including across MPI boundaries. SS and SP have
different spatial sums; their zero-displacement values need not agree. The raw
Y5 and T5 channels require the stored physical_from_pyquda signs for comparison
with GPT's uniform gamma_mu gamma5 convention.

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

All traces close both spin and color indices. The historical explicit einsum
implementation is not a reference for the contraction algebra.
"""

from pathlib import Path

import h5py
import numpy as np

from pyquda import getGridCoord, getGridSize, getMPIComm
from pyquda_comm import getRankFromCoord
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
    _reduce_spatial_projected,
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


def _spin_color_matrix(xp, prop):
    """Pack lexicographic spin/color indices into volume-batched 12x12 matrices."""
    return xp.transpose(prop, (0, 1, 2, 3, 4, 6, 5, 7)).reshape(-1, 12, 12)


def _soft_factor_spin_matrices(gamma_ls, gamma5, reference):
    xp = _get_xp_from_array(reference)
    eye3 = matrix_on_backend(np.eye(3, dtype=reference.dtype), reference)
    spin = xp.matmul(gamma_ls, gamma5)
    spin12 = xp.einsum("sil,ba->sialb", spin, eye3).reshape(-1, 12, 12)
    gamma5_12 = xp.einsum("mn,ba->mbna", gamma5, eye3).reshape(12, 12)
    return spin12, gamma5_12


def _unpack_soft_factor_block(xp, packed, spatial):
    block = packed.reshape(packed.shape[0], *spatial, 4, 3, 4, 3)
    return xp.transpose(block, (0, 1, 2, 3, 4, 5, 7, 6, 8))


def _prepare_soft_factor_left(left, gamma_ls, gamma5):
    """Prepare the source-side operand once per contraction, not per shift."""
    xp = _get_xp_from_array(left)
    spin12, gamma5_12 = _soft_factor_spin_matrices(gamma_ls, gamma5, left)
    factor = xp.matmul(_spin_color_matrix(xp, left)[None], spin12[:, None])
    return factor, gamma5_12, left.shape[:4]


def _soft_factor_block_from_left(prepared, right):
    factor, gamma5_12, spatial = prepared
    xp = _get_xp_from_array(right)
    right_t = xp.matmul(xp.swapaxes(_spin_color_matrix(xp, right), -1, -2), gamma5_12)
    return _unpack_soft_factor_block(xp, xp.matmul(factor, right_t[None]), spatial)


def _prepare_soft_factor_right(gamma_ls, gamma5, right):
    """Prepare the fixed sink-side operand once per sink separation."""
    xp = _get_xp_from_array(right)
    spin12, gamma5_12 = _soft_factor_spin_matrices(gamma_ls, gamma5, right)
    right_t = xp.matmul(xp.swapaxes(_spin_color_matrix(xp, right), -1, -2), gamma5_12)
    return xp.matmul(spin12[:, None], right_t[None]), right.shape[:4]


def _soft_factor_block_from_right(left, prepared):
    factor, spatial = prepared
    xp = _get_xp_from_array(left)
    packed = xp.matmul(_spin_color_matrix(xp, left)[None], factor)
    return _unpack_soft_factor_block(xp, packed, spatial)


def _shifted_lexico(prop, length, direction):
    """Convert once, then consume the lexicographic walk's temporary views."""
    yield from _shifted_lexico_array(prop.lexico(False), length, direction)


def _shifted_lexico_array(field, length, direction):
    """Yield positive shifts with reusable buffers and ordinary periodic halos.

    The input is read-only. Later yields are temporary views into two work
    buffers: callers must consume them before advancing this walk again.
    Independent walks own separate buffers, including nested bT/bz walks.
    """
    yield field
    if length == 0:
        return
    xp = _get_xp_from_array(field)
    axis = 3 - direction
    first, last = [slice(None)] * field.ndim, [slice(None)] * field.ndim
    first[axis], last[axis] = 0, -1
    first, last = tuple(first), tuple(last)
    interior_dst, interior_src = [slice(None)] * field.ndim, [slice(None)] * field.ndim
    interior_dst[axis], interior_src[axis] = slice(None, -1), slice(1, None)
    interior_dst, interior_src = tuple(interior_dst), tuple(interior_src)
    buffers = [zeros_on_backend(field.shape, field.dtype, xp, field) for _ in range(min(length, 2))]
    grid = getGridSize()
    distributed = grid[direction] != 1
    if distributed:
        coord = list(getGridCoord())
        dest, source = coord.copy(), coord.copy()
        dest[direction] = (coord[direction] - 1) % grid[direction]
        source[direction] = (coord[direction] + 1) % grid[direction]
        dest, source = getRankFromCoord(dest), getRankFromCoord(source)
        comm = getMPIComm()
        face_shape = field.shape[:axis] + field.shape[axis + 1:]
        packed_face = zeros_on_backend(face_shape, field.dtype, xp, field)
        recvbuf = np.empty(face_shape, dtype=field.dtype)
    current = field
    for displacement in range(1, length + 1):
        out = buffers[(displacement - 1) % len(buffers)]
        out[interior_dst] = current[interior_src]
        if distributed:
            packed_face[:] = current[first]
            sendbuf = np.ascontiguousarray(array_to_numpy(packed_face))
            comm.Sendrecv(sendbuf, dest=dest, recvbuf=recvbuf, source=source)
            incoming = matrix_on_backend(recvbuf, field)
            out[last] = incoming
        else:
            out[last] = current[first]
        current = out
        yield current


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
            comm = getMPIComm()
            comm.Barrier()
            error = None
            root_exception = None
            if comm.Get_rank() == 0:
                try:
                    with h5py.File(save_h5, "a") as f:
                        for key, value in attrs.items():
                            f.attrs[key] = value
                except Exception as exc:
                    root_exception = exc
                    error = f"{type(exc).__name__}: {exc}"
            error = comm.bcast(error, root=0)
            if error is not None:
                raise RuntimeError(f"Failed to write wall-propagator attributes: {error}") from root_exception

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

    def contract_wall_2pt(self, latt_info, prop_fw, prop_bw, quark_mom_fw, quark_mom_bw, pion_pair_label):
        """Trace separately projected, globally summed wall lines (GPT SS)."""
        xp = _get_xp_from_array(prop_fw.data)
        src_matrix, sink_matrix = self.pion_channel_pairs[pion_pair_label]
        src_gamma = matrix_on_backend(src_matrix, prop_fw.data)
        sink_gamma = matrix_on_backend(sink_matrix, prop_fw.data)
        gamma5 = matrix_on_backend(G5, prop_fw.data)
        forward = self.apply_phase(prop_fw, [-p for p in quark_mom_fw], 1)
        backward = self.apply_phase(prop_bw, quark_mom_bw, 1)
        local_sums = xp.stack([
            xp.einsum("tzyxijab->ijabt", prop.lexico(False), optimize=True)
            for prop in (forward, backward)
        ])
        summed = _reduce_spatial_projected(array_to_numpy(local_sums))
        if latt_info.mpi_rank != 0:
            return None
        # Reduce the open matrices before multiplying, retaining cross-rank terms.
        fw, bw = summed.transpose(0, 5, 1, 3, 2, 4).reshape(2, -1, 12, 12)
        src12, sink12, g512 = [
            np.kron(array_to_numpy(matrix), np.eye(3))
            for matrix in (src_gamma, sink_gamma, gamma5)
        ]
        bar = g512 @ bw.conj().swapaxes(-1, -2) @ g512
        return np.trace(src12 @ bar @ sink12 @ fw, axis1=-2, axis2=-1)

    def contract_tmdwf_check(self, latt_info, prop_fw, prop_bw, pion_mom, pion_pair_label):
        """Return GPT SP traces as [sink_gamma, separation, global_time]."""
        xp = _get_xp_from_array(prop_fw.data)
        src_matrix, _ = self.pion_channel_pairs[pion_pair_label]
        src_gamma = matrix_on_backend(src_matrix, prop_fw.data)
        gamma5 = matrix_on_backend(G5, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        sink_gammas = matrix_stack_on_backend(list(_raw_gamma_by_label.values()), prop_fw.data)
        prepared = _prepare_soft_factor_left(prop_fw_phase.lexico(False), src_gamma[None], gamma5)
        local_shape = (len(GAMMA_LABELS), len(self.bT_dir), self.bT_length + 1, self.bz_length + 1, latt_info.size[3])
        corr_local_collect = zeros_on_backend(local_shape, prop_fw.data.dtype, xp, prop_fw.data)
        for idir, bT_dir in enumerate(self.bT_dir):
            for bT, shifted_bw in enumerate(_shifted_lexico(prop_bw, self.bT_length, bT_dir)):
                for bz, shifted in enumerate(_shifted_lexico_array(shifted_bw, self.bz_length, 2)):
                    block = _soft_factor_block_from_left(prepared, shifted.conj())[0]
                    corr_local_collect[:, idir, bT, bz] = xp.einsum(
                        "tzyxjiaa,gij->gt", block, sink_gammas, optimize=True)
        corr_global = _reduce_spatial_projected(array_to_numpy(corr_local_collect))
        return corr_global.reshape(len(GAMMA_LABELS), -1, latt_info.global_size[3]) if latt_info.mpi_rank == 0 else None

    def contract_soft_factor(self, latt_info, prop_fw, prop_bw_src, prop_sink_bw, prop_sink_fw, pion_mom):
        xp = _get_xp_from_array(prop_fw.data)
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

        phased_sink_backward = self.apply_phase(
            prop_sink_bw,
            [-2 * pion_mom[0], -2 * pion_mom[1], -2 * pion_mom[2]],
            1,
        )
        source_left = _prepare_soft_factor_left(prop_fw.lexico(False), src_ls, gamma5)
        sink_right = _prepare_soft_factor_right(sink_ls, gamma5, prop_sink_fw.lexico(False).conj())
        local_shape = (len(pion_pair_labels), len(gamma_pair_labels), len(self.bT_dir), self.bT_length + 1, latt_info.size[3])
        corr_local_collect = zeros_on_backend(local_shape, prop_fw.data.dtype, xp, prop_fw.data)
        for idir, bT_dir in enumerate(self.bT_dir):
            source_walk = _shifted_lexico(prop_bw_src, self.bT_length, bT_dir)
            sink_walk = _shifted_lexico(phased_sink_backward, self.bT_length, bT_dir)
            for bT, (source_shift, sink_shift) in enumerate(zip(source_walk, sink_walk)):
                tmp_1 = _soft_factor_block_from_left(source_left, source_shift.conj())
                tmp_2 = _soft_factor_block_from_right(sink_shift, sink_right)
                for isrc in range(len(pion_pair_labels)):
                    color_spin = xp.einsum("tzyxjiba,tzyxklab->tjikl", tmp_1[isrc], tmp_2[isrc], optimize=True)
                    corr_by_gamma = xp.einsum("tjikl,gik,glj->gt", color_spin, gamma2_ls, gamma1_ls, optimize=True)
                    corr_local_collect[isrc, :, idir, bT] = corr_by_gamma
                    mpi_print(latt_info, f"Contract pion soft factor bT={bT} dir={bT_dir} pion_pair={pion_pair_labels[isrc]} gamma_pairs={len(gamma_pair_labels)}")
                del tmp_1, tmp_2
        corr_collect = _reduce_spatial_projected(array_to_numpy(corr_local_collect))
        if latt_info.mpi_rank == 0:
            corr_collect = np.asarray(corr_collect, dtype=np.complex128)
        return corr_collect, pion_pair_labels, gamma_pair_labels
