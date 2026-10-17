"""Connected nucleon CG/PDF contractions with exact Cartesian projections.

Sequential-source construction is intentionally outside this module. Ordinary
CG shifts never replace gauge-covariant GI transport. All output times and
Wilson-index ordering are preserved; only rank zero receives complete results.
"""

import numpy as np

from pyquda import getGridCoord, getGridSize, getMPIComm
from pyquda_comm import getRankFromCoord

from pyquda_measurement_utils.pion_utils_vibe_develop import zeros_on_backend
from pyquda_measurement_utils.qtmd_operator_utils import shift_propagator_pdf_gi
from pyquda_measurement_utils.tools import (
    _asarray_on_queue, _get_xp_from_array, _reduce_spatial_projected,
    array_to_numpy, mpi_print,
)


def _shift_one(field, out, direction, sign, faces):
    """Write an ordinary signed nearest-neighbor shift into distinct storage."""
    xp = _get_xp_from_array(field)
    axis = 3 - direction
    dst, src = [slice(None)] * field.ndim, [slice(None)] * field.ndim
    dst[axis] = slice(None, -1) if sign > 0 else slice(1, None)
    src[axis] = slice(1, None) if sign > 0 else slice(None, -1)
    out[tuple(dst)] = field[tuple(src)]
    send, receive = [slice(None)] * field.ndim, [slice(None)] * field.ndim
    send[axis], receive[axis] = (0, -1) if sign > 0 else (-1, 0)
    grid = getGridSize()
    if grid[direction] == 1:
        out[tuple(receive)] = field[tuple(send)]
        return
    if direction not in faces:
        shape = field.shape[:axis] + field.shape[axis + 1:]
        faces[direction] = (zeros_on_backend(shape, field.dtype, xp, field),
                            np.empty(shape, dtype=field.dtype))
    packed_face, recvbuf = faces[direction]
    packed_face[:] = field[tuple(send)]
    sendbuf = np.ascontiguousarray(array_to_numpy(packed_face))
    coord = list(getGridCoord())
    dest, source = coord.copy(), coord.copy()
    dest[direction] = (coord[direction] - sign) % grid[direction]
    source[direction] = (coord[direction] + sign) % grid[direction]
    getMPIComm().Sendrecv(sendbuf, dest=getRankFromCoord(dest),
                         recvbuf=recvbuf, source=getRankFromCoord(source))
    incoming = _asarray_on_queue(recvbuf, xp, field)
    out[tuple(receive)] = incoming


def _cg_walk(initial, indices, operator_kind="CG_qTMD"):
    """Yield temporary displaced views; consume each before advancing again."""
    xp = _get_xp_from_array(initial)
    buffers = [zeros_on_backend(initial.shape, initial.dtype, xp, initial) for _ in range(2)]
    faces = {}
    current, previous, steps = initial, None, 0
    for index in indices:
        if (previous is None or index[3] != previous[3]
                or (operator_kind == "CG_PDF" and index[1] in {0, -1})):
            current, previous = initial, [0, 0, 0, index[3]]
        for delta, direction in ((index[0] - previous[0], index[3]),
                                 (index[1] - previous[1], 2)):
            for _ in range(abs(int(delta))):
                out = buffers[steps % 2]
                _shift_one(current, out, direction, 1 if delta > 0 else -1, faces)
                current, steps = out, steps + 1
        previous = index
        yield current


def _pack_forward(lexico):
    xp = _get_xp_from_array(lexico)
    # [m,j,f,c] -> [j,c,f,m], matching the sequential [i,j,c,f] matrix.
    return xp.ascontiguousarray(lexico.transpose(0, 1, 2, 3, 5, 7, 6, 4)).reshape(
        *lexico.shape[:4], 36, 4)


def _pack_sequential(info, sequential):
    xp = _get_xp_from_array(sequential)
    lexico = info.lexico(sequential, True, xp.__name__)
    return xp.ascontiguousarray(lexico.transpose(0, 1, 2, 3, 4, 6, 5, 7, 8)).reshape(
        *lexico.shape[:5], 4, 36)


def _projection_factors(info, momenta, origin, reference):
    """Small one-dimensional positive-sign phases including global origins."""
    momenta = np.asarray(momenta, dtype=int)[:, :3]
    axes = [np.unique(momenta[:, direction]) for direction in range(3)]
    expected = np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]])
    if not np.array_equal(momenta, expected):
        raise ValueError("Connected separable projection requires Cartesian x/y/z momentum ordering")
    xp = _get_xp_from_array(reference)
    coord = getGridCoord()
    factors = []
    for direction, momenta_axis in enumerate(axes):
        positions = np.arange(info.size[direction]) + coord[direction] * info.size[direction] - origin[direction]
        host = np.exp(2j * np.pi * momenta_axis[:, None] * positions[None, :] / info.global_size[direction])
        factors.append(_asarray_on_queue(host.astype(reference.dtype), xp, reference))
    return factors, len(axes[2]) == 1 and axes[2][0] == 0


def _project_spin(spin, factors, zero_z):
    """Project [polarization,t,z,y,x,i,m] before the small Gamma transform."""
    xp = _get_xp_from_array(spin)
    px, py, pz = factors
    if zero_z:
        summed_z = spin.sum(axis=2)
        projected_x = xp.einsum("ax,ptyxim->ptyaim", px, summed_z, optimize=True)
        projected_xy = xp.einsum("by,ptyaim->pabimt", py, projected_x, optimize=True)
        return projected_xy.reshape(spin.shape[0], -1, 4, 4, spin.shape[1])
    projected_x = xp.einsum("ax,ptzyxim->ptzyaim", px, spin, optimize=True)
    projected_xy = xp.einsum("by,ptzyaim->ptzabim", py, projected_x, optimize=True)
    projected_xyz = xp.einsum("cz,ptzabim->pabcimt", pz, projected_xy, optimize=True)
    return projected_xyz.reshape(spin.shape[0], -1, 4, 4, spin.shape[1])


def contract_nucleon_qtmd(
    *, latt_info, gauge, forward_prop, sequential_down, sequential_up,
    gamma_stack_device, momentum_list, source_position, wilson_indices,
    operator_kind, block_size=16,
):
    """Return legacy [wilson,polarization,momentum,gamma,time] D/U arrays."""
    if operator_kind not in {"CG_qTMD", "CG_PDF", "GI_PDF"}:
        raise ValueError(f"Unsupported connected operator {operator_kind!r}")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    xp = _get_xp_from_array(forward_prop.data)
    sequential = [_pack_sequential(latt_info, value) for value in (sequential_down, sequential_up)]
    factors, zero_z = _projection_factors(latt_info, momentum_list, source_position, forward_prop.data)
    count, npol = len(wilson_indices), sequential_down.shape[0]
    shape = (min(block_size, count), 2, npol, len(momentum_list), 4, 4, latt_info.size[3])
    batch = zeros_on_backend(shape, forward_prop.data.dtype, xp, forward_prop.data)
    root = getMPIComm().Get_rank() == 0
    result_shape = (count, npol, len(momentum_list), len(gamma_stack_device), latt_info.global_size[3])
    outputs = [np.empty(result_shape, dtype=forward_prop.data.dtype) for _ in range(2)] if root else [None, None]
    gamma_host = array_to_numpy(gamma_stack_device) if root else None
    if operator_kind.startswith("CG"):
        initial = _pack_forward(forward_prop.lexico(False))
        walk = _cg_walk(initial, wilson_indices, operator_kind)
    else:
        shifted_prop = None
    for index, wilson_index in enumerate(wilson_indices):
        if operator_kind.startswith("CG"):
            current = next(walk)
        else:
            if index == 0 or wilson_index[1] in {0, -1}:
                shifted_prop = forward_prop.copy()
                previous = [0, 0, 0, 0]
            else:
                previous = wilson_indices[index - 1]
            shifted_prop = shift_propagator_pdf_gi(gauge, shifted_prop, wilson_index, previous)
            current = _pack_forward(shifted_prop.lexico(False))
        for flavor, packed_seq in enumerate(sequential):
            spin = xp.matmul(packed_seq, current[None])
            projected = _project_spin(spin, factors, zero_z)
            batch[index % block_size, flavor] = projected
            del spin, projected
        used = index % block_size + 1
        if used == block_size or index == count - 1:
            reduced = _reduce_spatial_projected(array_to_numpy(batch[:used]))
            if root:
                gamma_projected = np.einsum("gim,bfpqimt->bfpqgt", gamma_host, reduced, optimize=True)
                start = index - used + 1
                for flavor in range(2):
                    outputs[flavor][start:index + 1] = gamma_projected[:, flavor]
            mpi_print(latt_info, f"Contract {operator_kind} block {index + 1}/{count}")
    return tuple(outputs)
