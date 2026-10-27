"""Independent dense-matrix definitions for the soft-factor CPU tests.

No production packing, block builders, contraction einsums, or shift walkers
are used here. Internal indices are copied explicitly into matrix rows/columns.
"""

import numpy as np


def matrices(field):
    ns, _, nc, _ = field.shape[-4:]
    result = np.empty(field.shape[:-4] + (ns * nc, ns * nc), dtype=field.dtype)
    for i in range(ns):
        for j in range(ns):
            for a in range(nc):
                for b in range(nc):
                    result[..., i * nc + a, j * nc + b] = field[..., i, j, a, b]
    return result


def field_from_matrices(packed, ns=4, nc=3):
    result = np.empty(packed.shape[:-2] + (ns, ns, nc, nc), dtype=packed.dtype)
    for i in range(ns):
        for j in range(ns):
            for a in range(nc):
                for b in range(nc):
                    result[..., i, j, a, b] = packed[..., i * nc + a, j * nc + b]
    return result


def spin(matrix):
    return np.kron(np.asarray(matrix), np.eye(3))


def adjoint(matrix):
    return matrix.conj().swapaxes(-1, -2)


def positive_shift(field, distance, direction):
    axis = 3 - direction
    return np.take(field, (np.arange(field.shape[axis]) + distance) % field.shape[axis], axis=axis)


def phased(field, momentum):
    _, nz, ny, nx = field.shape[:4]
    z, y, x = np.indices((nz, ny, nx))
    angle = momentum[0] * x / nx + momentum[1] * y / ny + momentum[2] * z / nz
    return field * np.exp(2j * np.pi * angle)[None, ..., None, None, None, None]


def trace_spatial(product):
    return np.trace(product, axis1=-2, axis2=-1).sum(axis=(1, 2, 3))


def wall_2pt(forward, backward, source, sink, gamma5, kf, kb):
    fw = matrices(phased(forward, [-p for p in kf])).sum(axis=(1, 2, 3))
    bw = matrices(phased(backward, kb)).sum(axis=(1, 2, 3))
    bar = spin(gamma5) @ adjoint(bw) @ spin(gamma5)
    return np.trace(spin(source) @ bar @ spin(sink) @ fw, axis1=-2, axis2=-1)


def tmdwf(forward, backward, source, gamma5, sinks, momentum, directions, nb, nz):
    fw = matrices(phased(forward, [-p for p in momentum]))
    output = []
    for direction in directions:
        for b in range(nb + 1):
            for z in range(nz + 1):
                shifted = positive_shift(positive_shift(backward, b, direction), z, 2)
                bar = spin(gamma5) @ adjoint(matrices(shifted)) @ spin(gamma5)
                output.append([trace_spatial(spin(source) @ bar @ spin(sink) @ fw) for sink in sinks])
    return np.stack(output, axis=1)


def four_point(measurement, fields, gamma5, raw, momentum):
    fw, bw, sink_bw, sink_fw = fields
    sink_bw = phased(sink_bw, [-2 * p for p in momentum])
    fw, sink_fw = matrices(fw), matrices(sink_fw)
    output = []
    for src, sink in measurement.pion_channel_pairs.values():
        directions = []
        for direction in measurement.bT_dir:
            separations = []
            for b in range(measurement.bT_length + 1):
                a = fw @ spin(src) @ spin(gamma5) @ adjoint(matrices(positive_shift(bw, b, direction))) @ spin(gamma5)
                c = matrices(positive_shift(sink_bw, b, direction)) @ spin(sink) @ spin(gamma5) @ adjoint(sink_fw) @ spin(gamma5)
                separations.append([
                    trace_spatial(a @ spin(raw[g2]) @ c @ spin(raw[g1]))
                    for g1, g2 in measurement.gamma_channel_pairs.values()
                ])
            directions.append(np.stack(separations, axis=1))
        output.append(np.stack(directions, axis=1))
    return np.asarray(output)
