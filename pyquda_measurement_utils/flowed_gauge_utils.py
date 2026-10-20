"""Gauge-only clover and flow primitives shared by gluonic observables."""

from opt_einsum import contract
from pyquda_comm.array import arrayIdentity


def advance_gauge_flow(gauge, flow_type, n_steps, epsilon):
    """Advance an existing field; scheduling belongs to the measurement."""
    if flow_type == "wilson":
        gauge.wilsonFlow(n_steps, epsilon=epsilon)
    elif flow_type == "symanzik":
        gauge.symanzikFlow(n_steps, epsilon=epsilon)
    else:
        raise ValueError("flow_type must be wilson or symanzik")


def clover_traceless(gauge, mu, nu):
    """Return H_mu,nu = -i (C-C^dagger)_traceless/8 (Hermitian)."""
    loops = [
        [mu, nu, mu + 4, nu + 4],
        [nu, mu + 4, nu + 4, mu],
        [mu + 4, nu + 4, mu, nu],
        [nu + 4, mu, nu, mu + 4],
    ]
    field = gauge.loop([loops] * 4, coeff=[1.0] * 4)
    data = field.data
    anti = 0.125 * (data - data.swapaxes(-2, -1).conjugate())
    nc = anti.shape[-1]
    trace = contract("...ii->...", anti)
    identity = arrayIdentity(nc, anti.dtype, field.location)
    anti -= trace[..., None, None] * identity / nc
    data[...] = (-1j) * anti
    return field


def all_clover_traceless(gauge):
    """Build the antisymmetric Hermitian field-strength table."""
    fields = [[None] * 4 for _ in range(4)]
    for mu in range(4):
        for nu in range(mu + 1, 4):
            value = clover_traceless(gauge, mu, nu).data[0]
            fields[mu][nu] = value
            fields[nu][mu] = -value
    return fields
