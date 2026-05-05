"""Flowed disconnected one-point operator measurements.

This module collects gauge-field and stochastic-fermion one-point measurements
that are commonly used as disconnected-diagram building blocks.  It is a thin
front-end over the validated flowed one-point code in ``pion_EMT_vibe_develop``:

    FlowedDisconnectedQuark1pt
        stochastic quark one-point traces, including EMT-like ``Tmunu`` and
        ``CHI`` diagnostics;

    FlowedDisconnectedGluon1pt
        flowed gluonic EMT one-point building blocks from clover field
        strengths.

Why "disconnected"?
-------------------
Hadron disconnected three-point functions factor into a hadron two-point
function and a vacuum/operator loop, followed by ensemble averaging and vacuum
subtraction.  This module computes the operator-loop side:

    L_O(q, t_flow, tau) = sum_x Phi_q(x) Tr[O(t_flow, x)].

The hadron two-point function, vacuum subtraction, and final connected plus
disconnected combination are intentionally left to downstream analysis.

Current quark operators
-----------------------
The current quark implementation measures the same stochastic flowed EMT
building blocks used by the EMT workflows.  For a Z_n noise source ``xi`` and
solution ``eta = D^{-1} xi`` it computes

    CHI[0](q, t) = sum_x Phi_q(x) xi^dagger(x) eta(x),
    CHI[1](q, t) = sum_x Phi_q(x) xi^dagger(x) xi(x),

and

    T_{nu mu}^q(q, t)
      = -1/2 sum_x Phi_q(x)
        xi^dagger(x) gamma_nu [D_{+mu} - D_{-mu}] eta(x).

The diagonal zero-momentum trace is useful for ringed-fermion normalization:

    sum_mu T_{mu mu}^q(0, t)
      = -1/2 <bar_chi overleftrightarrow{not D} chi>.

The HDF5 output stores this information under ``avg/Tmunu/T11`` through
``avg/Tmunu/T44``.

Current gluon operators
-----------------------
The current gluon implementation measures

    T_{mu nu}^g(q, t)
      = 2 / V3 sum_x Phi_q(x)
        sum_{rho != mu,nu} Tr[F_{mu rho}(x) F_{nu rho}(x)].

This is a flowed gluonic EMT building block.  Renormalization coefficients,
trace/mixing terms, and vacuum subtraction are analysis-level operations.

Validation status
-----------------
These classes reuse the same numerical kernels that are smoke-tested by the
pion/proton EMT workflows.  The separate application scripts in
``application/flowed_disconnected_operator_1pt`` should still be validated with
the intended production geometry, momentum grid, flow-time range, and random
source strategy before physics production.
"""

import numpy as np

from pyquda_measurement_utils.pion_EMT_vibe_develop import GluonEMT, QuarkEMT


class FlowedDisconnectedQuark1pt(QuarkEMT):
    """Stochastic flowed quark one-point disconnected-operator loops."""

    @staticmethod
    def ringed_fermion_kinetic_from_Tmunu(Tmunu, q_index=0):
        """Return ``sum_mu T_{mu mu}(q_index, flow, t)`` from averaged Tmunu.

        ``Tmunu`` is expected to have the writer convention used by
        ``save_emt_quark_1pt_hdf5``:

            Tmunu[mu, nu, q, flow, time].

        The returned array has shape ``[flow, time]``.
        """
        Tmunu = np.asarray(Tmunu)
        return Tmunu[0, 0, q_index] + Tmunu[1, 1, q_index] + Tmunu[2, 2, q_index] + Tmunu[3, 3, q_index]


class FlowedDisconnectedGluon1pt(GluonEMT):
    """Flowed gluon one-point disconnected-operator building blocks."""


__all__ = ["FlowedDisconnectedQuark1pt", "FlowedDisconnectedGluon1pt"]
