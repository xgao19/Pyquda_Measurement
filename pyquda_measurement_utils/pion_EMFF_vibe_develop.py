"""
Pion connected electromagnetic form-factor contractions in PyQUDA.

This module is a local-current specialization of the pion qTMD workflow.  The
connected pion electromagnetic three-point function starts from a source at x0,
a current insertion at x = (tau, x), and a fixed sink at y = (tsep, y):

    C3_g(q, tau; pf, tsep) =
        sum_x sum_y
        exp(-i q . (x - x0)) exp(-i pf . (y - x0))
        Tr[
            S_anti(y, x0) Gamma_sink
            S_q(y, x) Gamma_g S_q(x, x0) Gamma_src
        ].

The final-state pion momentum is set by ``parameters["pf"]``.  Momentum
transfer values are set by ``parameters["qext"]``.  The initial momentum is
therefore pi = pf - q in the usual three-point convention.  The physical EMFF
current is selected from the vector gamma choices, while the code scans all 16
gamma structures for diagnostics and reuse.

Fixed-sink sequential source
----------------------------
The sink sum is absorbed into the meson backward sequential propagator built by
``create_meson_bw_seq_pyquda``.  On the sink time slice it constructs

    eta_seq(y; pf, tsep) =
        delta_{t_y,tsep}
        phase_pf(y - x0)
        Gamma_seq S_neg(y, x0),

    Gamma_seq = gamma5 Gamma_sink^dagger gamma5,

where ``phase_pf`` is produced by ``MomentumPhase`` with the same sign
convention used by the application.  The sequential propagator solves

    D S_seq = eta_seq.

After the inversion this module converts it to an antiquark-like backward line
with gamma5 hermiticity,

    S_seq_anti(x, x0; pf, tsep) = gamma5 S_seq(x, x0)^dagger gamma5.

The contraction evaluated by ``contract_EMFF`` is then

    C3_g(q, tau; pf, tsep) =
        sum_x exp(-i q . (x - x0))
        Tr[
            S_seq_anti(x, x0; pf, tsep)
            Gamma_g
            S_q(x, x0)
            Gamma_src
        ].

Boost-smearing convention
-------------------------
Pion measurements have a quark and an antiquark line, so this module supports
independent source and sink boosts for both lines:

    pos_boost_src: source smearing for the forward quark propagator.
    pos_boost_sink: sink smearing for the forward quark propagator in C2.
    neg_boost_src: source smearing for the antiquark propagator in C2.
    neg_boost_sink: sink smearing for the antiquark propagator and sequential
                    source.

If these four parameters are not provided, they fall back to ``pos_boost`` and
``neg_boost``.  When source and sink boosts are equal, this reproduces the
older pion qTMD/TMDWF behavior.

The current implementation computes connected diagrams only.  It scans all 16
bilinear sink/current gamma structures in the standard project order:

    5, T, T5, X, X5, Y, Y5, Z, Z5, I, SXT, SXY, SXZ, SYT, SYZ, SZT.
"""

import numpy as np

from pyquda_utils import core
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt,
    gamma_stack,
    meson_backward_line,
    my_gammas,
    source_gamma_stack,
)


class pion_EMFF:
    def __init__(self, parameters):
        self.pf = parameters["pf"]
        self.qlist = parameters["qext"]
        self.pilist = parameters["p_2pt"]

        self.width = parameters["width"]
        self.pos_boost_src = parameters.get("pos_boost_src", parameters.get("pos_boost", [0, 0, 0]))
        self.pos_boost_sink = parameters.get("pos_boost_sink", parameters.get("pos_boost", [0, 0, 0]))
        self.neg_boost_src = parameters.get("neg_boost_src", parameters.get("neg_boost", [0, 0, 0]))
        self.neg_boost_sink = parameters.get("neg_boost_sink", parameters.get("neg_boost", [0, 0, 0]))

        self.t_insert = parameters["t_insert"]
        self.save_propagators = parameters["save_propagators"]

    def contract_2pt_pion(self, latt_info, prop_pos, prop_neg, phases, tag, src_gamma="fixed_g5"):
        mpi_print(latt_info, "Begin pion EMFF sink smearing")
        prop_pos = boosted_smearing(prop_pos, w=self.width, boost=self.pos_boost_sink)
        prop_neg = boosted_smearing(prop_neg, w=self.width, boost=self.neg_boost_sink)
        mpi_print(latt_info, "Pion EMFF sink smearing completed")

        corr = contract_pion_2pt(latt_info, prop_pos, prop_neg, phases, src_gamma=src_gamma)

        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr

    def contract_EMFF(self, latt_info, prop_pos, seq_bw_prop, phases, src_gamma="fixed_g5"):
        xp = _get_xp_from_array(prop_pos.data)
        phases = _asarray_on_queue(phases, xp, prop_pos.data)
        current_gamma_ls = gamma_stack(prop_pos.data)
        source_gamma_ls = source_gamma_stack(src_gamma, current_gamma_ls, prop_pos.data)
        seq_bw_line = meson_backward_line(seq_bw_prop)

        current_inserted = xp.einsum("wtzyxjicf,gim->gwtzyxjmcf", seq_bw_line, current_gamma_ls, optimize=True)
        corr_local = xp.einsum(
            "gwtzyxjiab,wtzyxilba,glj->gwtzyx",
            current_inserted,
            prop_pos.data,
            source_gamma_ls,
            optimize=True,
        )
        corr = xp.einsum("qwtzyx,gwtzyx->gqt", phases, corr_local, optimize=True)
        return core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])
