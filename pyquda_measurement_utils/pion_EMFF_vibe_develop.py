"""
Pion connected electromagnetic form-factor contractions in PyQUDA.

This module is a local-current specialization of the pion qTMD workflow.  It
computes connected pion electromagnetic three-point functions with a source at
x0, a current insertion at x = (tau, x), and a fixed sink at
y = (tsep, y).  The intended momentum convention is

    pf   = final pion momentum projected at the sink,
    q    = qext = momentum injected by the current,
    pi   = initial pion momentum at the source,
    q    = pf - pi,
    pi   = pf - q.

For example, in a Breit-frame setup with pf = +Pz and pi = -Pz, the current
momentum is q = +2Pz.  In the production application this corresponds to
pf = [0, 0, 3, 0] and qext = [0, 0, 6, 0], so pi = [0, 0, -3, 0].

With these labels, the connected three-point function is

    C3_g(q, tau; pf, tsep) =
        sum_x sum_y
        exp(-i q . (x - x0)) exp(-i pf . (y - x0))
        Tr[
            S_anti(y, x0) Gamma_sink
            S_q(y, x) Gamma_g S_q(x, x0) Gamma_src
        ].

The final-state pion momentum is set by ``parameters["pf"]``.  Momentum
transfer values are set by ``parameters["qext"]``.  The two-point momenta used
for normalization and ratio construction are set by ``parameters["p_2pt"]``.
For each saved three-point dataset at a given qext, the analysis should combine
it with two-point functions at both

    pf,
    pi = pf - qext.

If only the opposite sign of a pion momentum was saved in the two-point file,
the analysis may use parity symmetry, C2(p) = C2(-p), after confirming the
same smearing and source/sink convention.  The physical EMFF baseline usually
uses Gamma_sink = gamma5, Gamma_src = gamma5, and the temporal vector current
Gamma_g = gamma_T, labeled as ``T`` in ``my_gammas``.  The code still scans all
16 gamma structures for diagnostics and reuse.

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

    pos_boost_src: source-side smearing for the forward quark propagator.
    pos_boost_sink: sink-side smearing for the forward quark line in C3.
    neg_boost_src: source-side smearing for the antiquark propagator.
    neg_boost_sink: sink-side smearing for the antiquark line and sequential
                    source in C3.

The pion two-point function has no current momentum transfer, so it uses the
source-side boosts on both source and sink ends.  The independent sink-side
boosts are reserved for the EMFF three-point sequential sink.

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
    array_to_numpy,
    contract_pion_2pt_multi_src_gamma,
    gamma_stack,
    meson_backward_line,
    my_gammas,
    source_gamma_stack,
    source_gamma_provenance,
    zeros_on_backend,
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

    def contract_2pt_pion(
        self, latt_info, prop_pos, prop_neg, phases, tag,
        src_gamma="5", attrs=None,
    ):
        self.contract_2pt_pion_multi_src_gamma(
            latt_info,
            prop_pos,
            prop_neg,
            phases,
            {src_gamma: tag},
            {src_gamma: attrs},
        )

    def contract_2pt_pion_multi_src_gamma(
        self, latt_info, prop_pos, prop_neg, phases, tags_by_src_gamma,
        attrs_by_src_gamma=None,
    ):
        mpi_print(latt_info, "Begin pion EMFF sink smearing")
        prop_pos = boosted_smearing(prop_pos, w=self.width, boost=self.pos_boost_src)
        prop_neg = boosted_smearing(prop_neg, w=self.width, boost=self.neg_boost_src)
        mpi_print(latt_info, "Pion EMFF sink smearing completed")

        corr_by_src = contract_pion_2pt_multi_src_gamma(
            latt_info,
            prop_pos,
            prop_neg,
            phases,
            list(tags_by_src_gamma),
        )

        if latt_info.mpi_rank == 0:
            attrs_by_src_gamma = attrs_by_src_gamma or {}
            for src_gamma, tag in tags_by_src_gamma.items():
                output_attrs = dict(attrs_by_src_gamma.get(src_gamma) or {})
                output_attrs.update(source_gamma_provenance(src_gamma))
                save_proton_c2pt_hdf5(
                    corr_by_src[src_gamma],
                    tag,
                    my_gammas,
                    self.pilist,
                    attrs=output_attrs,
                )
        del corr_by_src

    def contract_EMFF(self, latt_info, prop_pos, seq_bw_prop, phases, src_gamma="5"):
        return self.contract_EMFF_multi_src_gamma(
            latt_info,
            prop_pos,
            seq_bw_prop,
            phases,
            [src_gamma],
        )[src_gamma]

    def contract_EMFF_multi_src_gamma(self, latt_info, prop_pos, seq_bw_prop, phases, src_gammas):
        xp = _get_xp_from_array(prop_pos.data)
        phases = _asarray_on_queue(phases, xp, prop_pos.data)
        current_gamma_ls = gamma_stack(prop_pos.data)
        source_gamma_ls_by_src = {
            src_gamma: source_gamma_stack(src_gamma, current_gamma_ls, prop_pos.data)
            for src_gamma in src_gammas
        }
        seq_bw_line = meson_backward_line(seq_bw_prop)
        corr_local_by_src = {
            src_gamma: zeros_on_backend(
                (len(current_gamma_ls), phases.shape[0], latt_info.size[3]),
                dtype=prop_pos.data.dtype,
                xp=xp,
                reference_array=prop_pos.data,
            )
            for src_gamma in src_gammas
        }

        for gamma_idx, current_gamma in enumerate(current_gamma_ls):
            current_inserted = xp.einsum("wtzyxjicf,im->wtzyxjmcf", seq_bw_line, current_gamma, optimize=True)
            for src_gamma in src_gammas:
                corr_site = xp.einsum(
                    "wtzyxjiab,wtzyxilba,lj->wtzyx",
                    current_inserted,
                    prop_pos.data,
                    source_gamma_ls_by_src[src_gamma][gamma_idx],
                    optimize=True,
                )
                corr_local_by_src[src_gamma][gamma_idx] = xp.einsum("qwtzyx,wtzyx->qt", phases, corr_site, optimize=True)
                del corr_site
            del current_inserted

        corr_by_src = {
            src_gamma: core.gatherLattice(array_to_numpy(corr_local), [2, -1, -1, -1])
            for src_gamma, corr_local in corr_local_by_src.items()
        }

        del corr_local_by_src, source_gamma_ls_by_src, seq_bw_line
        return corr_by_src
