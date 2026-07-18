"""
Pion connected qTMD and PDF contractions in PyQUDA.

This module is the pion analogue of ``proton_qTMD_pyquda.py``.  The main
structural difference is that a pion correlator contains one quark line and one
antiquark line, while the proton code contracts three quark lines.  The
positive-boost line is the fixed-sink spectator and the negative-boost line is
the active line on which the qTMD/PDF operator acts, matching pion EMT.

Conventions and propagators
---------------------------
Let S_q(x, y) be the quark propagator from source y to sink x.  The antiquark
line is constructed with gamma5 hermiticity,

    S_anti(x, y) = gamma5 * S_q(x, y)^dagger * gamma5.

In the code this is implemented by ``meson_backward_line``.  The PyQUDA
propagator layout is kept in even-odd lattice order,

    prop.data[w, t, z, y, x_cb, spin_sink, spin_src, color_sink, color_src].

The helper ``gamma_stack`` prepares the 16 bilinear gamma matrices in the same
order as the existing proton and pion-TMDWF workflows:

    5, T, T5, X, X5, Y, Y5, Z, Z5, I, SXT, SXY, SXZ, SYT, SYZ, SZT.

Two-point function
------------------
For a source gamma Gamma_src and each sink gamma Gamma_g, the connected pion
two-point function is

    C2_g(p, t) =
        sum_x exp(-i p . (x - x0))
        Tr_spin,color[
            S_anti(x, x0) Gamma_g S_q(x, x0) Gamma_src
        ].

The default ``src_gamma='5'`` gives the usual pseudoscalar pion source,
while all 16 sink gamma structures are still scanned and saved.  This is cheap
and useful for diagnostics or later operator studies.

Three-point function and fixed-sink sequential source
-----------------------------------------------------
The connected pion qTMD three-point function starts from a source at x0, an
operator insertion at x = (tau, x), and a fixed sink at y = (tsep, y):

    C3_g(q, b, tau; pf, tsep) =
        sum_x sum_y
        exp(-i q . (x - x0)) exp(-i pf . (y - x0))
        Tr[
            S_anti(y, x0) Gamma_sink
            S_q(y, x) Gamma_g O_b S_q(x, x0) Gamma_src
        ].

Gamma_sink is the fixed pion sink interpolator used to build the sequential
source.  Gamma_g is the scanned insertion gamma.  O_b is the nonlocal qTMD/PDF
operator applied to the forward quark line at the insertion.

The sink sum is absorbed into a fixed-sink sequential propagator.  In
``create_meson_bw_seq_pyquda`` the right-hand side is built on the sink time
slice,

    eta_seq(y; pf, tsep) =
        delta_{t_y,tsep}
        phase_pf(y - x0)
        Gamma_seq Q_pos_SS(y, x0),

    Gamma_seq = gamma5 Gamma_sink^dagger gamma5,

where ``phase_pf`` is produced by ``MomentumPhase`` with the same sign
convention used by the application.  The sequential propagator solves

    D S_seq = eta_seq.

After the inversion this module forms the antiquark-like backward line

    S_seq_anti(x, x0; pf, tsep) = gamma5 S_seq(x, x0)^dagger gamma5.

The contraction used in the code is therefore the sink-summed form

    C3_g(q, b, tau; pf, tsep) =
        sum_x exp(-i q . (x - x0))
        Tr[
            S_seq_anti(x, x0; pf, tsep)
            Gamma_g
            O_b Q_neg_SP(x, x0)
            Gamma_src
        ].

CG qTMD operator
----------------
For the CG qTMD path, O_b is currently a coordinate-gauge style displacement
without explicit gauge links:

    O_b S_q(x, x0) = S_q(x + bT * e_perp + bz * ez, x0).

The transverse direction is scanned over x and y, stored as ``b_X`` and
``b_Y`` in the HDF5 output.  The code contracts both directions separately so
successive shifts stay small.

PDF operators
-------------
The PDF path is the straight-z special case with bT = 0:

    b = bz * ez.

Two variants are supported:

    CG_PDF:  ordinary lattice shift, no explicit gauge link.
    GI_PDF:  covariant z-direction shift using the gauge field links.

The GI path repeatedly applies ``gauge.pure_gauge.covDev`` in +z or -z for
unit changes in bz.  The current implementation intentionally only supports
nearest-neighbor incremental changes in the ordered Wilson-line list; larger
jumps raise an error instead of silently producing a wrong path.

Output shape convention
-----------------------
The contraction routines naturally return arrays as

    [Wilson_index, gamma, momentum, time].

The Perlmutter application transposes them before calling the shared qTMD HDF5
writer, which expects

    [Wilson_index, momentum, gamma, time].

This explicit transpose is important: without it, a single gamma file can be
written with the wrong momentum/gamma interpretation.
"""

import numpy as np

from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.qtmd_operator_utils import (
    apply_gi_qtmd_staple_to_propagator,
    build_gi_qtmd_staple_links,
    shift_propagator_pdf_gi,
    shift_qtmd_cg,
)
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.tools import mpi_print
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    contract_pion_2pt,
    contract_pion_gamma_scan_from_backward_line,
    meson_backward_line,
    my_gammas,
    source_gamma_provenance,
)


class pion_TMD:
    def __init__(self, parameters):
        self.pilist = parameters["p_2pt"]

        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]

    def contract_2pt_pion(
        self, latt_info, spectator_prop, active_prop, phases, tag,
        src_gamma="5", attrs=None,
    ):
        mpi_print(latt_info, "Begin pion sink smearing")
        spectator_prop = boosted_smearing(
            spectator_prop, w=self.width, boost=self.pos_boost
        )
        active_prop = boosted_smearing(
            active_prop, w=self.width, boost=self.neg_boost
        )
        mpi_print(latt_info, "Pion sink smearing completed")

        corr = contract_pion_2pt(
            latt_info, spectator_prop, active_prop, phases, src_gamma=src_gamma
        )

        if latt_info.mpi_rank == 0:
            output_attrs = dict(attrs or {})
            output_attrs.update(source_gamma_provenance(src_gamma))
            save_proton_c2pt_hdf5(
                corr, tag, my_gammas, self.pilist, attrs=output_attrs
            )
        del corr

    def contract_qTMD_CG(self, latt_info, active_prop, seq_bw_prop, phases, W_index_list_dir0, W_index_list_dir1, src_gamma="5"):
        seq_bw_line = meson_backward_line(seq_bw_prop)

        pion_TMDs = []
        W_index_list = W_index_list_dir0 + W_index_list_dir1

        tmd_active_prop_dir0 = active_prop.copy()
        for iW, W_index in enumerate(W_index_list_dir0):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir0[iW - 1]
            tmd_active_prop_dir0 = shift_qtmd_cg(
                tmd_active_prop_dir0, W_index, W_index_previous
            )
            corr = contract_pion_gamma_scan_from_backward_line(
                latt_info, tmd_active_prop_dir0, seq_bw_line, phases, [src_gamma]
            )[src_gamma]
            if latt_info.mpi_rank == 0:
                pion_TMDs.append(corr)
        del tmd_active_prop_dir0

        tmd_active_prop_dir1 = active_prop.copy()
        for iW, W_index in enumerate(W_index_list_dir1):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1 + len(W_index_list_dir0)}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir1[iW - 1]
            tmd_active_prop_dir1 = shift_qtmd_cg(
                tmd_active_prop_dir1, W_index, W_index_previous
            )
            corr = contract_pion_gamma_scan_from_backward_line(
                latt_info, tmd_active_prop_dir1, seq_bw_line, phases, [src_gamma]
            )[src_gamma]
            if latt_info.mpi_rank == 0:
                pion_TMDs.append(corr)
        del tmd_active_prop_dir1

        return np.asarray(pion_TMDs) if latt_info.mpi_rank == 0 else None

    def contract_qTMD_GI(self, latt_info, gauge, active_prop, seq_bw_prop, phases, W_index_list_dir0, W_index_list_dir1, src_gamma="5"):
        seq_bw_line = meson_backward_line(seq_bw_prop)

        W_index_list = W_index_list_dir0 + W_index_list_dir1
        mpi_print(latt_info, f"Build {len(W_index_list)} connected pion GI_qTMD staple transporters.")
        staple_links = build_gi_qtmd_staple_links(gauge, W_index_list)

        pion_TMDs = []
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract pion qTMD GI {iW + 1}/{len(W_index_list)} {W_index}")
            shifted_prop = apply_gi_qtmd_staple_to_propagator(
                active_prop, W_index, staple_links
            )
            corr = contract_pion_gamma_scan_from_backward_line(
                latt_info, shifted_prop, seq_bw_line, phases, [src_gamma]
            )[src_gamma]
            if latt_info.mpi_rank == 0:
                pion_TMDs.append(corr)
            del shifted_prop

        return np.asarray(pion_TMDs) if latt_info.mpi_rank == 0 else None

    def contract_PDF(self, latt_info, gauge, active_prop, seq_bw_prop, phases, W_index_list, src_gamma="5", gauge_invariant=True):
        seq_bw_line = meson_backward_line(seq_bw_prop)

        pion_PDFs = []
        pdf_active_prop = active_prop.copy()
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract pion PDF {'GI' if gauge_invariant else 'CG'} {iW + 1}/{len(W_index_list)} {W_index}")
            if W_index[1] == 0:
                W_index_previous = [0, 0, 0, 0]
                pdf_active_prop = active_prop.copy()
            elif W_index[1] > 0:
                W_index_previous = W_index_list[iW - 1]
            elif W_index[1] == -1:
                W_index_previous = [0, 0, 0, 0]
                pdf_active_prop = active_prop.copy()
            else:
                W_index_previous = W_index_list[iW - 1]

            if gauge_invariant:
                pdf_active_prop = shift_propagator_pdf_gi(
                    gauge, pdf_active_prop, W_index, W_index_previous
                )
            else:
                pdf_active_prop = shift_qtmd_cg(
                    pdf_active_prop, W_index, W_index_previous
                )

            corr = contract_pion_gamma_scan_from_backward_line(
                latt_info, pdf_active_prop, seq_bw_line, phases, [src_gamma]
            )[src_gamma]
            if latt_info.mpi_rank == 0:
                pion_PDFs.append(corr)
        del pdf_active_prop

        return np.asarray(pion_PDFs) if latt_info.mpi_rank == 0 else None
