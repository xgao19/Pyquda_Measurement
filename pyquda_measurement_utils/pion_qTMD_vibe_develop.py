"""
Pion connected qTMD and PDF contractions in PyQUDA.

This module is the pion analogue of ``proton_qTMD_pyquda.py``.  The main
structural difference is that a pion correlator contains one quark line and one
antiquark line, while the proton code contracts three quark lines.  For this
reason the pion code uses ``pos_boost`` and ``neg_boost`` to describe the
smearing boosts of the quark and antiquark lines, matching the convention used
in ``pion_qTMDWF_pyquda.py``.

Conventions and propagators
---------------------------
Let S_q(x, y) be the quark propagator from source y to sink x.  The antiquark
line is constructed with gamma5 hermiticity,

    S_anti(x, y) = gamma5 * S_q(x, y)^dagger * gamma5.

In the code this is implemented by ``_meson_backward_line``.  The PyQUDA
propagator layout is kept in even-odd lattice order,

    prop.data[w, t, z, y, x_cb, spin_sink, spin_src, color_sink, color_src].

The helper ``_gamma_stack`` prepares the 16 bilinear gamma matrices in the same
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

The default ``src_gamma='fixed_g5'`` gives the usual pseudoscalar pion source,
while all 16 sink gamma structures are still scanned and saved.  This is cheap
and useful for diagnostics or later operator studies.

Fixed-sink sequential source for three-point functions
------------------------------------------------------
The application builds a meson sequential propagator with
``create_meson_bw_seq_pyquda``.  It fixes the final-state momentum pf and the
sink time tsep = t_insert, then this module converts that sequential propagator
to an antiquark-like backward line using gamma5 hermiticity.  The connected
three-point function then has the schematic form

    C3_g(q, b, tau; pf) =
        sum_x exp(-i q . (x - x0))
        Tr[
            S_seq_anti(x, x0; pf, tsep)
            Gamma_g
            O_b S_q(x, x0)
            Gamma_src
        ].

Here O_b is the nonlocal displacement/Wilson-line operator applied to the
forward quark line before contraction.

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

from pyquda_utils import core, gamma
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print


my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
my_pyquda_gammas = [
    gamma.gamma(15),
    gamma.gamma(8),
    gamma.gamma(7),
    gamma.gamma(1),
    gamma.gamma(14),
    gamma.gamma(2),
    gamma.gamma(13),
    gamma.gamma(4),
    gamma.gamma(11),
    gamma.gamma(0),
    gamma.gamma(9),
    gamma.gamma(3),
    gamma.gamma(5),
    gamma.gamma(10),
    gamma.gamma(6),
    gamma.gamma(12),
]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]

G5 = gamma.gamma(15)


def _gamma_stack(reference_array):
    xp = _get_xp_from_array(reference_array)
    first_gamma = my_pyquda_gammas[0]
    if xp.__name__ == "dpnp":
        gamma_ls = xp.empty((len(my_pyquda_gammas),) + first_gamma.shape, dtype=first_gamma.dtype, device=first_gamma.device)
    else:
        gamma_ls = xp.empty((len(my_pyquda_gammas),) + first_gamma.shape, dtype=first_gamma.dtype)

    for gamma_idx, gamma_matrix in enumerate(my_pyquda_gammas):
        gamma_ls[gamma_idx] = _asarray_on_queue(gamma_matrix, xp, reference_array)
    return gamma_ls


def _gamma_from_label(label):
    if label not in my_gammas:
        raise ValueError(f"Invalid gamma label: {label}. Expected one of {my_gammas}.")
    return my_pyquda_gammas[my_gammas.index(label)]


def _source_gamma_stack(src_gamma, sink_gamma_ls, reference_array):
    xp = _get_xp_from_array(reference_array)
    gamma5 = _asarray_on_queue(G5, xp, reference_array)

    if src_gamma == "fixed_g5":
        source_gamma_ls = sink_gamma_ls.copy()
        source_gamma_ls[:] = gamma5
    elif src_gamma == "same_as_sink":
        source_gamma_ls = sink_gamma_ls.copy()
    elif src_gamma == "dagger_of_sink":
        source_gamma_ls = xp.einsum("ab,gbc,cd->gad", gamma5, xp.swapaxes(sink_gamma_ls.conj(), 1, 2), gamma5, optimize=True)
    elif src_gamma in my_gammas:
        source_gamma_ls = sink_gamma_ls.copy()
        source_gamma_ls[:] = _asarray_on_queue(_gamma_from_label(src_gamma), xp, reference_array)
    else:
        raise ValueError(
            f"Invalid src_gamma: {src_gamma}. "
            "Use a gamma label or one of ['fixed_g5', 'same_as_sink', 'dagger_of_sink']."
        )
    return source_gamma_ls


def _meson_backward_line(prop):
    xp = _get_xp_from_array(prop.data)
    gamma5 = _asarray_on_queue(G5, xp, prop.data)
    return xp.einsum("ij,wtzyxilab,kl->wtzyxkjba", gamma5, prop.data.conj(), gamma5, optimize=True)


class pion_TMD:
    def __init__(self, parameters):
        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]

        self.pf = parameters["pf"]
        self.qlist = parameters["qext"]
        self.qlist_PDF = parameters.get("qext_PDF", self.qlist)
        self.pilist = parameters["p_2pt"]

        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]

        self.t_insert = parameters["t_insert"]
        self.save_propagators = parameters["save_propagators"]

    def contract_2pt_pion(self, latt_info, prop_f, prop_b, phases, tag, src_gamma="fixed_g5"):
        mpi_print(latt_info, "Begin pion sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        prop_b = boosted_smearing(prop_b, w=self.width, boost=self.neg_boost)
        mpi_print(latt_info, "Pion sink smearing completed")

        xp = _get_xp_from_array(prop_f.data)
        sink_gamma_ls = _gamma_stack(prop_f.data)
        source_gamma_ls = _source_gamma_stack(src_gamma, sink_gamma_ls, prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)

        bw_prop = _meson_backward_line(prop_b)
        bw_prop = xp.einsum("wtzyxjicf,gim->gwtzyxjmcf", bw_prop, sink_gamma_ls, optimize=True)
        corr_local = xp.einsum("gwtzyxjiab,wtzyxilba,glj->gwtzyx", bw_prop, prop_f.data, source_gamma_ls, optimize=True)
        corr = core.gatherLattice(xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->gqt", phases, corr_local, optimize=True)), [2, -1, -1, -1])

        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr, corr_local, bw_prop

    def contract_qTMD_CG(self, latt_info, prop_f, seq_bw_prop, phases, W_index_list_dir0, W_index_list_dir1, src_gamma="fixed_g5"):
        xp = _get_xp_from_array(prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        sink_gamma_ls = _gamma_stack(prop_f.data)
        source_gamma_ls = _source_gamma_stack(src_gamma, sink_gamma_ls, prop_f.data)
        seq_bw_line = _meson_backward_line(seq_bw_prop)

        pion_TMDs = []
        W_index_list = W_index_list_dir0 + W_index_list_dir1

        tmd_forward_prop_dir0 = prop_f.copy()
        for iW, W_index in enumerate(W_index_list_dir0):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir0[iW - 1]
            tmd_forward_prop_dir0 = self.create_fw_prop_TMD_CG(tmd_forward_prop_dir0, W_index, W_index_previous)
            pion_TMDs.append(self._contract_qTMD_one_shift(seq_bw_line, tmd_forward_prop_dir0, sink_gamma_ls, source_gamma_ls, phases))
        del tmd_forward_prop_dir0

        tmd_forward_prop_dir1 = prop_f.copy()
        for iW, W_index in enumerate(W_index_list_dir1):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1 + len(W_index_list_dir0)}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir1[iW - 1]
            tmd_forward_prop_dir1 = self.create_fw_prop_TMD_CG(tmd_forward_prop_dir1, W_index, W_index_previous)
            pion_TMDs.append(self._contract_qTMD_one_shift(seq_bw_line, tmd_forward_prop_dir1, sink_gamma_ls, source_gamma_ls, phases))
        del tmd_forward_prop_dir1

        return np.array(pion_TMDs)

    def contract_PDF(self, latt_info, gauge, prop_f, seq_bw_prop, phases, W_index_list, src_gamma="fixed_g5", gauge_invariant=True):
        xp = _get_xp_from_array(prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        sink_gamma_ls = _gamma_stack(prop_f.data)
        source_gamma_ls = _source_gamma_stack(src_gamma, sink_gamma_ls, prop_f.data)
        seq_bw_line = _meson_backward_line(seq_bw_prop)

        pion_PDFs = []
        pdf_forward_prop = prop_f.copy()
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract pion PDF {'GI' if gauge_invariant else 'CG'} {iW + 1}/{len(W_index_list)} {W_index}")
            if W_index[1] == 0:
                W_index_previous = [0, 0, 0, 0]
                pdf_forward_prop = prop_f.copy()
            elif W_index[1] > 0:
                W_index_previous = W_index_list[iW - 1]
            elif W_index[1] == -1:
                W_index_previous = [0, 0, 0, 0]
                pdf_forward_prop = prop_f.copy()
            else:
                W_index_previous = W_index_list[iW - 1]

            if gauge_invariant:
                pdf_forward_prop = self.create_fw_prop_PDF_GI(gauge, pdf_forward_prop, W_index, W_index_previous)
            else:
                pdf_forward_prop = self.create_fw_prop_TMD_CG(pdf_forward_prop, W_index, W_index_previous)

            pion_PDFs.append(self._contract_qTMD_one_shift(seq_bw_line, pdf_forward_prop, sink_gamma_ls, source_gamma_ls, phases))
        del pdf_forward_prop

        return np.array(pion_PDFs)

    def _contract_qTMD_one_shift(self, seq_bw_line, shifted_prop, sink_gamma_ls, source_gamma_ls, phases):
        xp = _get_xp_from_array(shifted_prop.data)
        sink_inserted = xp.einsum("wtzyxjicf,gim->gwtzyxjmcf", seq_bw_line, sink_gamma_ls, optimize=True)
        corr_local = xp.einsum(
            "gwtzyxjiab,wtzyxilba,glj->gwtzyx",
            sink_inserted,
            shifted_prop.data,
            source_gamma_ls,
            optimize=True,
        )
        corr = xp.einsum("qwtzyx,gwtzyx->gqt", phases, corr_local, optimize=True)
        return core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])

    def create_TMD_Wilsonline_index_list_CG(self):
        index_list_trans0 = []
        index_list_trans1 = []

        for current_bz in range(0, self.b_z + 1):
            for current_b_T in range(0, self.b_T + 1):
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])

        return self._reorder_wilson_indices(index_list_trans0), self._reorder_wilson_indices(index_list_trans1)

    def _reorder_wilson_indices(self, index_list):
        sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
        reordered = []
        i = 0
        while i < len(sorted_list) - 1:
            current = sorted_list[i]
            next_index = sorted_list[i + 1]
            if abs(current[0] - next_index[0]) > 1 or abs(current[1] - next_index[1]) > 1:
                best_match = next_index
                best_diff = max(abs(current[0] - next_index[0]), abs(current[1] - next_index[1]))
                for candidate in sorted_list[i + 2 :]:
                    diff = max(abs(current[0] - candidate[0]), abs(current[1] - candidate[1]))
                    if diff < best_diff:
                        best_match = candidate
                        best_diff = diff
                if best_match != next_index:
                    best_index = sorted_list.index(best_match)
                    sorted_list[i + 1], sorted_list[best_index] = sorted_list[best_index], sorted_list[i + 1]
            reordered.append(current)
            i += 1

        if i < len(sorted_list):
            reordered.append(sorted_list[-1])
        return reordered

    def create_fw_prop_TMD_CG(self, prop_f, W_index, W_index_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3]
        z_direction = 2

        previous_b_T = W_index_previous[0]
        previous_bz = W_index_previous[1]

        return prop_f.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), z_direction)

    def create_PDF_Wilsonline_index_list(self):
        index_list = []

        for current_bz in range(0, self.b_z + 1):
            index_list.append([0, current_bz, 0, 0])

        for current_bz in range(0, self.b_z + 1):
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])

        return index_list

    def create_fw_prop_PDF_GI(self, gauge, prop_f, W_index, W_index_previous):
        current_bz = W_index[1]
        previous_bz = W_index_previous[1]

        for spin in range(4):
            for color in range(3):
                fermion = prop_f.getFermion(spin, color)
                if current_bz - previous_bz == 0:
                    fermion_shift = fermion
                elif current_bz - previous_bz == 1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                elif current_bz - previous_bz == -1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 6)
                else:
                    raise ValueError("Invalid shift for PDF Wilson line")
                prop_f.setFermion(fermion_shift, spin, color)

        return prop_f
