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

from pathlib import Path

import h5py
import numpy as np

from pyquda_utils import core, gamma, phase, source

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
)
from pyquda_measurement_utils.io_corr import ensure_parent_dir
from pyquda_measurement_utils.pion_utils_vibe_develop import array_to_numpy
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print


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


def momentum_tag(momentum):
    return "qx" + str(momentum[0]) + "qy" + str(momentum[1]) + "qz" + str(momentum[2])


def as_momentum_3(momentum):
    if len(momentum) == 4:
        return [int(momentum[0]), int(momentum[1]), int(momentum[2])]
    return [int(momentum[0]), int(momentum[1]), int(momentum[2])]


def _gamma_matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def _matrix_on_backend(matrix, xp, reference_array):
    if type(matrix).__module__.split(".")[0] == xp.__name__:
        return matrix
    if hasattr(matrix, "get"):
        matrix = matrix.get()
    return _asarray_on_queue(matrix, xp, reference_array)


def _matrix_stack(gamma_dict, keys, reference_array):
    xp = _get_xp_from_array(reference_array)
    return xp.stack([
        _matrix_on_backend(
            _gamma_matrix(gamma_dict[key]), xp, reference_array
        )
        for key in keys
    ])


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
            with h5py.File(save_h5, "a") as f:
                for key, value in attrs.items():
                    f.attrs[key] = value

    def load_wall_propagator(self, latt_info, tag):
        return core.LatticePropagator.loadH5(tag + ".h5", "propagator")

    def apply_phase(self, prop, momentum, sign=1, x0=None):
        x0 = [0, 0, 0, 0] if x0 is None else x0
        xp = _get_xp_from_array(prop.data)
        mom_phase = phase.MomentumPhase(prop.latt_info).getPhase(as_momentum_3(momentum), x0=x0)
        if sign == -1:
            mom_phase = mom_phase.conj()
        mom_phase = _matrix_on_backend(mom_phase, xp, prop.data)
        phased = prop.copy()
        phased.data[:] = phased.data * mom_phase[:, :, :, :, :, None, None, None, None]
        return phased

    def contract_wall_2pt(self, latt_info, prop_fw, prop_bw, pion_mom, pion_pair_label):
        xp = _get_xp_from_array(prop_fw.data)
        src_matrix, sink_matrix = self.pion_channel_pairs[pion_pair_label]
        src_gamma = _matrix_on_backend(_gamma_matrix(src_matrix), xp, prop_fw.data)
        sink_gamma = _matrix_on_backend(_gamma_matrix(sink_matrix), xp, prop_fw.data)
        gamma5 = _matrix_on_backend(_gamma_matrix(G5), xp, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        prop_fw_t = prop_fw_phase.lexico(False)
        prop_bw_bar = xp.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, prop_bw.lexico(False).conj(), gamma5, optimize=True)
        prop_bw_src_sink = xp.einsum("ik,tzyxklca,ln->tzyxinca", src_gamma, prop_bw_bar, sink_gamma, optimize=True)
        corr_local = xp.einsum("tzyxjiab,tzyxilba->tzyx", prop_bw_src_sink, prop_fw_t, optimize=True)
        corr_t = xp.einsum("tzyx->t", corr_local, optimize=True)
        return core.gatherLattice(array_to_numpy(corr_t), [0, -1, -1, -1])

    def contract_tmdwf_check(self, latt_info, prop_fw, prop_bw, pion_mom, pion_pair_label):
        xp = _get_xp_from_array(prop_fw.data)
        src_matrix, _ = self.pion_channel_pairs[pion_pair_label]
        src_gamma = _matrix_on_backend(_gamma_matrix(src_matrix), xp, prop_fw.data)
        gamma5 = _matrix_on_backend(_gamma_matrix(G5), xp, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        prop_fw_t = prop_fw_phase.lexico(False)
        corr_list = []
        for bT_dir in self.bT_dir:
            shifted_bw = prop_bw.copy()
            for bT in range(self.bT_length + 1):
                if bT != 0:
                    shifted_bw = prop_bw.shift(bT, bT_dir)
                for bz in range(self.bz_length + 1):
                    shifted = shifted_bw if bz == 0 else shifted_bw.shift(bz, 2)
                    shifted_bar = xp.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, shifted.lexico(False).conj(), gamma5, optimize=True)
                    left = xp.einsum("ij,tzyxjlca->tzyxilca", src_gamma, shifted_bar, optimize=True)
                    corr_local = xp.einsum("tzyxjiab,tzyxilba->tzyx", left, prop_fw_t, optimize=True)
                    corr_t = xp.einsum("tzyx->t", corr_local, optimize=True)
                    corr_list.append(core.gatherLattice(array_to_numpy(corr_t), [0, -1, -1, -1]))
        return np.asarray(corr_list)

    def contract_soft_factor(self, latt_info, prop_fw, prop_bw_src, prop_sink_bw, prop_sink_fw, pion_mom):
        xp = _get_xp_from_array(prop_fw.data)
        gamma5 = _matrix_on_backend(_gamma_matrix(G5), xp, prop_fw.data)
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
        src_ls = _matrix_stack(pion_src_matrices, pion_pair_labels, prop_fw.data)
        sink_ls = _matrix_stack(pion_sink_matrices, pion_pair_labels, prop_fw.data)
        gamma1_ls = _matrix_stack(gamma1_matrices, gamma_pair_labels, prop_fw.data)
        gamma2_ls = _matrix_stack(gamma2_matrices, gamma_pair_labels, prop_fw.data)

        Gw = prop_fw.lexico(False)
        Gw_bperp_dagger = prop_bw_src.lexico(False)
        Gw_dagger = prop_sink_fw.lexico(False)
        Gw_bperp = self.apply_phase(prop_sink_bw, [-2 * pion_mom[0], -2 * pion_mom[1], -2 * pion_mom[2]], 1).lexico(False)
        Gw_dagger_conj = Gw_dagger.conj()

        shape = (len(pion_pair_labels), len(gamma_pair_labels), len(self.bT_dir), self.bT_length + 1, latt_info.global_size[3])
        corr_collect = np.empty(shape, dtype=np.complex128) if latt_info.mpi_rank == 0 else None
        for idir, bT_dir in enumerate(self.bT_dir):
            for bT in range(self.bT_length + 1):
                axis = 3 - bT_dir
                Gw_bperp_shift = xp.roll(Gw_bperp, shift=bT, axis=axis)
                Gw_bperp_dagger_shift = xp.roll(Gw_bperp_dagger.conj(), shift=bT, axis=axis)
                tmp_1 = xp.einsum("tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc", Gw, src_ls, gamma5, Gw_bperp_dagger_shift, gamma5, optimize=True)
                tmp_2 = xp.einsum("tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc", Gw_bperp_shift, sink_ls, gamma5, Gw_dagger_conj, gamma5, optimize=True)
                for isrc in range(len(pion_pair_labels)):
                    for igm in range(len(gamma_pair_labels)):
                        mpi_print(latt_info, f"Contract pion soft factor bT={bT} dir={bT_dir} pion_pair={pion_pair_labels[isrc]} gamma_pair={gamma_pair_labels[igm]}")
                        corr_local = xp.einsum(
                            "tzyxjiba,ik,tzyxklba,lj->tzyx",
                            tmp_1[isrc],
                            gamma2_ls[igm],
                            tmp_2[isrc],
                            gamma1_ls[igm],
                            optimize=True,
                        )
                        corr_t = xp.einsum("tzyx->t", corr_local, optimize=True)
                        corr_global = core.gatherLattice(array_to_numpy(corr_t), [0, -1, -1, -1])
                        if latt_info.mpi_rank == 0:
                            corr_collect[isrc, igm, idir, bT] = corr_global
        return corr_collect, pion_pair_labels, gamma_pair_labels
