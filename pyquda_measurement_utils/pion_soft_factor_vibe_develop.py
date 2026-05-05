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

from pyquda_measurement_utils.io_corr import ensure_parent_dir
from pyquda_measurement_utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print


soft_factor_gammas = ["5", "I", "X", "Y", "X5", "Y5"]
soft_factor_pyquda_gammas = {
    "5": gamma.gamma(15),
    "I": gamma.gamma(0),
    "X": gamma.gamma(1),
    "Y": gamma.gamma(2),
    "X5": gamma.gamma(1) @ gamma.gamma(15),
    "Y5": gamma.gamma(2) @ gamma.gamma(15),
}
soft_factor_pion_interpolators = {
    "Z5-X5": gamma.gamma(4) @ gamma.gamma(15) - gamma.gamma(1) @ gamma.gamma(15),
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


def _to_numpy(xp, arr):
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(arr)
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _matrix_stack(gamma_dict, keys, reference_array):
    xp = _get_xp_from_array(reference_array)
    first_gamma = _gamma_matrix(gamma_dict[keys[0]])
    if xp.__name__ == "dpnp":
        matrix_ls = xp.empty((len(keys),) + first_gamma.shape, dtype=first_gamma.dtype, device=first_gamma.device)
    else:
        matrix_ls = xp.empty((len(keys),) + first_gamma.shape, dtype=first_gamma.dtype)
    for idx, key in enumerate(keys):
        matrix_ls[idx] = _matrix_on_backend(_gamma_matrix(gamma_dict[key]), xp, reference_array)
    return matrix_ls


class pion_soft_factor:
    def __init__(self, parameters):
        self.quark_mom = [as_momentum_3(mom) for mom in parameters["quark_mom"]]
        self.bT_dir = parameters["bT_dir"]
        self.bT_length = parameters["bT_length"]
        self.bz_length = parameters.get("bz_length", 0)
        self.tsep_list = parameters["tsep_list"]
        self.pion_src = parameters.get("pion_src", soft_factor_pion_interpolators)
        self.pion_sink = parameters.get("pion_sink", soft_factor_pion_interpolators)
        self.Gamma1 = parameters.get("Gamma1", soft_factor_pyquda_gammas)
        self.Gamma2 = parameters.get("Gamma2", soft_factor_pyquda_gammas)

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

    def contract_wall_2pt(self, latt_info, prop_fw, prop_bw, pion_mom, src_key, sink_keys):
        xp = _get_xp_from_array(prop_fw.data)
        src_gamma = _matrix_on_backend(_gamma_matrix(self.pion_src[src_key]), xp, prop_fw.data)
        sink_gamma_ls = _matrix_stack(self.pion_sink, sink_keys, prop_fw.data)
        gamma5 = _matrix_on_backend(_gamma_matrix(G5), xp, prop_fw.data)
        prop_fw_phase = self.apply_phase(prop_fw, [-pion_mom[0], -pion_mom[1], -pion_mom[2]], 1)
        prop_fw_t = prop_fw_phase.lexico(False)
        prop_bw_bar = xp.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, prop_bw.lexico(False).conj(), gamma5, optimize=True)
        prop_bw_src_sink = xp.einsum("ik,tzyxklca,gln->gtzyxinca", src_gamma, prop_bw_bar, sink_gamma_ls, optimize=True)
        corr_local = xp.einsum("gtzyxjiab,tzyxilba->gtzyx", prop_bw_src_sink, prop_fw_t, optimize=True)
        corr_t = xp.einsum("gtzyx->gt", corr_local, optimize=True)
        return core.gatherLattice(_to_numpy(xp, corr_t), [1, -1, -1, -1])

    def contract_tmdwf_check(self, latt_info, prop_fw, prop_bw, pion_mom, src_key):
        xp = _get_xp_from_array(prop_fw.data)
        src_gamma = _matrix_on_backend(_gamma_matrix(self.pion_src[src_key]), xp, prop_fw.data)
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
                    corr_list.append(core.gatherLattice(_to_numpy(xp, corr_t), [0, -1, -1, -1]))
        return np.asarray(corr_list)

    def contract_soft_factor(self, latt_info, prop_fw, prop_bw_src, prop_sink_bw, prop_sink_fw, pion_mom):
        xp = _get_xp_from_array(prop_fw.data)
        gamma5 = _matrix_on_backend(_gamma_matrix(G5), xp, prop_fw.data)
        src_keys = list(self.pion_src.keys())
        sink_keys = list(self.pion_sink.keys())
        gamma1_keys = list(self.Gamma1.keys())
        gamma2_keys = list(self.Gamma2.keys())
        src_ls = _matrix_stack(self.pion_src, src_keys, prop_fw.data)
        sink_ls = _matrix_stack(self.pion_sink, sink_keys, prop_fw.data)
        gamma1_ls = _matrix_stack(self.Gamma1, gamma1_keys, prop_fw.data)
        gamma2_ls = _matrix_stack(self.Gamma2, gamma2_keys, prop_fw.data)

        Gw = prop_fw.lexico(False)
        Gw_bperp_dagger = prop_bw_src.lexico(False)
        Gw_dagger = prop_sink_fw.lexico(False)
        Gw_bperp = self.apply_phase(prop_sink_bw, [-2 * pion_mom[0], -2 * pion_mom[1], -2 * pion_mom[2]], 1).lexico(False)
        Gw_dagger_conj = Gw_dagger.conj()

        shape = (len(src_keys), len(gamma1_keys), len(self.bT_dir), self.bT_length + 1, latt_info.global_size[3])
        corr_collect = np.empty(shape, dtype=np.complex128) if latt_info.mpi_rank == 0 else None
        for idir, bT_dir in enumerate(self.bT_dir):
            for bT in range(self.bT_length + 1):
                axis = 3 - bT_dir
                Gw_bperp_shift = xp.roll(Gw_bperp, shift=bT, axis=axis)
                Gw_bperp_dagger_shift = xp.roll(Gw_bperp_dagger.conj(), shift=bT, axis=axis)
                tmp_1 = xp.einsum("tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc", Gw, src_ls, gamma5, Gw_bperp_dagger_shift, gamma5, optimize=True)
                tmp_2 = xp.einsum("tzyxjiba,sik,kl,tzyxmlca,mn->stzyxjnbc", Gw_bperp_shift, sink_ls, gamma5, Gw_dagger_conj, gamma5, optimize=True)
                for isrc in range(len(src_keys)):
                    for igm in range(len(gamma1_keys)):
                        mpi_print(latt_info, f"Contract pion soft factor bT={bT} dir={bT_dir} src={src_keys[isrc]} gamma={gamma1_keys[igm]}")
                        corr_local = xp.einsum(
                            "tzyxjiba,ik,tzyxklba,lj->tzyx",
                            tmp_1[isrc],
                            gamma2_ls[igm],
                            tmp_2[isrc],
                            gamma1_ls[igm],
                            optimize=True,
                        )
                        corr_t = xp.einsum("tzyx->t", corr_local, optimize=True)
                        corr_global = core.gatherLattice(_to_numpy(xp, corr_t), [0, -1, -1, -1])
                        if latt_info.mpi_rank == 0:
                            corr_collect[isrc, igm, idir, bT] = corr_global
        return corr_collect, src_keys, sink_keys, gamma1_keys, gamma2_keys
