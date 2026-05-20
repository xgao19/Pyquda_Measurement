"""
Proton connected qTMD and PDF contractions in PyQUDA.

This module implements the connected proton two-point function and the
connected fixed-sink sequential-source contractions used by the nucleon qTMD
applications.  It is intentionally baryon-specific: unlike the pion code, the
proton contains three quark lines and therefore needs a color-antisymmetric
epsilon contraction, flavor-dependent sequential sources, and spin projection
matrices.

Gamma and interpolator conventions
----------------------------------
The 16 bilinear operator insertions are ordered as

    5, T, T5, X, X5, Y, Y5, Z, Z5, I, SXT, SXY, SXZ, SYT, SYZ, SZT.

The proton interpolating operator is schematically

    chi_alpha(x) =
        epsilon_abc
        [u_a^T(x) C Gamma_interp d_b(x)] u_{c, alpha}(x),

with supported interpolation choices currently mapped as

    "5"  -> C gamma5,
    "T5" -> C gamma_t gamma5,
    "Z5" -> C gamma_z gamma5.

The spin/polarization projection is supplied by ``PolProjections`` in
``bw_seq_pyquda.py`` when the backward sequential source is built.

Two-point function
------------------
For a sink bilinear Gamma_g and source/sink interpolator Gamma_interp, the
proton two-point function has the schematic form

    C2_g(p, t) =
        sum_x exp(-i p . (x - x0))
        P_{alpha alpha'}
        < chi_alpha(x) Gamma_g(t) chi_bar_{alpha'}(x0) >.

After Wick contraction this becomes the standard pair of baryon contraction
terms built from three quark propagators and two color epsilon tensors.  In
``contract_2pt_TMD`` the two terms are split into smaller einsums for memory
reasons, but algebraically they correspond to

    epsilon_abc epsilon_def
    Gamma_interp Gamma_interp
    S_u S_d S_u

with the required exchange term from the two identical up quarks.  The code
currently assumes isospin-symmetric propagators, so the same forward propagator
is reused for all three valence lines unless the caller provides otherwise.

Fixed-sink sequential sources
-----------------------------
The three-point functions are evaluated with fixed-sink sequential propagators
created by ``create_bw_seq_pyquda``.  Separate sequential sources are needed for
flavor insertions:

    flavor = 1: insertion on an up-quark line,
    flavor = 2: insertion on the down-quark line.

The application contracts both sequential propagators with the shifted forward
line.  This gives connected diagrams only.  Disconnected diagrams are not
included in this module.

Connected qTMD three-point function
-----------------------------------
The connected qTMD contraction used in the application is schematically

    C3_g^flavor(q, b, tau; pf, tsep) =
        sum_x exp(-i q . (x - x0))
        Tr_spin,color[
            Seq_flavor(x; pf, tsep, P)
            Gamma_g
            O_b S_q(x, x0)
        ],

where ``Seq_flavor`` already contains the baryon sink contraction, the final
state momentum pf, the sink time tsep, and the chosen spin projection P.  The
momentum transfer is q = pf - pi in the usual three-point convention.

CG qTMD operator
----------------
For the CG qTMD path, the nonlocal operator O_b is implemented as a coordinate
shift of the forward propagator without explicit gauge links:

    O_b S_q(x, x0) = S_q(x + bT * e_perp + bz * ez, x0).

The transverse direction is scanned over x and y and saved as ``b_X`` and
``b_Y``.  Wilson-line indices are ordered to keep successive shifts small.

PDF operators
-------------
The PDF path is the bT = 0 straight-z special case.  The proton application has
two variants:

    CG_PDF:  ordinary lattice shift, no explicit gauge link.
    GI_PDF:  covariant z-direction shift using gauge.pure_gauge.covDev.

The GI version represents a gauge-invariant straight Wilson line through
successive gauge-covariant nearest-neighbor shifts in +z or -z.  The helper
raises an error for non-nearest-neighbor jumps, which protects against
accidentally skipping links.

Array and output shape conventions
----------------------------------
The application stores connected qTMD/PDF data with the writer convention

    [Wilson_index, polarization, momentum, gamma, time]

for proton correlators before selecting one polarization/flavor/gamma for each
HDF5 file.  Pion correlators use a reduced shape because there is no flavor or
polarization dimension.

Important limitations
---------------------
This code is for connected proton diagrams only.  It does not compute
disconnected insertions, flavor mixing, or operator renormalization factors.
Those should be added as separate, explicit workflows rather than hidden inside
the current connected contraction path.
"""

from pyquda_utils import core, gamma
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    create_fermion_TMD_GI,
    create_fermion_TMD_GI_from_link,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import create_gi_qtmd_wilsonline_index_lists
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.tools import _get_xp_from_array, mpi_print, _asarray_on_queue


my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#! Add PyQUDA gamma matrices by order
my_pyquda_gammas = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]

Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)
CgT5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7)
CgZ5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11)


"""
================================================================================
                                proton_TMD
================================================================================
"""
class proton_TMD():
    def __init__(self, parameters):

        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T

        self.pf = parameters["pf"] # momentum of final nucleon state; pf = pi + q
        self.plist = parameters["qext"]
        self.qlist = parameters["qext_PDF"]
        #self.plist = [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in parameters["qext"] for y in parameters["qext"] for z in [0]}]
        #self.plist = [[x,y,z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating momentum transfers for TMD
        #self.qlist = [[x,y,z,0] for x in parameters["qext_PDF"] for y in parameters["qext_PDF"] for z in parameters["qext_PDF"]] # generating momentum transfers for PDF
        #self.pilist = [[parameters["pf"][0]-x,parameters["pf"][1]-y,parameters["pf"][2]-z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating pi = pf - q
        self.pilist = parameters["p_2pt"]  # 2pt momentum

        self.width = parameters["width"] # Gaussian smearing width
        self.boost_in = parameters["boost_in"] # ?? Forward propagator boost smearing
        self.boost_out = parameters["boost_out"] # ?? Backward propagator boost smearing
        self.pos_boost = self.boost_in # Forward propagator boost smearing for 2pt

        self.pol_list = parameters["pol"] # projection of nucleon state
        self.t_insert = parameters["t_insert"] # time separation of three point function
        self.save_propagators = parameters["save_propagators"] # if save propagators
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_TMD(self, latt_info, prop_f, phases, tag, interpolator = "5"): 
        if interpolator == "5":
            gamma_insert = Cg5
        elif interpolator == "T5":
            gamma_insert = CgT5
        elif interpolator == "Z5":
            gamma_insert = CgZ5
        else:
            raise ValueError(f"Invalid interpolator: {interpolator}")
        
        
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        mpi_print(latt_info, "Sink smearing completed")
        
        xp = _get_xp_from_array(prop_f.data)
        #* IMPORTANT: keep complex dtype.
        # Some gamma matrices (e.g. Z / Z5 in the QUDA bitmask basis) can be purely imaginary.
        # If we allocate float here, imaginary parts get dropped and those gammas become identically 0,
        # which makes the corresponding c2pt channels exactly zero.
        P_2pt_gamma_host = xp.zeros((16, latt_info.Lt, 4, 4), dtype=prop_f.data.dtype)
        P_2pt_gamma = _asarray_on_queue(P_2pt_gamma_host, xp, prop_f.data)

        for gamma_idx, gamma_pyq_host in enumerate(my_pyquda_gammas):
            gamma_device = _asarray_on_queue(gamma_pyq_host, xp, prop_f.data)
            
            P_2pt_local = _asarray_on_queue(xp.zeros((latt_info.Lt, 4, 4), dtype=prop_f.data.dtype), xp, prop_f.data)
            P_2pt_local[:] = gamma_device
            P_2pt_gamma[gamma_idx] = P_2pt_local
            
        epsilon_host = xp.zeros((3,3,3), dtype=prop_f.data.real.dtype)
        for a in range (3):
            b = (a+1) % 3
            c = (a+2) % 3
            epsilon_host[a,b,c] = 1
            epsilon_host[a,c,b] = -1
        epsilon = _asarray_on_queue(epsilon_host, xp, prop_f.data)
        
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        gamma_insert = _asarray_on_queue(gamma_insert, xp, prop_f.data)
        
        #! Optimized version of the 2pt TMD contraction (memory-friendly)
        #! Strategy: split large einsum into small 2-tensor / 3-tensor contractions

        # ============================================================
        # --- Term 1 ---
        # original:
        # - epsilon(abc)*epsilon(def)*G(ij)*G(kl)*P2pt(gtmn)*P1(ikad)*P2(jlbe)*P3(mncf)*Phases
        # ============================================================

        # -----------------------------
        # 1) Sink block (split)
        # term1_sink = einsum("abc, ij, ...ikad, ...jlbe -> ...klcde")
        # -----------------------------

        # (a) contract epsilon(abc) with first propagator P1(i,k,a,d)
        #     "abc, ...ikad -> ...ikbcd"
        t1_s1 = xp.einsum(
            "abc, wtzyxikad -> wtzyxikbcd",
            epsilon, prop_f.data,
            optimize=True
        )

        # (b) contract gamma_insert(ij) with second propagator P2(j,l,b,e)
        #     "ij, ...jlbe -> ...ilbe"
        t1_s2 = xp.einsum(
            "ij, wtzyxjlbe -> wtzyxilbe",
            gamma_insert, prop_f.data,
            optimize=True
        )

        # (c) combine the two partial sink blocks, contract i and b
        #     "...ikbcd, ...ilbe -> ...klcde"
        term1_sink = xp.einsum(
            "wtzyxikbcd, wtzyxilbe -> wtzyxklcde",
            t1_s1, t1_s2,
            optimize=True
        )

        del t1_s1, t1_s2


        # -----------------------------
        # 2) P3 block (already 2 tensors)
        # term1_p3 = einsum("gtmn, ...mncf -> g...cf")
        # -----------------------------
        term1_p3 = xp.einsum(
            "gtmn, wtzyxmncf -> gwtzyxcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )


        # -----------------------------
        # 3) Final assembly (split)
        # original:
        # term1 = einsum("def, pwtzyx, kl, ...klcde, g...cf -> gpt")
        # -----------------------------

        # (a) contract epsilon(def) with term1_sink on d,e
        #     "def, ...klcde -> ...klcf"
        t1_f1 = xp.einsum(
            "def, wtzyxklcde -> wtzyxklcf",
            epsilon, term1_sink,
            optimize=True
        )
        del term1_sink

        # (b) contract gamma_insert(k,l)
        #     "kl, ...klcf -> ...cf"
        t1_f2 = xp.einsum(
            "kl, wtzyxklcf -> wtzyxcf",
            gamma_insert, t1_f1,
            optimize=True
        )
        del t1_f1

        # (c) contract with P3 block on c,f
        #     "...cf, g...cf -> g..."
        t1_f3 = xp.einsum(
            "wtzyxcf, gwtzyxcf -> gwtzyx",
            t1_f2, term1_p3,
            optimize=True
        )
        del t1_f2, term1_p3

        # (d) contract phases
        #     "p..., g... -> gpt"
        term1 = xp.einsum(
            "pwtzyx, gwtzyx -> gpt",
            phases, t1_f3,
            optimize=True
        )
        del t1_f3


        # ============================================================
        # --- Term 2 ---
        # original:
        # - epsilon(abc)*epsilon(def)*G(ij)*G(kl)*P2pt(gtmn)*P1(ikad)*P2(jnbe)*P3(mlcf)*Phases
        # ============================================================

        # -----------------------------
        # 1) Sink block (split)
        # term2_sink = einsum("abc, ij, ...ikad, ...jnbe -> ...kncde")
        # -----------------------------

        # (a) epsilon with P1
        #     "abc, ...ikad -> ...ikbcd"
        t2_s1 = xp.einsum(
            "abc, wtzyxikad -> wtzyxikbcd",
            epsilon, prop_f.data,
            optimize=True
        )

        # (b) gamma_insert with P2(j,n,b,e)
        #     "ij, ...jnbe -> ...inbe"
        t2_s2 = xp.einsum(
            "ij, wtzyxjnbe -> wtzyxinbe",
            gamma_insert, prop_f.data,
            optimize=True
        )

        # (c) combine, contract i and b
        #     "...ikbcd, ...inbe -> ...kncde"
        term2_sink = xp.einsum(
            "wtzyxikbcd, wtzyxinbe -> wtzyxkncde",
            t2_s1, t2_s2,
            optimize=True
        )

        del t2_s1, t2_s2


        # -----------------------------
        # 2) P3 block (already 2 tensors)
        # term2_p3 = einsum("gtmn, ...mlcf -> g...nlcf")
        # -----------------------------
        term2_p3 = xp.einsum(
            "gtmn, wtzyxmlcf -> gwtzyxnlcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )


        # -----------------------------
        # 3) Final assembly (split)
        # original:
        # term2 = einsum("def, pwtzyx, kl, ...kncde, g...nlcf -> gpt")
        # -----------------------------

        # (a) contract epsilon(def) with term2_sink on d,e
        #     "def, ...kncde -> ...kncf"
        t2_f1 = xp.einsum(
            "def, wtzyxkncde -> wtzyxkncf",
            epsilon, term2_sink,
            optimize=True
        )
        del term2_sink

        # (b) contract sink-part and P3-part on (n,c,f), keep (g,k,l,w,t,z,y,x)
        #     "...kncf, g...nlcf -> g...kl"
        t2_f2 = xp.einsum(
            "wtzyxkncf, gwtzyxnlcf -> gwtzyxkl",
            t2_f1, term2_p3,
            optimize=True
        )
        del t2_f1, term2_p3

        # (c) contract gamma_insert(k,l)
        #     "kl, g...kl -> g..."
        t2_f3 = xp.einsum(
            "kl, gwtzyxkl -> gwtzyx",
            gamma_insert, t2_f2,
            optimize=True
        )
        del t2_f2

        # (d) contract phases
        term2 = xp.einsum(
            "pwtzyx, gwtzyx -> gpt",
            phases, t2_f3,
            optimize=True
        )
        del t2_f3

        # --- Final Result ---
        # original code is (- Einsum1 - Einsum2)
        corr = - term1 - term2

        corr_collect = core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])
        
        
        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(corr_collect, tag, my_gammas, self.pilist)
        result = corr_collect
        del corr
        return result
    
        
    def create_TMD_Wilsonline_index_list_CG(self):
        index_list_trans0 = []
        index_list_trans1 = []
        
        for current_bz in range(0, self.b_z+1):
            for current_b_T in range(0, self.b_T+1):
                # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])
                
        # Reorder index lists to minimize differences between adjacent indices
        def reorder_indices(index_list):
            # Sort by bT first, then bz to minimize jumps
            sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
            reordered = []
            
            # Process pairs of indices to minimize differences
            i = 0
            while i < len(sorted_list)-1:
                curr = sorted_list[i]
                next = sorted_list[i+1]
                
                # If difference is more than 1 in either bT or bz, try to find better match
                if abs(curr[0] - next[0]) > 1 or abs(curr[1] - next[1]) > 1:
                    # Look ahead for better match
                    best_match = next
                    best_diff = max(abs(curr[0] - next[0]), abs(curr[1] - next[1]))
                    
                    for j in range(i+2, len(sorted_list)):
                        candidate = sorted_list[j]
                        diff = max(abs(curr[0] - candidate[0]), abs(curr[1] - candidate[1]))
                        if diff < best_diff:
                            best_match = candidate
                            best_diff = diff
                    
                    # Swap to get better ordering
                    if best_match != next:
                        idx = sorted_list.index(best_match)
                        sorted_list[i+1], sorted_list[idx] = sorted_list[idx], sorted_list[i+1]
                
                reordered.append(curr)
                i += 1
                
            if i < len(sorted_list):
                reordered.append(sorted_list[-1])
                
            return reordered
            
        index_list_trans0 = reorder_indices(index_list_trans0)
        index_list_trans1 = reorder_indices(index_list_trans1)
                
        return index_list_trans0, index_list_trans1

    def create_TMD_Wilsonline_index_list_GI(self):
        return create_gi_qtmd_wilsonline_index_lists(self.eta, self.b_z, self.b_T)
                    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_TMD_CG(self, prop_f, W_index, WL_indices_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3] # 0, 1
        Zdir = 2
        
        previous_b_T = WL_indices_previous[0]
        previous_bz = WL_indices_previous[1]
        
        prop_shift = prop_f.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), Zdir)

        return prop_shift

    def create_fw_prop_TMD_GI(self, gauge, prop_f, W_index, staple_links=None):
        prop_shift = prop_f.copy()
        staple_link = None if staple_links is None else staple_links[tuple(W_index)]

        for spin in range(4):
            for color in range(3):
                fermion = prop_f.getFermion(spin, color)
                if staple_link is None:
                    fermion_shift = create_fermion_TMD_GI(gauge, fermion, W_index)
                else:
                    fermion_shift = create_fermion_TMD_GI_from_link(staple_link, fermion, W_index)
                prop_shift.setFermion(fermion_shift, spin, color)

        return prop_shift
    
    def create_PDF_Wilsonline_index_list(self):
        index_list = []
        
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
            index_list.append([0, current_bz, 0, 0])
            
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])
                    
        return index_list
    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_PDF_GI(self, gauge, prop_f_pyq, W_index, WL_indices_previous):

        current_bz = W_index[1]
        previous_bz = WL_indices_previous[1]

        #! PyQUDA: forward prop
        for spin in range(4):
            for color in range(3):
                fermion = prop_f_pyq.getFermion(spin, color)
                if current_bz - previous_bz == 0:
                    fermion_shift = fermion
                elif current_bz - previous_bz == 1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                elif current_bz - previous_bz == -1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 6) # -z direction
                else:
                    raise ValueError("Invalid shift for PDF Wilson line")
                #\psi'(x)=U_\mu(x)\psi(x+\hat\mu)0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
                prop_f_pyq.setFermion(fermion_shift, spin, color)

        return prop_f_pyq
