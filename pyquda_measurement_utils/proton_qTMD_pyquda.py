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

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PYQUDA_GAMMA_IDS,
)
from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    create_fermion_TMD_GI_from_link,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import create_gi_qtmd_wilsonline_index_lists
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.proton_utils_vibe_develop import contract_proton_c2


my_gammas = list(GAMMA_LABELS)
pyquda_gammas_order = list(PYQUDA_GAMMA_IDS)


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
    def contract_2pt_TMD(
        self, latt_info, prop_f, phases, tag, interpolator="5", attrs=None
    ):
        """Contract and write proton C2 through the shared calculation kernel."""
        corr_collect = contract_proton_c2(
            latt_info,
            prop_f,
            phases,
            interpolator=interpolator,
            sink_smearing=True,
            smearing_width=self.width,
            smearing_boost=self.pos_boost,
        )
        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(
                corr_collect, tag, list(GAMMA_LABELS), self.pilist, attrs=attrs
            )
        return corr_collect

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

    def create_fw_prop_TMD_GI(self, gauge, prop_f, W_index, staple_links):
        prop_shift = prop_f.copy()
        staple_link = staple_links[tuple(W_index)]

        for spin in range(4):
            for color in range(3):
                fermion = prop_f.getFermion(spin, color)
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
