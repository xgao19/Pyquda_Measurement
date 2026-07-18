"""Connected proton C2 and nonlocal-line helpers for qTMD/PDF production."""

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
)
from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    create_fermion_TMD_GI_from_link,
)
from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import create_gi_qtmd_wilsonline_index_lists
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.proton_utils_vibe_develop import contract_proton_c2

class proton_TMD():
    def __init__(self, parameters):
        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T
        self.pilist = parameters["p_2pt"]  # 2pt momentum
        self.width = parameters["width"] # Gaussian smearing width
        self.boost_out = parameters["boost_out"] # Sink-propagator boost smearing
        
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
            smearing_boost=self.boost_out,
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

    def create_fw_prop_TMD_GI(self, prop_f, W_index, staple_links):
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
