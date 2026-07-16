import numpy as np

from pyquda_utils import core
from pyquda_measurement_utils.boosted_smearing_pyquda import boosted_smearing
from pyquda_measurement_utils.fermion_bilinear_basis import basis_attrs
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.pion_utils_vibe_develop import (
    _array_to_numpy,
    contract_pion_2pt,
    contract_pion_2pt_multi_src_gamma,
    gamma_stack,
    meson_backward_line,
    my_gammas,
    my_pyquda_gammas,
    pyquda_gammas_order,
    source_gamma_stack,
)
from pyquda_measurement_utils.tools import (
    _asarray_on_queue,
    _get_xp_from_array,
    mpi_print,
)


"""
================================================================================
                                proton_TMD
================================================================================
"""
class pion_TMDWF_measurement():
    def __init__(self, parameters):

        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_pion(
        self,
        latt_info,
        prop_f,
        prop_b,
        phases,
        tag,
        source_gamma_label="5",
        attrs=None,
    ):
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        prop_b = boosted_smearing(prop_b, w=self.width, boost=self.neg_boost)
        mpi_print(latt_info, "Sink smearing completed")

        corr = contract_pion_2pt(
            latt_info,
            prop_f,
            prop_b,
            phases,
            src_gamma=source_gamma_label,
        )
        if latt_info.mpi_rank == 0:
            common_attrs = dict(basis_attrs())
            if attrs:
                common_attrs.update(attrs)
            common_attrs.update({
                "source_gamma_mode": (
                    "fixed" if source_gamma_label in my_gammas else source_gamma_label
                ),
                "source_gamma_label": source_gamma_label,
                "sink_interpolator": "all_16_gamma_scan",
                "dataset_axes": "sink_gamma,momentum,time",
            })
            save_proton_c2pt_hdf5(
                corr,
                tag,
                my_gammas,
                self.plist,
                attrs=common_attrs,
                write_gamma_basis=True,
            )
        return corr

    def contract_2pt_pion_multi_src_gamma(
        self,
        latt_info,
        prop_f,
        prop_b,
        phases,
        tags_by_source,
        attrs=None,
    ):
        source_gamma_labels = list(tags_by_source)
        if not source_gamma_labels:
            raise ValueError("tags_by_source must contain at least one source Gamma label")
        invalid_labels = [label for label in source_gamma_labels if label not in my_gammas]
        if invalid_labels:
            raise ValueError(
                f"qTMDWF C2 requires explicit canonical source Gamma labels; got {invalid_labels}"
            )

        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        prop_b = boosted_smearing(prop_b, w=self.width, boost=self.neg_boost)
        mpi_print(latt_info, "Sink smearing completed")

        corr_by_source = contract_pion_2pt_multi_src_gamma(
            latt_info,
            prop_f,
            prop_b,
            phases,
            source_gamma_labels,
        )
        if latt_info.mpi_rank == 0:
            common_attrs = dict(basis_attrs())
            if attrs:
                common_attrs.update(attrs)
            common_attrs.update({
                "source_gamma_mode": "fixed",
                "sink_interpolator": "all_16_gamma_scan",
                "dataset_axes": "sink_gamma,momentum,time",
            })
            for source_gamma_label, corr in corr_by_source.items():
                source_attrs = dict(common_attrs)
                source_attrs["source_gamma_label"] = source_gamma_label
                save_proton_c2pt_hdf5(
                    corr,
                    tags_by_source[source_gamma_label],
                    my_gammas,
                    self.plist,
                    attrs=source_attrs,
                    write_gamma_basis=True,
                )
        return corr_by_source

    def contract_DA(
        self,
        latt_info,
        gauge,
        prop_f,
        prop_b,
        phases,
        W_index_list,
        source_gamma_labels,
        *,
        gauge_invariant,
    ):
        """Contract the straight-link DA operator on the forward quark line.

        The Wilson-index sequence must contain the positive branch followed by
        the negative branch.  The latter is restarted from the undisplaced
        forward propagator so every gauge-covariant update is one lattice step.
        """
        source_gamma_labels = list(source_gamma_labels)
        invalid_labels = [label for label in source_gamma_labels if label not in my_gammas]
        if not source_gamma_labels or invalid_labels:
            raise ValueError(
                "DA source Gamma labels must be a non-empty list of canonical "
                f"labels; invalid entries: {invalid_labels}"
            )
        if gauge_invariant and gauge is None:
            raise ValueError("GI DA contraction requires a gauge field")

        xp = _get_xp_from_array(prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        sink_gamma_ls = gamma_stack(prop_f.data)
        source_gamma_by_label = {
            label: source_gamma_stack(label, sink_gamma_ls, prop_f.data)[0]
            for label in source_gamma_labels
        }
        backward_line = meson_backward_line(prop_b)
        collected = {label: [] for label in source_gamma_labels}

        shifted_forward = prop_f.copy()
        previous_index = [0, 0, 0, 0]
        for W_index in W_index_list:
            if int(W_index[0]) != 0:
                raise ValueError("DA contraction requires b_T=0 Wilson indices")
            if int(W_index[1]) == -1:
                shifted_forward = prop_f.copy()
                previous_index = [0, 0, 0, 0]

            if gauge_invariant:
                shifted_forward = self.create_fw_prop_PDF_GI(
                    gauge, shifted_forward, W_index, previous_index
                )
            else:
                shifted_forward = self.create_fw_prop_TMD_CG(
                    shifted_forward, W_index, previous_index
                )

            for source_label, source_gamma in source_gamma_by_label.items():
                corr_local = xp.empty(
                    (phases.shape[0], len(sink_gamma_ls), latt_info.size[3]),
                    dtype=prop_f.data.dtype,
                    **(
                        {"device": prop_f.data.device}
                        if xp.__name__ == "dpnp"
                        else {}
                    ),
                )
                for gamma_idx, sink_gamma in enumerate(sink_gamma_ls):
                    sink_inserted = xp.einsum(
                        "wtzyxjicf,im->wtzyxjmcf",
                        backward_line,
                        sink_gamma,
                        optimize=True,
                    )
                    corr_site = xp.einsum(
                        "wtzyxjiab,wtzyxilba,lj->wtzyx",
                        sink_inserted,
                        shifted_forward.data,
                        source_gamma,
                        optimize=True,
                    )
                    corr_local[:, gamma_idx] = xp.einsum(
                        "qwtzyx,wtzyx->qt", phases, corr_site, optimize=True
                    )
                    del sink_inserted, corr_site

                corr = core.gatherLattice(
                    _array_to_numpy(corr_local), [2, -1, -1, -1]
                )
                if latt_info.mpi_rank == 0:
                    collected[source_label].append(corr)
                del corr_local
            previous_index = W_index

        del backward_line, shifted_forward
        if latt_info.mpi_rank == 0:
            return [
                (label, np.asarray(collected[label])) for label in source_gamma_labels
            ]
        return [(label, None) for label in source_gamma_labels]
    
        
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
    
    #! PyQUDA: one-step gauge-covariant straight-link transport in +-z
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
