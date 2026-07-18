"""Shared qTMD/PDF displacement and Wilson-line transport utilities."""

from pyquda.field import LatticeFermion, LatticeGauge, LatticeLink
from pyquda_utils.convert import fermionToLink, linkToFermion

from pyquda_measurement_utils.tools import _get_xp_from_array


def _reorder_wilson_indices(index_list):
    """Preserve the existing nearest-neighbor greedy execution order."""
    sorted_list = sorted(index_list, key=lambda item: (item[0], item[1]))
    reordered = []
    index = 0
    while index < len(sorted_list) - 1:
        current = sorted_list[index]
        next_index = sorted_list[index + 1]
        if (
            abs(current[0] - next_index[0]) > 1
            or abs(current[1] - next_index[1]) > 1
        ):
            best_match = next_index
            best_diff = max(
                abs(current[0] - next_index[0]),
                abs(current[1] - next_index[1]),
            )
            for candidate in sorted_list[index + 2 :]:
                difference = max(
                    abs(current[0] - candidate[0]),
                    abs(current[1] - candidate[1]),
                )
                if difference < best_diff:
                    best_match = candidate
                    best_diff = difference
            if best_match != next_index:
                best_index = sorted_list.index(best_match)
                sorted_list[index + 1], sorted_list[best_index] = (
                    sorted_list[best_index],
                    sorted_list[index + 1],
                )
        reordered.append(current)
        index += 1

    if index < len(sorted_list):
        reordered.append(sorted_list[-1])
    return reordered


def create_cg_qtmd_wilsonline_index_lists(max_b_z, max_b_T):
    """Create the current CG qTMD execution lists for transverse x/y."""
    index_list_trans0 = []
    index_list_trans1 = []
    for current_bz in range(0, int(max_b_z) + 1):
        for current_b_T in range(0, int(max_b_T) + 1):
            index_list_trans0.append([current_b_T, current_bz, 0, 0])
            index_list_trans1.append([current_b_T, current_bz, 0, 1])
            if current_bz != 0:
                index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                index_list_trans1.append([current_b_T, -current_bz, 0, 1])
    return (
        _reorder_wilson_indices(index_list_trans0),
        _reorder_wilson_indices(index_list_trans1),
    )


def create_gi_qtmd_wilsonline_index_lists(eta_list, max_b_z, max_b_T):
    """Create fixed-length GI qTMD Wilson-index lists for transverse x/y."""
    index_list_trans0 = []
    index_list_trans1 = []
    for eta in eta_list:
        eta = int(eta)
        for current_bz in range(0, int(max_b_z) + 1, 2):
            if eta < current_bz // 2:
                continue
            for current_b_T in range(0, int(max_b_T) + 1):
                index_list_trans0.append([current_b_T, current_bz, eta, 0])
                index_list_trans1.append([current_b_T, current_bz, eta, 1])
                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, eta, 0])
                    index_list_trans1.append([current_b_T, -current_bz, eta, 1])
    return index_list_trans0, index_list_trans1


def create_pdf_wilsonline_index_list(max_b_z):
    """Create the canonical straight-z PDF displacement list."""
    indices = [[0, current_bz, 0, 0] for current_bz in range(0, int(max_b_z) + 1)]
    indices.extend(
        [0, -current_bz, 0, 0]
        for current_bz in range(1, int(max_b_z) + 1)
    )
    return indices


def shift_qtmd_cg(field, W_index, W_index_previous):
    """Apply the existing incremental coordinate-gauge qTMD displacement."""
    current_b_T, current_bz, _eta, transverse_direction = W_index
    previous_b_T, previous_bz = W_index_previous[:2]
    return field.shift(
        round(current_b_T - previous_b_T), transverse_direction
    ).shift(round(current_bz - previous_bz), 2)


def shift_fermion_pdf_gi(gauge, fermion, W_index, W_index_previous):
    """Apply one incremental gauge-covariant straight-z displacement."""
    delta_bz = W_index[1] - W_index_previous[1]
    if delta_bz == 0:
        return fermion
    if delta_bz == 1:
        return gauge.pure_gauge.covDev(fermion, 2)
    if delta_bz == -1:
        return gauge.pure_gauge.covDev(fermion, 6)
    raise ValueError("Invalid shift for PDF Wilson line")


def shift_propagator_pdf_gi(gauge, propagator, W_index, W_index_previous):
    """Apply one incremental straight-z displacement to a propagator."""
    for spin in range(4):
        for color in range(3):
            fermion = propagator.getFermion(spin, color)
            shifted = shift_fermion_pdf_gi(
                gauge, fermion, W_index, W_index_previous
            )
            propagator.setFermion(shifted, spin, color)
    return propagator


def gi_qtmd_staple_segments(W_index):
    """Return signed nearest-neighbor segments for a fixed-length staple."""
    b_T, b_z, eta, transverse_direction = [
        int(round(value)) for value in W_index
    ]
    if b_T < 0:
        raise ValueError("GI_qTMD requires non-negative b_T")
    if b_z % 2 != 0:
        raise ValueError(
            "GI_qTMD requires even b_z in the fixed-staple-length convention"
        )
    if eta < 0:
        raise ValueError("GI_qTMD requires non-negative eta")
    if eta < abs(b_z) // 2:
        raise ValueError("GI_qTMD requires eta >= abs(b_z) / 2")
    if transverse_direction not in {0, 1}:
        raise ValueError("GI_qTMD transverse_direction should be 0 or 1")

    half_bz = b_z // 2
    return [
        (2, eta + half_bz),
        (transverse_direction, b_T),
        (2, half_bz - eta),
    ]


def _apply_signed_covariant_shift(gauge, fermion, direction, steps):
    shifted = fermion
    if steps > 0:
        for _ in range(steps):
            shifted = gauge.pure_gauge.covDev(shifted, direction)
    elif steps < 0:
        for _ in range(-steps):
            shifted = gauge.pure_gauge.covDev(shifted, direction + 4)
    return shifted


def _transport_staple_field(gauge, fermion, W_index):
    """Transport the endpoint field in reverse geometric path order."""
    shifted = fermion.copy()
    for direction, steps in reversed(gi_qtmd_staple_segments(W_index)):
        shifted = _apply_signed_covariant_shift(
            gauge, shifted, direction, steps
        )
    return shifted


def build_gi_qtmd_staple_link(gauge: LatticeGauge, W_index):
    """Build a gauge-only staple transporter matching direct covDev."""
    link = LatticeLink(gauge.latt_info)
    transported = _transport_staple_field(
        gauge, linkToFermion(link), W_index
    )
    return fermionToLink(transported)


def build_gi_qtmd_staple_links(gauge: LatticeGauge, W_index_list):
    """Build reusable gauge-only staple transporters."""
    return {
        tuple(W_index): build_gi_qtmd_staple_link(gauge, W_index)
        for W_index in W_index_list
    }


def apply_gi_qtmd_staple_to_fermion(
    staple_link: LatticeLink, fermion: LatticeFermion, W_index
):
    """Apply a cached GI qTMD staple transporter to a fermion."""
    b_T, b_z, _eta, transverse_direction = [
        int(round(value)) for value in W_index
    ]
    endpoint = fermion.shift(b_T, transverse_direction).shift(b_z, 2)
    shifted = LatticeFermion(fermion.latt_info)
    xp = _get_xp_from_array(fermion.data)
    shifted.data[:] = xp.einsum(
        "wtzyxab,wtzyxib->wtzyxia",
        staple_link.data,
        endpoint.data,
        optimize=True,
    )
    return shifted


def apply_gi_qtmd_staple_to_propagator(
    propagator, W_index, staple_links
):
    """Apply a cached GI qTMD staple transporter to a propagator."""
    shifted = propagator.copy()
    staple_link = staple_links[tuple(W_index)]
    for spin in range(4):
        for color in range(3):
            fermion = propagator.getFermion(spin, color)
            shifted_fermion = apply_gi_qtmd_staple_to_fermion(
                staple_link, fermion, W_index
            )
            shifted.setFermion(shifted_fermion, spin, color)
    return shifted


__all__ = [
    "apply_gi_qtmd_staple_to_fermion",
    "apply_gi_qtmd_staple_to_propagator",
    "build_gi_qtmd_staple_link",
    "build_gi_qtmd_staple_links",
    "create_cg_qtmd_wilsonline_index_lists",
    "create_gi_qtmd_wilsonline_index_lists",
    "create_pdf_wilsonline_index_list",
    "gi_qtmd_staple_segments",
    "shift_fermion_pdf_gi",
    "shift_propagator_pdf_gi",
    "shift_qtmd_cg",
]
