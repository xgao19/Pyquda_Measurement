import pytest

from pyquda_measurement_utils.io_corr import (
    get_c2pt_file_tag,
    get_disconnected_qTMD_1pt_file_tag,
    get_emt_gluon_1pt_file_tag,
    get_emt_proton_quark_3pt_file_tag,
    get_flowed_quark_ringed_norm_file_tag,
    get_pion_EMFF_file_tag,
    get_pion_soft_factor_file_tag,
    get_pion_soft_factor_prop_file_tag,
    get_qTMD_file_tag,
    get_qTMDWF_file_tag,
    get_sample_log_tag,
)


def test_standard_measurement_tag_helpers_are_deterministic():
    src = [1, 2, 3, 4]

    assert get_sample_log_tag("ama", src, "sm") == "ama_x1y2z3t4_sm"
    assert get_c2pt_file_tag("/data", "lat", 7, "CG", src, "sm") == "/data/c2pt/lat.c2pt.7.CG.x1y2z3t4.sm"
    assert get_qTMD_file_tag("/data", "lat", 7, "GI", src, "sm") == "/data/qTMD/lat.qTMD.7.GI.x1y2z3t4.sm"
    assert get_qTMDWF_file_tag("/data", "lat", 7, "WF", src, "sm") == "/data/qTMDWF/lat.qTMDWF.7.WF.x1y2z3t4.sm"
    assert get_disconnected_qTMD_1pt_file_tag("/data", "lat", 7, "loop", src, "sm") == "/data/qTMD1pt/lat.qTMD1pt.7.loop.x1y2z3t4.sm"
    assert get_pion_EMFF_file_tag("/data", "lat", 7, "EMFF", src, "sm") == "/data/pion_EMFF/lat.pion_EMFF.7.EMFF.x1y2z3t4.sm"


def test_soft_factor_tag_helpers_preserve_forward_backward_momenta():
    src = [0, 0, 0, 5]
    fw = [0, 0, 2]
    bw = [0, 0, -2]

    assert get_pion_soft_factor_file_tag("/data", "lat", 8, "CG.wall", src, "sm", fw, bw).endswith(
        "/pion_soft_factor/lat.pion_soft_factor.8.CG.wall.x0y0z0t5.sm.fw_qx0qy0qz2.bw_qx0qy0qz-2"
    )
    assert get_pion_soft_factor_prop_file_tag("/data", "lat", 8, "CG.wall", src, "sm", fw).endswith(
        "/pion_soft_factor_prop/lat.pion_soft_factor_prop.8.CG.wall.x0y0z0t5.sm.qx0qy0qz2"
    )


def test_emt_tag_helpers_preserve_measurement_kind():
    src = [1, 0, 2, 3]

    assert get_emt_gluon_1pt_file_tag("/data", "lat", 9, "G", src, "sm") == "/data/EMTg/lat.EMTg.9.G.x1y0z2t3.sm"
    assert get_emt_proton_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", [1, -2, 3, 0], 9
    ) == "/data/EMTproton3pt/lat.EMTproton3pt.9.Q.x1y0z2t3.sm.PX1PY-2PZ3dt9"
    assert get_emt_proton_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", [1, -2, 3, 0], [9]
    ) == "/data/EMTproton3pt/lat.EMTproton3pt.9.Q.x1y0z2t3.sm.PX1PY-2PZ3dt9"
    with pytest.raises(ValueError):
        get_emt_proton_quark_3pt_file_tag("/data", "lat", 9, "Q", src, "sm", [0, 0, 0, 0], [6, 9, 12])
    assert get_flowed_quark_ringed_norm_file_tag("/data", "lat", 9, "R", src, "sm") == "/data/FlowedQuarkRinged/lat.FlowedQuarkRinged.9.R.x1y0z2t3.sm"
