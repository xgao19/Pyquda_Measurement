import pytest

from pyquda_measurement_utils.io_corr import (
    get_c2pt_file_tag,
    get_disconnected_qTMD_loop_file_tag,
    get_emt_gluon_loop_file_tag,
    get_emt_meson_2pt_file_tag,
    get_emt_quark_3pt_file_tag,
    get_emt_quark_loop_file_tag,
    get_emt_proton_quark_3pt_file_tag,
    get_interpolator_channel_tag,
    get_flowed_quark_ringed_norm_file_tag,
    get_pion_EMFF_file_tag,
    get_pion_channel_tag,
    get_pion_soft_factor_file_tag,
    get_pion_soft_factor_c2pt_file_tag,
    get_pion_soft_factor_qTMDWF_file_tag,
    get_pion_soft_factor_prop_file_tag,
    get_qTMD_file_tag,
    get_qTMDWF_file_tag,
    get_sample_log_tag,
    get_spatial_vector_tag,
)


def test_shared_channel_and_spatial_vector_tags_are_explicit():
    setup = "1HYP_GSRC_W9_binx0y0z0_boutx0y0z0"

    assert get_spatial_vector_tag([0, 0, 0]) == "x0y0z0"
    assert get_spatial_vector_tag([-1, 0, 2]) == "x-1y0z2"
    with pytest.raises(ValueError):
        get_spatial_vector_tag([0, 0])
    assert get_interpolator_channel_tag(setup, "5") == (
        f"{setup}.src5"
    )
    assert get_interpolator_channel_tag(setup, "5", "5") == (
        f"{setup}.src5.sink5"
    )


def test_pion_channel_tag_keeps_interpolators_outside_setup_identity():
    setup = "custom_smearing"

    assert get_pion_channel_tag(setup, "5") == (
        "custom_smearing.src5"
    )
    assert get_pion_channel_tag(setup, "5", "5") == (
        "custom_smearing.src5.sink5"
    )
    assert get_pion_channel_tag(setup, "5", "T5") != (
        get_pion_channel_tag(setup, "5", "5")
    )


def test_standard_measurement_tag_helpers_are_deterministic():
    src = [1, 2, 3, 4]

    assert get_sample_log_tag("ama", src, "sm") == "ama_x1y2z3t4_sm"
    assert get_c2pt_file_tag("/data", "lat", 7, "CG", src, "sm") == "/data/c2pt/lat.c2pt.7.CG.x1y2z3t4.sm"
    assert get_qTMD_file_tag("/data", "lat", 7, "GI", src, "sm") == "/data/qTMD/lat.qTMD.7.GI.x1y2z3t4.sm"
    assert get_qTMDWF_file_tag("/data", "lat", 7, "WF", src, "sm") == "/data/qTMDWF/lat.qTMDWF.7.WF.x1y2z3t4.sm"
    assert get_disconnected_qTMD_loop_file_tag("/data", "lat", 7, "loop", "sm") == "/data/qTMD1pt/lat.qTMD1pt.7.loop.sm"
    assert get_pion_EMFF_file_tag("/data", "lat", 7, "EMFF", src, "sm") == "/data/pion_EMFF/lat.pion_EMFF.7.EMFF.x1y2z3t4.sm"


def test_soft_factor_tags_use_wall_time_quark_set_cfg_and_single_tsep():
    tag = get_pion_soft_factor_file_tag("/data", "lat", 8, 0, 5, "sm", [5, 4], 6)
    assert tag == "/data/pion_soft_factor/8/lat.pion_soft_factor.8.0.wallt5.sm.qz4_5.dt6"
    assert get_pion_soft_factor_c2pt_file_tag("/data", "lat", 8, 0, 5, "sm", [4, 5]) == (
        "/data/pion_soft_factor_c2pt/8/lat.pion_soft_factor_c2pt.8.0.wallt5.sm.qz4_5"
    )
    assert get_pion_soft_factor_qTMDWF_file_tag("/data", "lat", 8, 0, 5, "sm", [4, 5]) == (
        "/data/pion_soft_factor_qTMDWF/8/lat.pion_soft_factor_qTMDWF.8.0.wallt5.sm.qz4_5"
    )
    for bad in ([4, 4], [], [4.5, 5]):
        with pytest.raises(ValueError):
            get_pion_soft_factor_file_tag("/data", "lat", 8, 0, 5, "sm", bad, 6)
    with pytest.raises(ValueError):
        get_pion_soft_factor_file_tag("/data", "lat", 8, 0, [0, 0, 0, 5], "sm", [4, 5], 6)
    with pytest.raises(ValueError):
        get_pion_soft_factor_file_tag("/data", "lat", 8, 0, 5, "sm", [4, 5], [6, 8])


def test_saved_prop_tag_is_independent_of_correlator_collection_names():
    assert get_pion_soft_factor_prop_file_tag("/data", "lat", 8, "CG.wall", [0, 0, 0, 5], "sm", [0, 0, 2]).endswith(
        "/pion_soft_factor_prop/lat.pion_soft_factor_prop.8.CG.wall.x0y0z0t5.sm.qx0qy0qz2"
    )


def test_emt_tag_helpers_preserve_measurement_kind():
    src = [1, 0, 2, 3]

    assert get_emt_gluon_loop_file_tag("/data", "lat", 9, "G", "sm") == "/data/EMTgluonLoop/lat.EMTgluonLoop.9.G.sm"
    assert get_emt_quark_loop_file_tag("/data", "lat", 9, "Q", "sm") == "/data/EMTquarkLoop/lat.EMTquarkLoop.9.Q.sm"
    assert get_emt_meson_2pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", "5"
    ) == "/data/EMT2pt/lat.EMT2pt.9.Q.x1y0z2t3.sm.src5"
    assert get_emt_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", "5", "T5"
    ) == "/data/EMT3pt/lat.EMT3pt.9.Q.x1y0z2t3.sm.src5.sinkT5"
    assert get_emt_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", "T", "5"
    ) != get_emt_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", "5", "T"
    )
    assert get_emt_proton_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", [1, -2, 3, 0], 9
    ) == "/data/EMTproton3pt/lat.EMTproton3pt.9.Q.x1y0z2t3.sm.PX1PY-2PZ3dt9"
    assert get_emt_proton_quark_3pt_file_tag(
        "/data", "lat", 9, "Q", src, "sm", [1, -2, 3, 0], [9]
    ) == "/data/EMTproton3pt/lat.EMTproton3pt.9.Q.x1y0z2t3.sm.PX1PY-2PZ3dt9"
    with pytest.raises(ValueError):
        get_emt_proton_quark_3pt_file_tag("/data", "lat", 9, "Q", src, "sm", [0, 0, 0, 0], [6, 9, 12])
    assert get_flowed_quark_ringed_norm_file_tag("/data", "lat", 9, "R", "sm") == "/data/FlowedQuarkRinged/lat.FlowedQuarkRinged.9.R.sm"
