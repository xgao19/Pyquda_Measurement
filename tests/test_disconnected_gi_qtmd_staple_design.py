from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import gi_qtmd_staple_segments


def test_gi_qtmd_staple_local_limit():
    assert gi_qtmd_staple_segments([0, 0, 0, 0]) == [(2, 0), (0, 0), (2, 0)]
    assert gi_qtmd_staple_segments([0, 0, 0, 1]) == [(2, 0), (1, 0), (2, 0)]


def test_gi_qtmd_staple_straight_pdf_limit():
    assert gi_qtmd_staple_segments([0, 3, 3, 0]) == [(2, 3), (0, 0), (2, 0)]
    assert gi_qtmd_staple_segments([0, -2, -2, 1]) == [(2, -2), (1, 0), (2, 0)]


def test_gi_qtmd_staple_three_segment_path():
    assert gi_qtmd_staple_segments([2, 1, 4, 0]) == [(2, 4), (0, 2), (2, -3)]
    assert gi_qtmd_staple_segments([3, -1, 2, 1]) == [(2, 2), (1, 3), (2, -3)]


if __name__ == "__main__":
    test_gi_qtmd_staple_local_limit()
    test_gi_qtmd_staple_straight_pdf_limit()
    test_gi_qtmd_staple_three_segment_path()
    print("[GI qTMD staple design sanity check]")
    print("path = z(eta), transverse(b_T), z(b_z - eta)")
