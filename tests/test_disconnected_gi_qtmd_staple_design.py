from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import gi_qtmd_staple_segments


def _path_length(segments):
    return sum(abs(steps) for _direction, steps in segments)


def test_gi_qtmd_staple_local_limit():
    assert gi_qtmd_staple_segments([0, 0, 0, 0]) == [(2, 0), (0, 0), (2, 0)]
    assert gi_qtmd_staple_segments([0, 0, 0, 1]) == [(2, 0), (1, 0), (2, 0)]


def test_gi_qtmd_staple_straight_pdf_limit():
    assert gi_qtmd_staple_segments([0, 2, 1, 0]) == [(2, 2), (0, 0), (2, 0)]
    assert gi_qtmd_staple_segments([0, -2, 1, 1]) == [(2, 0), (1, 0), (2, -2)]


def test_gi_qtmd_staple_fixed_length_path():
    segments = gi_qtmd_staple_segments([3, 4, 3, 0])
    assert segments == [(2, 5), (0, 3), (2, -1)]
    assert _path_length(segments) == 2 * 3 + 3

    segments = gi_qtmd_staple_segments([2, -4, 3, 1])
    assert segments == [(2, 1), (1, 2), (2, -5)]
    assert _path_length(segments) == 2 * 3 + 2


def test_gi_qtmd_staple_invalid_indices():
    invalid_indices = [
        [0, 1, 1, 0],
        [0, 4, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 2],
    ]
    for W_index in invalid_indices:
        try:
            gi_qtmd_staple_segments(W_index)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for W_index={W_index}")


if __name__ == "__main__":
    test_gi_qtmd_staple_local_limit()
    test_gi_qtmd_staple_straight_pdf_limit()
    test_gi_qtmd_staple_fixed_length_path()
    test_gi_qtmd_staple_invalid_indices()
    print("[GI qTMD staple design sanity check]")
    print("path = z(eta + b_z/2), transverse(b_T), z(b_z/2 - eta)")
