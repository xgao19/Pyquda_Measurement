from pyquda_measurement_utils.qtmd_operator_utils import (
    create_gi_qtmd_wilsonline_index_lists,
    gi_qtmd_staple_segments,
)


def _path_length(segments):
    return sum(abs(steps) for _direction, steps in segments)


def test_gi_qtmd_local_limit_has_zero_length_staple_for_both_transverse_directions():
    for direction in (0, 1):
        segments = gi_qtmd_staple_segments([0, 0, 0, direction])
        assert segments == [(2, 0), (direction, 0), (2, 0)]
        assert _path_length(segments) == 0


def test_gi_qtmd_even_bz_pdf_limit_reduces_to_straight_z_link():
    for bz in (-4, -2, 2, 4):
        eta = abs(bz) // 2
        segments = gi_qtmd_staple_segments([0, bz, eta, 0])
        assert all(direction == 2 or steps == 0 for direction, steps in segments)
        assert sum(steps for _direction, steps in segments) == bz
        assert _path_length(segments) == abs(bz)


def test_connected_and_disconnected_gi_index_builder_contains_same_local_and_pdf_limits():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([0, 1, 2], 4, 1)
    gi_indices = {tuple(idx) for idx in dir0 + dir1}

    for direction in (0, 1):
        assert (0, 0, 0, direction) in gi_indices
        for bz in (-4, -2, 2, 4):
            eta = abs(bz) // 2
            assert (0, bz, eta, direction) in gi_indices
            assert (1, bz, eta, direction) in gi_indices
