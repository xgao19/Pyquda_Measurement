import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import create_gi_qtmd_wilsonline_index_lists


def _pdf_index_list(max_bz):
    return [[0, bz, 0, 0] for bz in range(0, max_bz + 1)] + [[0, bz, 0, 0] for bz in range(-1, -max_bz - 1, -1)]


def test_connected_gi_qtmd_contains_local_and_straight_pdf_limit_indices():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([0, 1, 2], 4, 0)
    gi_indices = dir0 + dir1
    pdf_indices = _pdf_index_list(4)

    assert [0, 0, 0, 0] in gi_indices
    assert [0, 0, 0, 1] in gi_indices

    for _, bz, _, _ in pdf_indices:
        eta = abs(bz) // 2
        if bz % 2 == 0:
            assert [0, bz, eta, 0] in gi_indices
            assert [0, bz, eta, 1] in gi_indices


def test_connected_gi_qtmd_rejects_unphysical_fixed_length_staples():
    dir0, dir1 = create_gi_qtmd_wilsonline_index_lists([0, 1, 2], 4, 1)
    gi_indices = dir0 + dir1

    assert [1, 4, 1, 0] not in gi_indices
    assert [1, -4, 1, 1] not in gi_indices
    assert [1, 4, 2, 0] in gi_indices
    assert [1, -4, 2, 1] in gi_indices
