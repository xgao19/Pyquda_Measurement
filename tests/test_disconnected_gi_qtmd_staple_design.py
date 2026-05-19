from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    DisconnectedQuarkqTMD1pt,
    create_fermion_TMD_GI,
    gi_qtmd_staple_segments,
)


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


def test_gi_qtmd_production_wilson_index_list():
    measurement = DisconnectedQuarkqTMD1pt(
        {
            "eta": [0, 1, 2],
            "b_z": 4,
            "b_T": 1,
            "qext": [[0, 0, 0, 0]],
        }
    )
    dir0, dir1 = measurement.create_TMD_Wilsonline_index_list_GI()

    assert [0, 0, 0, 0] in dir0
    assert [0, 2, 1, 0] in dir0
    assert [0, -2, 1, 0] in dir0
    assert [1, 4, 2, 0] in dir0
    assert [1, -4, 2, 0] in dir0
    assert [0, 4, 1, 0] not in dir0
    assert all(idx[3] == 0 for idx in dir0)
    assert all(idx[3] == 1 for idx in dir1)


class _FakeFermion:
    def __init__(self, path=None):
        self.path = list(path or [])

    def copy(self):
        return _FakeFermion(self.path)


class _FakePureGauge:
    def covDev(self, fermion, direction):
        return _FakeFermion(fermion.path + [direction])


class _FakeGauge:
    pure_gauge = _FakePureGauge()


def test_create_fermion_tmd_gi_applies_covariant_path():
    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [3, 4, 3, 0])
    assert shifted.path == [2, 2, 2, 2, 2, 0, 0, 0, 6]

    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [2, -4, 3, 1])
    assert shifted.path == [2, 1, 1, 6, 6, 6, 6, 6]


def test_create_fermion_tmd_gi_pdf_limit_path():
    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [0, 2, 1, 0])
    assert shifted.path == [2, 2]

    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [0, -2, 1, 1])
    assert shifted.path == [6, 6]


if __name__ == "__main__":
    test_gi_qtmd_staple_local_limit()
    test_gi_qtmd_staple_straight_pdf_limit()
    test_gi_qtmd_staple_fixed_length_path()
    test_gi_qtmd_staple_invalid_indices()
    test_gi_qtmd_production_wilson_index_list()
    test_create_fermion_tmd_gi_applies_covariant_path()
    test_create_fermion_tmd_gi_pdf_limit_path()
    print("[GI qTMD staple design sanity check]")
    print("path = z(eta + b_z/2), transverse(b_T), z(b_z/2 - eta)")
