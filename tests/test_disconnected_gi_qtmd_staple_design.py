import numpy as np

from pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop import (
    DisconnectedQuarkqTMD1pt,
    _transport_staple_field,
    gi_qtmd_staple_segments,
)
from qtmd_gi_reference import create_fermion_TMD_GI


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
            "config_num": 0,
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


def test_create_fermion_tmd_gi_calls_covdev_in_reverse_geometric_order():
    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [3, 4, 3, 0])
    assert shifted.path == [6, 0, 0, 0, 2, 2, 2, 2, 2]

    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [2, -4, 3, 1])
    assert shifted.path == [6, 6, 6, 6, 6, 1, 1, 2]


def test_create_fermion_tmd_gi_pdf_limit_path():
    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [0, 2, 1, 0])
    assert shifted.path == [2, 2]

    shifted = create_fermion_TMD_GI(_FakeGauge(), _FakeFermion(), [0, -2, 1, 1])
    assert shifted.path == [6, 6]


class _NumericFermion:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.complex128)

    def copy(self):
        return _NumericFermion(self.data.copy())


class _NumericPureGauge:
    def __init__(self, links):
        self.links = np.asarray(links, dtype=np.complex128)

    def covDev(self, fermion, covdev_direction):
        direction = covdev_direction % 4
        if direction >= 3:
            raise AssertionError(f"unexpected covDev direction {covdev_direction}")
        if covdev_direction < 4:
            shifted = np.roll(fermion.data, -1, axis=direction)
            link = self.links[direction]
        else:
            shifted = np.roll(fermion.data, 1, axis=direction)
            link = np.roll(self.links[direction], 1, axis=direction)
            link = link.conj().swapaxes(-1, -2)
        return _NumericFermion(
            np.einsum("xyzab,xyzb->xyza", link, shifted, optimize=True)
        )


class _NumericGauge:
    def __init__(self, links):
        self.pure_gauge = _NumericPureGauge(links)


def _random_su3(rng, shape):
    matrices = np.empty(shape + (3, 3), dtype=np.complex128)
    for index in np.ndindex(shape):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q, r = np.linalg.qr(raw)
        q = q @ np.diag(np.diag(r) / np.abs(np.diag(r))).conj()
        q /= np.linalg.det(q) ** (1.0 / 3.0)
        matrices[index] = q
    return matrices


def _explicit_geometric_transport(field, links, W_index):
    shape = field.shape[:3]
    result = np.empty_like(field)
    for site in np.ndindex(shape):
        position = list(site)
        transporter = np.eye(3, dtype=np.complex128)
        for direction, steps in gi_qtmd_staple_segments(W_index):
            step = 1 if steps >= 0 else -1
            for _ in range(abs(steps)):
                if step > 0:
                    transporter = transporter @ links[direction][tuple(position)]
                    position[direction] = (position[direction] + 1) % shape[direction]
                else:
                    position[direction] = (position[direction] - 1) % shape[direction]
                    transporter = (
                        transporter
                        @ links[direction][tuple(position)].conj().T
                    )
        result[site] = transporter @ field[tuple(position)]
    return result


def _legacy_forward_segment_transport(gauge, fermion, W_index):
    shifted = fermion.copy()
    for direction, steps in gi_qtmd_staple_segments(W_index):
        covdev_direction = direction if steps >= 0 else direction + 4
        for _ in range(abs(steps)):
            shifted = gauge.pure_gauge.covDev(shifted, covdev_direction)
    return shifted


def test_gi_qtmd_staple_matches_noncommuting_su3_ordered_product():
    rng = np.random.default_rng(20260718)
    shape = (5, 6, 7)
    links = _random_su3(rng, (3, *shape))
    field = rng.normal(size=shape + (3,)) + 1j * rng.normal(
        size=shape + (3,)
    )
    fermion = _NumericFermion(field)
    gauge = _NumericGauge(links)

    cases = [
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [2, 2, 2, 0],
        [2, -2, 2, 1],
    ]
    for W_index in cases:
        actual = _transport_staple_field(gauge, fermion, W_index).data
        expected = _explicit_geometric_transport(field, links, W_index)
        legacy = _legacy_forward_segment_transport(
            gauge, fermion, W_index
        ).data

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        relative_legacy_difference = np.linalg.norm(
            (legacy - expected).reshape(-1)
        ) / np.linalg.norm(expected.reshape(-1))
        assert relative_legacy_difference > 1e-8


if __name__ == "__main__":
    test_gi_qtmd_staple_local_limit()
    test_gi_qtmd_staple_straight_pdf_limit()
    test_gi_qtmd_staple_fixed_length_path()
    test_gi_qtmd_staple_invalid_indices()
    test_gi_qtmd_production_wilson_index_list()
    test_create_fermion_tmd_gi_calls_covdev_in_reverse_geometric_order()
    test_create_fermion_tmd_gi_pdf_limit_path()
    test_gi_qtmd_staple_matches_noncommuting_su3_ordered_product()
    print("[GI qTMD staple design sanity check]")
    print("path = z(eta + b_z/2), transverse(b_T), z(b_z/2 - eta)")
