from pyquda_measurement_utils.tools import srcLoc_distri_eq


def test_src_loc_distribution_has_4_to_the_4_sources_and_respects_origin():
    L = [8, 8, 8, 32]
    origin = [1, 2, 3, 4]
    srcs = srcLoc_distri_eq(L, origin)

    assert len(srcs) == 4**4
    assert srcs[0] == origin
    assert len({tuple(src) for src in srcs}) == len(srcs)
    assert all(0 <= src[mu] < L[mu] for src in srcs for mu in range(4))


def test_src_loc_distribution_wraps_periodically():
    L = [8, 8, 8, 8]
    origin = [7, 7, 7, 7]
    srcs = srcLoc_distri_eq(L, origin)

    assert [7, 7, 7, 7] in srcs
    assert [1, 1, 1, 1] in srcs
