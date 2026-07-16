import numpy as np

import pyquda_measurement_utils.pion_qTMDWF_pyquda as qda_module
from pyquda_measurement_utils.pion_qTMDWF_pyquda import pion_TMDWF_measurement
from pyquda_measurement_utils.pion_utils_vibe_develop import gamma_stack


class _TinyLatticeInfo:
    size = [1, 1, 1, 1]
    mpi_rank = 0


class _TinyPropagator:
    def __init__(self, data, path=()):
        self.data = np.asarray(data)
        self.path = tuple(path)

    def copy(self):
        return _TinyPropagator(self.data.copy(), self.path)


class _FakeDAMeasurement:
    contract_DA = pion_TMDWF_measurement.contract_DA

    def __init__(self):
        self.cg_calls = []
        self.gi_calls = []

    @staticmethod
    def _updated(prop, current, previous):
        delta = int(current[1]) - int(previous[1])
        return _TinyPropagator(prop.data * (1 + 0.1 * delta), prop.path + (delta,))

    def create_fw_prop_TMD_CG(self, prop, current, previous):
        self.cg_calls.append((prop.path, tuple(current), tuple(previous)))
        return self._updated(prop, current, previous)

    def create_fw_prop_PDF_GI(self, gauge, prop, current, previous):
        assert gauge is not None
        self.gi_calls.append((prop.path, tuple(current), tuple(previous)))
        return self._updated(prop, current, previous)


def _tiny_contract_inputs(seed=17):
    rng = np.random.default_rng(seed)
    shape = (2, 1, 1, 1, 1, 4, 4, 1, 1)
    forward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    backward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    phases = np.ones((1, 2, 1, 1, 1, 1), dtype=np.complex128)
    return _TinyPropagator(forward), _TinyPropagator(backward), phases


def test_da_transports_only_forward_and_restarts_negative_branch(monkeypatch):
    forward, backward, phases = _tiny_contract_inputs()
    measurement = _FakeDAMeasurement()
    backward_objects = []

    monkeypatch.setattr(
        qda_module,
        "gamma_stack",
        lambda reference: np.asarray([np.eye(4, dtype=np.complex128)]),
    )
    monkeypatch.setattr(
        qda_module,
        "source_gamma_stack",
        lambda label, sink, reference: np.asarray([np.eye(4, dtype=np.complex128)]),
    )
    monkeypatch.setattr(
        qda_module,
        "meson_backward_line",
        lambda prop: backward_objects.append(prop) or prop.data,
    )
    monkeypatch.setattr(qda_module.core, "gatherLattice", lambda values, axes: values)

    wilson_indices = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, -1, 0, 0],
        [0, -2, 0, 0],
    ]
    cg = measurement.contract_DA(
        _TinyLatticeInfo(),
        None,
        forward,
        backward,
        phases,
        wilson_indices,
        ["5"],
        gauge_invariant=False,
    )
    gi = measurement.contract_DA(
        _TinyLatticeInfo(),
        object(),
        forward,
        backward,
        phases,
        wilson_indices,
        ["5"],
        gauge_invariant=True,
    )

    expected_steps = [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((0, 1, 0, 0), (0, 0, 0, 0)),
        ((0, 2, 0, 0), (0, 1, 0, 0)),
        ((0, -1, 0, 0), (0, 0, 0, 0)),
        ((0, -2, 0, 0), (0, -1, 0, 0)),
    ]
    assert [(cur, prev) for _, cur, prev in measurement.cg_calls] == expected_steps
    assert [(cur, prev) for _, cur, prev in measurement.gi_calls] == expected_steps
    assert measurement.cg_calls[3][0] == ()
    assert measurement.gi_calls[3][0] == ()
    assert backward_objects == [backward, backward]
    assert cg[0][1].shape == (len(wilson_indices), 1, 1, 1)
    local_site = np.einsum(
        "wtzyxjiab,wtzyxilba,lj->wtzyx",
        backward.data,
        forward.data,
        np.eye(4),
        optimize=True,
    )
    local_reference = np.einsum(
        "qwtzyx,wtzyx->qt", phases, local_site, optimize=True
    )
    np.testing.assert_allclose(
        cg[0][1][0, :, 0, :], local_reference, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(cg[0][1], gi[0][1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cg[0][1][0], gi[0][1][0], rtol=1e-13, atol=1e-13)


class _NumericFermion:
    def __init__(self, data):
        self.data = np.asarray(data)


class _NumericPureGauge:
    def __init__(self, links):
        self.links = np.asarray(links)

    def covDev(self, fermion, direction):
        field = fermion.data
        if direction == 2:
            shifted = np.roll(field, -1, axis=1)
            result = np.einsum("tzab,tzsb->tzsa", self.links, shifted)
        elif direction == 6:
            links_back = np.roll(self.links, 1, axis=1).conj().swapaxes(-1, -2)
            shifted = np.roll(field, 1, axis=1)
            result = np.einsum("tzab,tzsb->tzsa", links_back, shifted)
        else:
            raise AssertionError(f"unexpected covDev direction {direction}")
        return _NumericFermion(result)


class _NumericGauge:
    def __init__(self, links):
        self.pure_gauge = _NumericPureGauge(links)


class _NumericPropagator:
    def __init__(self, fields):
        self.fields = {key: _NumericFermion(value.data.copy()) for key, value in fields.items()}

    @classmethod
    def random(cls, rng, nt, nz):
        return cls({
            (spin, color): _NumericFermion(
                rng.normal(size=(nt, nz, 4, 3))
                + 1j * rng.normal(size=(nt, nz, 4, 3))
            )
            for spin in range(4)
            for color in range(3)
        })

    def copy(self):
        return _NumericPropagator(self.fields)

    def getFermion(self, spin, color):
        return self.fields[(spin, color)]

    def setFermion(self, fermion, spin, color):
        self.fields[(spin, color)] = fermion


def _random_su3(rng, shape):
    matrices = np.empty(shape + (3, 3), dtype=np.complex128)
    for index in np.ndindex(shape):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        q, r = np.linalg.qr(raw)
        q = q @ np.diag(np.diag(r) / np.abs(np.diag(r))).conj()
        q /= np.linalg.det(q) ** (1.0 / 3.0)
        matrices[index] = q
    return matrices


def _straight_transport(field, links, separation):
    result = np.asarray(field).copy()
    direction = 1 if separation >= 0 else -1
    for _ in range(abs(separation)):
        if direction > 0:
            result = np.einsum(
                "tzab,tzsb->tzsa", links, np.roll(result, -1, axis=1)
            )
        else:
            links_back = np.roll(links, 1, axis=1).conj().swapaxes(-1, -2)
            result = np.einsum(
                "tzab,tzsb->tzsa", links_back, np.roll(result, 1, axis=1)
            )
    return result


def test_pdf_gi_transport_matches_explicit_straight_wilson_line():
    rng = np.random.default_rng(20260716)
    nt, nz = 2, 5
    links = _random_su3(rng, (nt, nz))
    original = _NumericPropagator.random(rng, nt, nz)
    measurement = object.__new__(pion_TMDWF_measurement)
    gauge = _NumericGauge(links)

    positive = original.copy()
    previous = [0, 0, 0, 0]
    for separation in (1, 2):
        current = [0, separation, 0, 0]
        positive = measurement.create_fw_prop_PDF_GI(
            gauge, positive, current, previous
        )
        for key, fermion in positive.fields.items():
            expected = _straight_transport(original.fields[key].data, links, separation)
            np.testing.assert_allclose(fermion.data, expected, rtol=1e-13, atol=1e-13)
        previous = current

    negative = original.copy()
    previous = [0, 0, 0, 0]
    for separation in (-1, -2):
        current = [0, separation, 0, 0]
        negative = measurement.create_fw_prop_PDF_GI(
            gauge, negative, current, previous
        )
        for key, fermion in negative.fields.items():
            expected = _straight_transport(original.fields[key].data, links, separation)
            np.testing.assert_allclose(fermion.data, expected, rtol=1e-13, atol=1e-13)
        previous = current


def test_straight_link_da_is_locally_gauge_invariant_but_cg_is_not():
    rng = np.random.default_rng(9107)
    nt, nz = 3, 5
    links = _random_su3(rng, (nt, nz))
    omega = _random_su3(rng, (nt, nz))
    quark = rng.normal(size=(nt, nz, 4, 3)) + 1j * rng.normal(
        size=(nt, nz, 4, 3)
    )
    antiquark = rng.normal(size=(nt, nz, 4, 3)) + 1j * rng.normal(
        size=(nt, nz, 4, 3)
    )

    omega_next = np.roll(omega, -1, axis=1)
    transformed_links = np.einsum(
        "tzab,tzbc,tzdc->tzad", omega, links, omega_next.conj()
    )
    transformed_quark = np.einsum("tzab,tzsb->tzsa", omega, quark)
    transformed_antiquark = np.einsum(
        "tzsb,tzba->tzsa", antiquark, omega.conj().swapaxes(-1, -2)
    )
    phases = np.exp(2j * np.pi * np.arange(nz) / nz)
    gammas = gamma_stack(np.zeros((1,), dtype=np.complex128))

    def correlator(left, right, link_field, separation, gauge_invariant):
        shifted = (
            _straight_transport(right, link_field, separation)
            if gauge_invariant
            else np.roll(right, -separation, axis=1)
        )
        values = []
        for sink_gamma in gammas:
            per_source = []
            for source_gamma in gammas:
                spin_matrix = sink_gamma @ source_gamma
                site = np.einsum(
                    "tzsc,sr,tzrc->tz", left, spin_matrix, shifted
                )
                per_source.append(np.einsum("z,tz->t", phases, site))
            values.append(per_source)
        return np.asarray(values)

    for separation in (-2, -1, 0, 1, 2):
        before = correlator(antiquark, quark, links, separation, True)
        after = correlator(
            transformed_antiquark,
            transformed_quark,
            transformed_links,
            separation,
            True,
        )
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)

    cg_before = correlator(antiquark, quark, links, 1, False)
    cg_after = correlator(
        transformed_antiquark,
        transformed_quark,
        transformed_links,
        1,
        False,
    )
    assert not np.allclose(cg_before, cg_after, rtol=1e-8, atol=1e-8)


def test_qda_application_uses_pdf_gi_and_preserves_sample_log_identity():
    from pathlib import Path

    source = Path("application/qDA/frontier_charm/pyquda_DA_k6.py").read_text()
    assert "create_fw_prop_TMD_GI" not in source
    assert "source_gamma_label=\"dagger_of_sink\"" in source
    assert 'get_sample_log_tag("ex", pos, sm_tag)' in source
    assert "Measurement.contract_DA(" in source
