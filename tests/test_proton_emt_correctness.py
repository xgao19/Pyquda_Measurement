import numpy as np

import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt_module
import pyquda_measurement_utils.bw_seq_pyquda as bw_module
import pyquda_measurement_utils.proton_EMT_vibe_develop as proton_module
import pyquda_measurement_utils.proton_qTMD_pyquda as qtmd_module
from pyquda_measurement_utils.proton_EMT_vibe_develop import ProtonQuarkEMT


def _measurement(smearing):
    return ProtonQuarkEMT({
        "config_num": 1,
        "qext": [[0, 0, 0, 0]],
        "pf": [0, 0, 0, 0],
        "p_2pt": [[0, 0, 0, 0]],
        "CG_GaussSmear": smearing,
        "boost_in": [0, 0, 0],
        "boost_out": [0, 0, 0],
        "width": 1.0,
        "pol": ["PpUnpol"],
        "t_insert": 2,
        "flow_type": "wilson",
        "flow_epsilon": 0.1,
        "flow_steps": 0,
    })


def test_proton_point_source_and_c2_sink_never_smear(monkeypatch):
    measurement = _measurement(False)

    class Info:
        mpi_rank = 0

    class Gauge:
        latt_info = Info()

    class Dirac:
        loads = []

        def loadGauge(self, gauge, thin_update_only=False):
            self.loads.append(thin_update_only)

    source_object = object()
    monkeypatch.setattr(proton_module.source, "propagator", lambda *_args: source_object)
    monkeypatch.setattr(proton_module.core, "invertPropagator", lambda *_args: source_object)
    monkeypatch.setattr(
        proton_module,
        "boosted_smearing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("smearing called")),
    )
    assert measurement._make_source_prop(
        Dirac(), Gauge(), [0, 0, 0, 0], restore_original_gauge=False
    ) is source_object

    captured = {}
    monkeypatch.setattr(
        measurement, "_gamma_stack_for", lambda _data: np.zeros((16, 4, 4))
    )
    monkeypatch.setattr(
        measurement, "_cached_backend_matrix", lambda *_args: np.zeros((4, 4))
    )
    monkeypatch.setattr(
        proton_module.phase,
        "MomentumPhase",
        lambda _info: type("P", (), {"getPhases": lambda *_args: np.ones((1, 1))})(),
    )
    monkeypatch.setattr(
        proton_module,
        "contract_proton_c2",
        lambda *_args, **kwargs: captured.update(kwargs) or np.zeros((16, 1, 1)),
    )
    monkeypatch.setattr(proton_module, "getMPIComm", lambda: type("C", (), {"bcast": staticmethod(lambda x, root=0: x)})())
    prop = type("Prop", (), {"data": np.zeros(1)})()
    measurement.contract_proton_2pt(Info(), prop, [0, 0, 0, 0])
    assert captured["sink_smearing"] is False


def test_proton_point_sequential_path_skips_both_smearing_calls(monkeypatch):
    class Info:
        mpi_rank = 0
        GLt = 4

    prop = type(
        "Prop",
        (),
        {
            "data": np.zeros((1, 1, 1, 1, 1, 4, 4, 3, 3), np.complex128),
            "latt_info": Info(),
        },
    )()
    monkeypatch.setattr(
        bw_module,
        "boosted_smearing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("smearing called")),
    )
    monkeypatch.setattr(bw_module, "up_quark_insertion_pyquda", lambda *_args: prop)
    monkeypatch.setattr(bw_module, "sequential12", lambda value, _time: value)
    monkeypatch.setattr(
        bw_module,
        "MomentumPhase",
        lambda _info: type(
            "P", (), {"getPhase": lambda *_args, **_kwargs: np.ones((1, 1, 1, 1, 1))}
        )(),
    )
    monkeypatch.setattr(bw_module.gamma, "gamma", lambda _index: np.eye(4))
    monkeypatch.setattr(
        bw_module, "_asarray_on_queue", lambda value, _xp, _ref: np.asarray(value)
    )
    monkeypatch.setattr(
        bw_module.core,
        "LatticePropagator",
        lambda info: type("Sequential", (), {"latt_info": info, "data": None})(),
    )
    monkeypatch.setattr(bw_module.core, "invertPropagator", lambda _dirac, src, *_args: src)
    result = list(bw_module._iter_bw_seq_raw(
        object(), prop, [0, 0, 0, 0], None, None, [0, 0, 0], 2,
        ["PpUnpol"], 1
    ))
    assert len(result) == 1


def test_qtmd_c2_is_a_thin_shared_kernel_caller(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        qtmd_module,
        "contract_proton_c2",
        lambda *_args, **kwargs: captured.update(kwargs) or np.zeros((16, 1, 1)),
    )
    helper = qtmd_module.proton_TMD({
        "eta": [0], "b_z": 0, "b_T": 0,
        "pf": [0, 0, 0, 0], "qext": [[0, 0, 0, 0]],
        "qext_PDF": [[0, 0, 0, 0]], "p_2pt": [[0, 0, 0, 0]],
        "width": 2.0, "boost_in": [1, 0, 0], "boost_out": [0, 0, 0],
        "pol": ["PpUnpol"], "t_insert": 2, "save_propagators": False,
    })
    info = type("Info", (), {"mpi_rank": 1})()
    helper.contract_2pt_TMD(info, object(), object(), "unused")
    assert captured["sink_smearing"] is True
    assert captured["smearing_width"] == 2.0
    assert captured["smearing_boost"] == [1, 0, 0]


def test_emt_gamma_cache_is_per_dtype_and_sycl_queue(monkeypatch):
    import pyquda_measurement_utils.flowed_fermion_bilinear_vibe_develop as bilinear

    measurement = _measurement(False)
    calls = []
    monkeypatch.setattr(bilinear, "_get_xp_from_array", lambda _ref: np)
    monkeypatch.setattr(
        bilinear,
        "gamma_stack",
        lambda ref: calls.append(ref) or np.zeros((16, 4, 4), dtype=ref.dtype),
    )

    class Ref:
        def __init__(self, dtype, queue):
            self.dtype = np.dtype(dtype)
            self.sycl_queue = queue

    queue_a, queue_b = object(), object()
    ref_a = Ref(np.complex128, queue_a)
    assert measurement._gamma_stack_for(ref_a) is measurement._gamma_stack_for(ref_a)
    measurement._gamma_stack_for(Ref(np.complex128, queue_b))
    measurement._gamma_stack_for(Ref(np.complex64, queue_a))
    assert len(calls) == 3
