"""CPU source, full-basis execution, resume and base-statistics regression tests."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import pyquda_measurement_utils.Disconnected_utils_vibe_develop as helpers
import pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop as emt
from pyquda_measurement_utils.fermion_bilinear_basis import basis_metadata


class Comm:
    Get_rank = staticmethod(lambda: 0)
    bcast = staticmethod(lambda value, root=0: value)
    Barrier = staticmethod(lambda: None)


class Grid:
    global_size = (8, 8, 8, 8)
    mpi_rank = 0
    t_boundary = -1

    def __init__(self, start=0, stop=8):
        self.start, self.stop = start, stop

    def coordinate(self, mu=None):
        coords = np.indices(self.global_size)[:, self.start:self.stop]
        return coords if mu is None else coords[mu]


class Source:
    def __init__(self, info, data):
        self.latt_info, self.data = info, data

    def copy(self):
        return Source(self.latt_info, self.data.copy())


def cpu_noise(info, config, base, stream_seed=0, n=4):
    indices = helpers.counter_zn_phase_indices(info, config, base, stream_seed, n)
    table = np.asarray([1, 1j, -1, -1j])
    return Source(info, table[indices])


def multiplier_plan():
    return dict(shape=[8]*4, distance=2, colors=16, sigma=[1, 2, 3, 4], version=1)


def measurement(monkeypatch, **options):
    monkeypatch.setattr(emt, "getMPIComm", lambda: Comm())
    params = dict(config_num=11, qext=[[0, 0, 0, 0]], flow_type="wilson", flow_epsilon=0.1, flow_steps=0)
    params.update(options)
    return emt.EMTDisconnectedQuark1pt(params)


def attrs_for(m, nbase=2, raw=False):
    attrs = m._measurement_attrs(Grid(), [0.1, 1, 1e-10, 100], [nbase, 4, 0], 11, 0,
                                 nbase * helpers.vectors_per_base(m.noise_scheme, m.hp_num_vectors, m.multiplier_plan) * m.spin_color_dilution_factor, 512)
    attrs.update(m._shard_storage_attrs(raw))
    return attrs


def test_emt_and_ringed_use_complete_hp(monkeypatch):
    assert measurement(monkeypatch, hp_num_vectors=7).hp_num_vectors == 1
    for count in (None, 1, 16, 256, 2.5, True):
        with pytest.raises((ValueError, TypeError)):
            measurement(monkeypatch, noise_scheme="hierarchical_probing", hp_num_vectors=count)
    with pytest.raises(ValueError, match="nested"):
        measurement(monkeypatch, noise_scheme="hierarchical_probing", hp_num_vectors=2,
                    hp_ordering="interleaved_xyzt_binary_projected_to_evenodd")
    from pyquda_measurement_utils.flowed_quark_ringed_norm import RingedQuark1pt
    ringed = RingedQuark1pt(dict(config_num=11, flow_type="wilson", flow_epsilon=0.1, flow_steps=0,
                               noise_scheme="hierarchical_probing", hp_num_vectors=32))
    assert ringed.hp_ordering == helpers.NESTED_HP_ORDERING
    with pytest.raises(ValueError):
        measurement(monkeypatch, multiplier_plan=multiplier_plan())


@pytest.mark.parametrize("scheme,count", [("hierarchical_probing", 32), ("multiplier_probing", 16)])
def test_sources_independent_of_batches_and_partition(monkeypatch, scheme, count):
    monkeypatch.setattr(helpers, "make_counter_zn_noise_fermion", cpu_noise)
    plan = multiplier_plan() if scheme == "multiplier_probing" else None

    def records(info, start, stop):
        return list(helpers.iter_noise_base_probe_interval(info, 2, start, stop, 4, scheme,
                    count, helpers.NESTED_HP_ORDERING, 11, 0, plan))

    full = records(Grid(), 0, count)
    split = records(Grid(), 0, 6) + records(Grid(), 6, count)
    left, right = records(Grid(0, 4), 0, count), records(Grid(4, 8), 0, count)
    for a, b, l, r in zip(full, split, left, right):
        assert a[:3] == b[:3]
        np.testing.assert_array_equal(a[3].data, b[3].data)
        np.testing.assert_array_equal(a[3].data, np.concatenate((l[3].data, r[3].data)))
    np.testing.assert_array_equal(cpu_noise(Grid(), 11, 2).data, full[0][3].data)
    assert any(not np.array_equal(full[0][3].data, rec[3].data) for rec in full[1:])


@pytest.mark.parametrize("scheme,count", [("zn", 1), ("hierarchical_probing", 2), ("multiplier_probing", 16)])
@pytest.mark.parametrize("dilution", ["none", "point"])
def test_source_ownership_and_indices(monkeypatch, scheme, count, dilution):
    base = cpu_noise(Grid(), 11, 3)
    original = base.data.copy()
    monkeypatch.setattr(helpers, "make_counter_zn_noise_fermion", lambda *a, **kw: base)
    plan = multiplier_plan() if scheme == "multiplier_probing" else None
    records = helpers.iter_noise_base_probe_interval(
        Grid(), 3, 0, count, 4, scheme, count, helpers.NESTED_HP_ORDERING,
        11, multiplier_plan=plan, spin_color_dilution=dilution,
    )
    channels = [(s, c) for s in range(4) for c in range(3)] if dilution == "point" else [(-1, -1)]
    retained = []
    for probe in range(count):
        if scheme == "hierarchical_probing":
            expected = helpers.apply_hierarchical_probe(base, probe, helpers.NESTED_HP_ORDERING)
        elif scheme == "multiplier_probing":
            expected = helpers.apply_multiplier_probe(base, probe, plan)
        else:
            expected = base.copy()
        for channel, (spin, color) in enumerate(channels):
            index, base_index, probe_index, source, s, c = next(records)
            assert (index, base_index, probe_index, s, c) == (
                (3 * count + probe) * len(channels) + channel, 3, probe, spin, color)
            wanted = (helpers.apply_spin_color_point_dilution(expected, spin, color)
                      if dilution == "point" else expected)
            assert source.data.tobytes() == wanted.data.tobytes()
            assert not np.shares_memory(source.data, base.data)
            assert all(not np.shares_memory(source.data, previous.data) for previous in retained)
            # Model a consumer modifying its source before requesting the next one.
            source.data[:] = 17 + index
            retained.append(source)
            np.testing.assert_array_equal(base.data, original)
    with pytest.raises(StopIteration):
        next(records)


def write_synthetic_shards(root, m, raw=False, nbase=2):
    tag = str(root / "loop")
    common = attrs_for(m, nbase, raw)
    common = {k: v for k, v in common.items() if k not in {"n_vec", "n_base_noise", "effective_n_inversions"}}
    factor = m.spin_color_dilution_factor
    common.update(block_interval_solves=6 * factor, output_kind=m._output_kind())
    count = helpers.vectors_per_base(m.noise_scheme, m.hp_num_vectors, m.multiplier_plan)
    for base in range(nbase):
        for part, start, stop in helpers.base_part_ranges(count, 6):
            attrs = helpers.shard_part_attrs(common, base, part, start, stop, count, m.spin_color_dilution)
            attrs["shard_mean_vector_count"] = stop-start
            raw_values = {}
            for name, shape in m._raw_batch_shapes(Grid(), (stop-start)*factor).items():
                values = np.repeat(np.arange(start+1, stop+1) + 100*base, factor)
                raw_values[name] = np.broadcast_to(values.reshape((-1,) + (1,)*(len(shape)-1)), shape).astype(complex)
            helpers.write_shard_part_hdf5(
                helpers.shard_part_path(root, tag, base, part, start, stop), attrs,
                metadata_datasets=basis_metadata(),
                raw_datasets=raw_values if raw else None,
                source_bookkeeping=helpers.part_source_bookkeeping(base, start, stop, count, m.spin_color_dilution) if raw else None,
                shard_mean_datasets=m._shard_means_from_raw(raw_values),
            )
    return tag


def test_complete_base_weighting_raw_mean_and_sem(tmp_path, monkeypatch):
    m = measurement(monkeypatch, noise_scheme="multiplier_probing", multiplier_plan=multiplier_plan())
    results = []
    for raw in (False, True):
        root = tmp_path / str(raw)
        tag = write_synthetic_shards(root, m, raw)
        output = emt.aggregate_emt_quark_1pt_bases(root, tag, 2)
        with h5py.File(output) as h5:
            values = h5["base/local_bilinear"][()]
            np.testing.assert_allclose(values[0], 8.5 / 512)
            np.testing.assert_allclose(values[1], 108.5 / 512)
            np.testing.assert_allclose(h5["avg/local_bilinear"], 58.5 / 512)
            np.testing.assert_allclose(h5["sem_real/local_bilinear"], 50 / 512)
            np.testing.assert_array_equal(h5["sem_imag/local_bilinear"], 0)
            results.append(values)
        if raw:
            canonical = emt.finalize_emt_quark_1pt_shards(root, tag, 2)
            with h5py.File(canonical) as old, h5py.File(output) as bases:
                np.testing.assert_allclose(old["avg/derivative_bilinear"], bases["avg/derivative_bilinear"])
                np.testing.assert_allclose(old["derived/ringed/kinetic_spacetime"], bases["avg/ringed/kinetic_spacetime"])
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("raw", [False, True])
def test_sc12_aggregation_feeds_new_analysis(tmp_path, monkeypatch, raw):
    from application.analysis_helper.emt_ringed_stochastic_comparison import load_ringed_base_series
    from application.analysis_helper.emt_quark_1pt_convergence import load_quark_1pt_base_series
    from application.analysis_helper.emt_proton_t44_analysis import load_t44_base_loops
    m = measurement(monkeypatch, noise_scheme="hierarchical_probing", hp_num_vectors=2,
                    spin_color_dilution="point")
    tag = write_synthetic_shards(tmp_path, m, raw=raw)
    output = emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 2)
    with h5py.File(output) as h5:
        assert h5.attrs["effective_n_inversions"] == 48
        np.testing.assert_allclose(h5["base/local_bilinear"][0], 12*1.5/512)
        np.testing.assert_allclose(h5["base/local_bilinear"][1], 12*101.5/512)
        expected_kinetic = h5["base/ringed/kinetic_spacetime"][:, 0]
        expected_t44 = h5["base/Tmunu/T44"][()]
        assert "hp_num_vectors" not in h5.attrs and "solves_per_base" not in h5.attrs
    ringed = load_ringed_base_series(output, flow_index=0)
    convergence = load_quark_1pt_base_series(output, flow_index=0)
    t44 = load_t44_base_loops(output)
    assert ringed.solves_per_base == convergence.solves_per_base == t44.solves_per_base == 24
    np.testing.assert_array_equal(ringed.base_values, expected_kinetic)
    np.testing.assert_array_equal(convergence.ringed_base_values, expected_kinetic)
    np.testing.assert_array_equal(t44.base_values, expected_t44)
    np.testing.assert_allclose(convergence.emt_base_values, expected_t44[:, 0, 0].mean(axis=-1))


def test_single_base_undefined_sem_and_missing_mean_rejected(tmp_path, monkeypatch):
    m = measurement(monkeypatch, noise_scheme="hierarchical_probing", hp_num_vectors=2)
    tag = write_synthetic_shards(tmp_path, m, raw=True, nbase=1)
    output = emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 1)
    with h5py.File(output) as h5:
        np.testing.assert_allclose(h5["avg/local_bilinear"], 1.5 / 512)
        assert np.isnan(h5["sem_real/local_bilinear"][()]).all()
        assert not h5.attrs["sem_defined"]
    for path in tmp_path.glob("*.base*.part*.h5"):
        with h5py.File(path, "r+") as part:
            del part["shard_mean"]
            part.attrs["shard_mean_stored"] = False
    with pytest.raises(ValueError, match="requires shard_mean"):
        emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 1)


@pytest.mark.parametrize("damage", ["old_schema", "missing_scheme", "missing_ordering", "old_ordering"])
def test_aggregation_rejects_uniform_obsolete_or_missing_metadata(tmp_path, monkeypatch, damage):
    m = measurement(monkeypatch, noise_scheme="hierarchical_probing", hp_num_vectors=2)
    tag = write_synthetic_shards(tmp_path, m, nbase=1)
    for path in tmp_path.glob("*.base*.part*.h5"):
        with h5py.File(path, "r+") as part:
            if damage == "old_schema":
                part.attrs["shard_schema"] = "disconnected_1pt_base_parts_v3"
            elif damage == "old_ordering":
                part.attrs["hp_ordering"] = "interleaved_xyzt_binary_projected_to_evenodd"
            else:
                del part.attrs["noise_scheme" if damage == "missing_scheme" else "hp_ordering"]
    with pytest.raises((KeyError, ValueError)):
        emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 1)
    assert not Path(tag + ".bases.h5").exists()


def test_zn_metadata_has_one_probe_count(monkeypatch):
    m = measurement(monkeypatch)
    attrs = attrs_for(m)
    assert "hp_ordering" not in attrs
    assert "hp_num_vectors" not in attrs
    assert attrs["vectors_per_base"] == 1
    assert "probing_algorithm" not in attrs


@pytest.mark.parametrize("raw", [False, True])
def test_new_schema_resume_and_old_log_rejection(tmp_path, monkeypatch, raw):
    m = measurement(monkeypatch)
    attrs = attrs_for(m, raw=raw)
    common = {k: v for k, v in attrs.items()
              if k not in {"n_base_noise", "effective_n_inversions"}}
    common.update(output_kind=m._output_kind(), block_interval_solves=6)
    log = tmp_path / "sample.log"
    log.write_text("# disconnected_sample_log_v1 legacy\nbase000000\n")
    with pytest.raises(ValueError, match="header mismatch"):
        helpers.prepare_sample_log(log, "loop", common)
    log = tmp_path / "new.log"
    assert helpers.prepare_sample_log(log, "loop", common) == set()
    helpers.append_completed_base(log, "loop", common, 0)
    assert helpers.prepare_sample_log(log, "loop", common) == {0}
    with pytest.raises(ValueError, match="header mismatch"):
        helpers.prepare_sample_log(log, "loop", dict(common, spin_color_dilution="point"))


@pytest.mark.parametrize("scheme,count", [("zn", 1), ("hierarchical_probing", 2)])
def test_qtmd_source_loop_accepts_extended_records(tmp_path, monkeypatch, scheme, count):
    import pyquda_measurement_utils.Disconnected_1pt_qTMD_vibe_develop as qtmd
    monkeypatch.setattr(qtmd, "getMPIComm", lambda: Comm())
    monkeypatch.setattr(helpers, "make_counter_zn_noise_fermion", cpu_noise)
    m = qtmd.DisconnectedQuarkqTMD1pt(dict(
        config_num=11, b_z=[0], b_T=[0], qext=[[0, 0, 0, 0]],
        noise_scheme=scheme, hp_num_vectors=count,
    ))
    seen = []

    def invert(source):
        seen.append(source.data.copy())
        return source.copy()

    monkeypatch.setattr(m, "_contract_one_operator_list", lambda *a, **kw:
                        np.ones((1, len(qtmd.my_gammas), 1, 8), complex))
    tag, log = str(tmp_path / "loop"), tmp_path / "sample.log"
    m._measure_base_shards(
        SimpleNamespace(latt_info=Grid()), SimpleNamespace(invert=invert),
        [1, 4, 0], tag, "CG_PDF", None, [[0, 0, 0, 0]], None,
        dict(qext=[[0, 0, 0, 0]], config_num=11, noise_stream=0),
        tmp_path, 0, 1, 64, log, set(),
    )
    assert len(seen) == count
    np.testing.assert_array_equal(seen[0], cpu_noise(Grid(), 11, 0).data)
    with h5py.File(helpers.shard_part_path(tmp_path, tag, 0, 0, 0, count)) as part:
        np.testing.assert_array_equal(part["raw/probe_index"], np.arange(count))
        assert part["raw/loop_pervec"].shape[0] == count
    assert log.read_text().splitlines()[-1] == "base000000"


@pytest.mark.parametrize("target,backend", [("hip", "cupy"), ("sycl", "dpnp")])
def test_gpu_entry_source_hash_control_flow(tmp_path, monkeypatch, target, backend):
    # Exercise entry-point wiring with CPU arrays, not GPU kernels or MPI.
    import hashlib
    import json
    import runpy
    import sys
    import pyquda
    import pyquda.field
    comm = SimpleNamespace(rank=0, size=1, bcast=Comm.bcast,
                           gather=lambda value, root=0: [value])
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=comm)))
    if target == "hip":
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True))))
    initialized = []
    monkeypatch.setattr(pyquda, "init", lambda *a, **kw: initialized.append((a, kw)), raising=False)
    monkeypatch.setattr(sys.modules["pyquda.field"], "LatticeInfo", lambda shape: Grid())
    monkeypatch.setattr(helpers, "make_counter_zn_noise_fermion", cpu_noise)
    monkeypatch.setenv("QUDA_RESOURCE_PATH", "unused-cpu-test-cache")
    output = tmp_path / "smoke"
    entry = Path(__file__).with_name("check_disconnected_probing_gpu.py")
    monkeypatch.setattr(sys, "argv", [str(entry), "--output", str(output), "--backend-target", target])
    runpy.run_path(str(entry))["main"]()
    assert initialized == [(([1, 1, 1, 1],), dict(
        backend=backend, backend_target=target, enable_mps=False))]
    result = json.loads((output / "smoke.json").read_text())
    coords = Grid().coordinate()
    ids = (coords[0] + 8*(coords[1] + 8*(coords[2] + 8*coords[3]))).ravel()
    noise = cpu_noise(Grid(), 11, 0).data
    parity = 1 - 2*(coords.sum(axis=0) % 2)
    expected = hashlib.sha256()
    # Match in-place floating sign multiplication, including signed zero bits.
    for signs in (np.ones(Grid.global_size), parity.astype(float)):
        values = noise.copy()
        values *= signs[..., None, None]
        expected.update(values.reshape(-1, 4, 3)[np.argsort(ids)].astype("<c16").tobytes())
    assert result["global_source_sha256"] == expected.hexdigest()
    assert result["backend_target"] == target and result["backend"] == backend


def test_fixed_plan_files_pass_checker():
    import json
    folder = Path(__file__).resolve().parents[1] / "application/EMT_disconnected_1pt/probing_plans"
    paths = sorted(folder.glob("*.json"))
    assert len(paths) == 5
    for path in paths:
        helpers.validate_multiplier_plan(json.loads(path.read_text()))


@pytest.mark.parametrize("damage", ["missing", "extra", "count", "plan", "ordering", "nan"])
def test_base_aggregation_rejects_bad_input_atomically(tmp_path, monkeypatch, damage):
    m = measurement(monkeypatch, noise_scheme="multiplier_probing", multiplier_plan=multiplier_plan())
    tag = write_synthetic_shards(tmp_path, m)
    output = Path(emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 2))
    previous = output.read_bytes()
    path = helpers.shard_part_path(tmp_path, tag, 1, 1, 6, 12)
    if damage == "missing":
        path.unlink()
    elif damage == "extra":
        path.with_name(path.name.replace("part0001", "part9999")).write_bytes(path.read_bytes())
    else:
        with h5py.File(path, "r+") as part:
            if damage == "count":
                part.attrs["shard_mean_vector_count"] = 5
            elif damage == "plan":
                part.attrs["multiplier_plan_json"] = "{}"
            elif damage == "ordering":
                part.attrs["hp_ordering"] = "legacy"
            else:
                part["shard_mean/local_bilinear"][...] = np.nan
    with pytest.raises(ValueError):
        emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 2)
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp.*"))


def test_multiplier_full_loop_and_interrupted_resume(tmp_path, monkeypatch):
    m = measurement(monkeypatch, noise_scheme="multiplier_probing", multiplier_plan=multiplier_plan())
    monkeypatch.setattr(helpers, "make_counter_zn_noise_fermion", cpu_noise)
    attrs = attrs_for(m)
    common = {k: v for k, v in attrs.items() if k not in {"n_vec", "n_base_noise", "effective_n_inversions"}}
    common.update(output_kind=m._output_kind(), block_interval_solves=6)
    tag, log = str(tmp_path / "loop"), tmp_path / "sample.log"
    helpers.prepare_sample_log(log, tag, common)
    seen = []
    fail = [True]

    def fake_measure(gauge, dirac, records, phases, **kwargs):
        if records[0][1] == 1 and records[0][2] >= 6 and fail[0]:
            raise RuntimeError("intentional interruption")
        seen.extend(rec[:3] for rec in records)
        shapes = m._raw_batch_shapes(Grid(), len(records))
        return {name: np.broadcast_to(np.array([r[2]+1+100*r[1] for r in records]).reshape((-1,) + (1,)*(len(shape)-1)), shape).astype(complex)
                for name, shape in shapes.items()}

    monkeypatch.setattr(m, "_invert_and_measure_batch", fake_measure)

    def run(completed):
        m._measure_base_shards(SimpleNamespace(latt_info=Grid()), object(), [2, 4, 0], tag,
                               None, attrs, tmp_path, log, 0, 2, 6, completed, 4)

    with pytest.raises(RuntimeError, match="interruption"):
        run(set())
    completed = helpers.prepare_sample_log(log, tag, common)
    assert completed == {0}
    fail[0] = False
    seen.clear()
    run(completed)
    assert [(b, k) for _, b, k in seen] == [(1, k) for k in range(16)]
    assert helpers.prepare_sample_log(log, tag, common) == {0, 1}
    assert len(log.read_text().splitlines()) == 3
    resumed = emt.aggregate_emt_quark_1pt_bases(tmp_path, tag, 2)
    expected_tag = write_synthetic_shards(tmp_path / "continuous", m)
    expected = emt.aggregate_emt_quark_1pt_bases(tmp_path / "continuous", expected_tag, 2)
    with h5py.File(resumed) as left, h5py.File(expected) as right:
        np.testing.assert_array_equal(left["base/derivative_bilinear"], right["base/derivative_bilinear"])
    with pytest.raises(ValueError, match="header mismatch"):
        helpers.prepare_sample_log(log, tag, dict(common, probing_algorithm="other"))
