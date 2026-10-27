"""Compare production SS/SP/C4 channels with native GPT CPU tensor operations.

Requires GPT and gpt-run source trees, not a compiled cgpt extension. The native
tensor/object-type classes, Gamma definitions and gpt-run expressions execute
unchanged. Site iteration, periodic shifts, spatial sums and the slice_trDA
C++ loop are adapted to Python; this is not a compiled lattice/MPI test.
"""

import argparse
import ast
import csv
import hashlib
import json
from math import pi, prod
from pathlib import Path
import re
import sys
from types import MethodType, SimpleNamespace

import numpy as np

from run_probing_cpu_tests import install_stubs


def load_definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names), (path, names)
    source = 'from __future__ import annotations\n' + ast.unparse(ast.Module(nodes, type_ignores=[]))
    exec(compile(source, str(path), 'exec'), namespace)


def expression(path, target, occurrence=0):
    matches = sorted((node for node in ast.walk(ast.parse(path.read_text()))
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target),
                     key=lambda node: node.lineno)
    node = matches[occurrence]
    return compile(ast.Expression(node.value), str(path), 'eval')


class Site:
    """Locate an unchanged native tensor; all arithmetic stays in GPT tensor."""

    def __init__(self, engine, field, coordinate):
        self.engine, self.field, self.coordinate = engine, field, coordinate

    def __mul__(self, other):
        return self.engine.eval(self) * self.engine.eval(other)

    def __rmul__(self, other):
        return self.engine.eval(other) * self.engine.eval(self)


class NativeGPT(SimpleNamespace):
    def __init__(self, root, labels):
        super().__init__(util=SimpleNamespace(is_num=np.isscalar, to_list=lambda x: x if isinstance(x, list) else [x]),
                         expr_unary=SimpleNamespace(BIT_SPINTRACE=1, BIT_COLORTRACE=2),
                         expr=type('UnusedExpression', (), {}))
        core = root / 'lib/gpt/core'
        ns = {'np': np, 'numpy': np, 'gpt': self, 'foundation': None, 'foundation_base': object}
        load_definitions(core / 'object_type/base.py', {'ot_base'}, ns)
        load_definitions(core / 'object_type/container.py',
                         {'ot_singlet', 'ot_matrix_color', 'ot_matrix_spin', 'ot_matrix_spin_color'}, ns)
        load_definitions(core / 'tensor.py', {'tensor'}, ns)
        self.tensor, self.ot_singlet = ns['tensor'], ns['ot_singlet']
        self.spin_type, self.field_type = ns['ot_matrix_spin'](4), ns['ot_matrix_spin_color'](4, 3)
        self.coordinate = (0, 0, 0, 0)

        gamma_ns = {'np': np}
        gamma_ns['matrices'] = eval(expression(core / 'gamma.py', 'matrices'), gamma_ns)
        load_definitions(core / 'gamma.py', {'fill_sigmas'}, gamma_ns)
        gamma_ns['fill_sigmas']()
        gamma_map = next(node.value for node in ast.parse((core / 'gamma.py').read_text()).body
                         if isinstance(node, ast.Assign) and node.targets[0].id == 'gamma')
        self.gamma = {ast.literal_eval(key): self.tensor(gamma_ns['matrices'][ast.literal_eval(value.args[0])], self.spin_type)
                      for key, value in zip(gamma_map.keys, gamma_map.values)}
        kernel_path = root / 'lib/cgpt/lib/slice_trace_gpu.h'
        kernel = kernel_path.read_text()
        gamma_enum = re.search(r'Gmu16\s*\(\s*\{(.*?)\}\s*\)', kernel, re.S).group(1)
        names = re.findall(r'Gamma::Algebra::(\w+)', gamma_enum)
        self.sink_gammas = [self.gamma_enum(name) for name in names]
        assert len(self.sink_gammas) == len(labels) == 16
        # Bind only the outer C++ loop; every product/trace uses native GPT tensor.
        assert 'tmp_vobj = Data2[0]*Data[nbasis]' in kernel
        assert 'trace(tmp_vobj*Gamma(Gmu16[ngamma]))' in kernel
        cpu_kernel = (root / 'lib/cgpt/lib/slice_trace.h').read_text()
        assert 'Data2_v[0][ss])*coalescedRead(Data_v[n_base][ss])' in cpu_kernel
        bridge = {'gpt': self, 'cgpt': SimpleNamespace(slice_traceDA=self.slice_trace_kernel)}
        load_definitions(core / 'transform.py', {'slice_trDA'}, bridge)
        self.slice_trDA = bridge['slice_trDA']

    def gamma_enum(self, name):
        if name == 'Identity':
            return self.gamma['I']
        if name == 'Gamma5':
            return self.gamma[5]
        if name.endswith('Gamma5'):
            return self.gamma[name[5:-6]] * self.gamma[5]
        return self.gamma[name[5:] if name.startswith('Gamma') else name]

    def __call__(self, value):
        return self.eval(value)

    def eval(self, value):
        if isinstance(value, Site):
            return self.tensor(value.field[value.coordinate], self.field_type)
        return value

    def adj(self, value):
        return self.eval(value).adj()

    def trace(self, value):
        return self.eval(value).trace(3)

    def slice(self, value, dimension):
        # The caller explicitly sums these site results with native tensor addition.
        assert dimension == 3
        return self.eval(value)

    def cshift(self, value, direction, distance):
        coordinate = list(value.coordinate)
        axis = 3 - direction
        coordinate[axis] = (coordinate[axis] + distance) % value.field.shape[axis]
        return Site(self, value.field, tuple(coordinate))

    def complex(self, grid):
        return 1.0 + 0.0j

    def identity(self, value):
        return value

    def exp_ixp(self, momentum):
        return np.exp(1j * np.dot(momentum, self.coordinate[::-1]))

    def slice_trace_kernel(self, lhs, rhs, momenta, dimension):
        assert dimension == 3
        return [[[[self.trace((rhs[0] * left) * gamma) * momentum]
                  for gamma in self.sink_gammas] for momentum in momenta] for left in lhs]


def native_results(engine, source_path, phase_path, fields, info, sources, pairs, c4_pairs, insertions, kf, kb):
    """Execute unmodified native gpt-run contraction expressions site by site."""
    g = engine
    method = next(node for node in ast.walk(ast.parse(phase_path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == 'apply_phase'
                  and len(node.args.args) == 5)
    ns = {'np': np, 'g': g}
    exec(compile(ast.Module([method], type_ignores=[]), str(phase_path), 'exec'), ns)
    measurement = SimpleNamespace()
    measurement.apply_phase = MethodType(ns['apply_phase'], measurement)
    grid = SimpleNamespace(fdimensions=np.array(info.global_size))
    env = {'g': g, 'Measurement': measurement, 'grid': grid, 'np': np,
           'quark_mom_fw': np.array([*kf, 0]), 'quark_mom_bw': np.array([*kb, 0]),
           'meson_mom': np.array([*(np.array(kf) + kb), 0]), 'Lt': info.size[3]}
    fw_sp = expression(source_path, 'prop_f_Pq', 0)
    fw_ss = expression(source_path, 'prop_f_Pq', 1)
    bar_ss = expression(source_path, 'prop_b_Pqbar')
    left_ss = expression(source_path, 'prop_b_Pqbar_src')
    corr = sorted((node for node in ast.walk(ast.parse(source_path.read_text()))
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'corr'), key=lambda node: node.lineno)
    sp_expr, ss_expr = [compile(ast.Expression(node.value), str(source_path), 'eval') for node in corr]
    block1, block2, temp1, temp2, c4_expr = [expression(source_path, name)
        for name in ('tmp_1', 'tmp_2', 'temp_1', 'temp_2', 'corr_ff')]
    phase_c4 = expression(source_path, 'Gw_bperp', -1)
    tensor = lambda value: g.tensor(np.asarray(value), g.spin_type)
    fw_summed = [None] * info.size[3]
    left_summed = [[None] * info.size[3] for _ in pairs]
    shape = fields[0].shape[:4]
    sp = np.zeros((len(sources), 16, 3 * 4 * 3, shape[0]), complex)
    c4 = np.zeros((4, len(insertions), 3, 4, shape[0]), complex)

    for coordinate in np.ndindex(shape):
        g.coordinate = coordinate
        t = coordinate[0]
        fw, bw, sink_bw, sink_fw = [Site(g, field, coordinate) for field in fields]
        env.update(prop_f=fw, prop_b=bw)
        forward = eval(fw_ss, env)
        fw_summed[t] = forward if fw_summed[t] is None else fw_summed[t] + forward
        env['prop_b_Pqbar'] = eval(bar_ss, env)
        for ipair, (src, sink) in enumerate(pairs.values()):
            env.update(temp_src=tensor(src), temp_sink=tensor(sink))
            left = eval(left_ss, env)
            old = left_summed[ipair][t]
            left_summed[ipair][t] = left if old is None else old + left
        env['prop_f_Pq'] = eval(fw_sp, env)
        for isrc, (label, src) in enumerate(sources.items()):
            env.update(pion_src={label: tensor(src)}, key=label)
            separation = 0
            for direction in range(3):
                for b in range(4):
                    for z in range(3):
                        env.update(bT_dir=direction, bT=b, bz_dir=2, bz=z)
                        sp[isrc, :, separation, t] += np.asarray(eval(sp_expr, env))[0, 0, :, 0]
                        separation += 1
        env.update(Gw=fw, Gw_dagger=sink_fw, i=0, j=0,
                   keys_src=['src'], keys_sink=['sink'], keys_gm1=['op'], keys_gm2=['op'])
        for direction in range(3):
            for b in range(4):
                shifted_bw = g.cshift(bw, direction, b)
                shifted_sink = g.cshift(sink_bw, direction, b)
                # The native script phases the entire sink field before shifting.
                g.coordinate = shifted_sink.coordinate
                env['Gw_bperp'] = shifted_sink
                phased_sink = eval(phase_c4, env)
                g.coordinate = coordinate
                env.update(Gw_bperp_tmp=phased_sink, Gw_bperp_dagger_tmp=shifted_bw)
                for ipair, (src, sink) in enumerate(c4_pairs.values()):
                    env.update(pion_src={'src': tensor(src)}, pion_sink={'sink': tensor(sink)})
                    env['tmp_1'], env['tmp_2'] = eval(block1, env), eval(block2, env)
                    for igamma, (gm1, gm2) in enumerate(insertions.values()):
                        env.update(Gamma1={'op': gm1}, Gamma2={'op': gm2})
                        env['temp_1'], env['temp_2'] = eval(temp1, env), eval(temp2, env)
                        c4[ipair, igamma, direction, b, t] += eval(c4_expr, env)
    ss = []
    for left in left_summed:
        env.update(prop_b_Pqbar_src=left, prop_f_Pq=fw_summed)
        ss.append(eval(ss_expr, env))
    return np.asarray(ss), sp, c4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpt-root', type=Path, required=True)
    parser.add_argument('--gpt-run-root', type=Path, required=True)
    parser.add_argument('--pyquda-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    install_stubs(args.pyquda_root)
    import pyquda_measurement_utils.pion_soft_factor_vibe_develop as soft
    from pyquda_measurement_utils.fermion_bilinear_basis import GAMMA_LABELS, PHYSICAL_FROM_PYQUDA

    g = NativeGPT(args.gpt_root, GAMMA_LABELS)
    physical = np.stack([gamma.array for gamma in g.sink_gammas])
    raw = np.stack(list(soft._raw_gamma_by_label.values()))
    np.testing.assert_array_equal(np.tensordot(PHYSICAL_FROM_PYQUDA, raw, axes=(1, 0)), physical)
    source = args.gpt_run_root / 'gpt/qTMD_softFF/64I_aurora/joint_run/qTMD_ff_4pt_parallel_contract.py'
    phase_source = args.gpt_run_root / 'gpt/gpt-pyquda/TEST_local_all/qTMD/pion_ff_class.py'
    qns = {'numpy': np, 'prod': prod, 'getArrayBackend': lambda: 'numpy',
           'arrayAsArray': lambda a, backend: np.asarray(a), 'arrayExp': lambda a, backend: np.exp(a), 'pi': pi}
    load_definitions(args.pyquda_root / 'pyquda_core/pyquda/field.py', {'evenodd', 'lexico'}, qns)
    load_definitions(args.pyquda_root / 'pyquda_utils/phase.py', {'MomentumPhase'}, qns)
    soft.phase.MomentumPhase = qns['MomentumPhase']
    soft.getGridSize = lambda: [1, 1, 1, 1]
    soft.mpi_print = lambda *a: None
    soft._reduce_spatial_projected = lambda a: a.copy()

    class Prop:
        def __init__(self, data, info, checkerboard=False):
            self.latt_info = info
            self.data = data if checkerboard else qns['evenodd'](data, [0, 1, 2, 3])

        def copy(self):
            return Prop(self.data.copy(), self.latt_info, True)

        def lexico(self, force_numpy):
            return qns['lexico'](self.data, [0, 1, 2, 3, 4])

    info = SimpleNamespace(size=[4, 3, 2, 2], global_size=[4, 3, 2, 2], grid_coord=[0]*4, mpi_rank=0)
    rng = np.random.default_rng(190)
    sources = {'5': g.gamma[5].array,
               'Z5-X5': (g.gamma['Z'] * g.gamma[5] - g.gamma['X'] * g.gamma[5]).array,
               'complex': rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))}
    pairs = {f'{name}__{label}': (src, physical[i])
             for name, src in sources.items() for i, label in enumerate(GAMMA_LABELS)}
    c4_pairs = {'Z5-X5__Z5-X5': (sources['Z5-X5'], sources['Z5-X5']),
                '5__5': (sources['5'], sources['5']),
                'X__Z5+X5': (g.gamma['X'].array, (g.gamma['Z'] * g.gamma[5] + g.gamma['X'] * g.gamma[5]).array),
                'complex__complex': (sources['complex'], rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))}
    pairs.update(c4_pairs)
    gamma_pairs = {label: (label, label) for label in GAMMA_LABELS}
    gamma_pairs.update({f'{a}__{b}': (a, b) for a, b in
                       [('X', 'Y'), ('Y', 'X'), ('Y5', 'X'), ('X', 'Y5'), ('T5', 'Z'), ('Z', 'T5')]})
    native_insertions = {key: tuple(g.sink_gammas[GAMMA_LABELS.index(label)] for label in labels)
                         for key, labels in gamma_pairs.items()}
    m = soft.pion_soft_factor({'quark_mom': [[0, 0, 0]], 'bT_dir': [0, 1, 2],
                              'bT_length': 3, 'bz_length': 2, 'tsep_list': [1],
                              'pion_channel_pairs': pairs, 'gamma_channel_pairs': gamma_pairs})
    signs = np.diag(PHYSICAL_FROM_PYQUDA).real
    c4_signs = np.array([signs[GAMMA_LABELS.index(a)] * signs[GAMMA_LABELS.index(b)]
                         for a, b in gamma_pairs.values()])
    rows, arrays = [], {}
    for seed in (191, 192, 193):
        rng = np.random.default_rng(seed)
        fields = [rng.normal(size=(2, 2, 3, 4, 4, 4, 3, 3))
                  + 1j*rng.normal(size=(2, 2, 3, 4, 4, 4, 3, 3)) for _ in range(4)]
        for kf, kb in (([0]*3, [0]*3), ([1, -1, 2], [-1, 0, 1])):
            case = f'seed{seed}_' + ('zero' if kf == [0]*3 else 'moving')
            props = [Prop(field, info) for field in fields]
            m.pion_channel_pairs = pairs
            ss = np.array([m.contract_wall_2pt(info, *props[:2], kf, kb, label) for label in pairs])
            sp = np.array([m.contract_tmdwf_check(info, *props[:2], list(np.array(kf)+kb), f'{label}__5')
                           for label in sources])
            m.pion_channel_pairs = c4_pairs
            c4 = m.contract_soft_factor(info, *props, list(np.array(kf)+kb))[0]
            expected = native_results(g, source, phase_source, fields, info, sources, pairs, c4_pairs, native_insertions, kf, kb)
            sp_physical = sp * signs[None, :, None, None]
            c4_physical = c4 * c4_signs[None, :, None, None, None]
            groups = [('2pt', list(pairs), ss, expected[0]),
                      ('qTMDWF', [f'{src}__{sink}' for src in sources for sink in GAMMA_LABELS],
                       sp_physical.reshape(-1, 36, 2), expected[1].reshape(-1, 36, 2)),
                      ('C4', [f'{pair}/{gamma}' for pair in c4_pairs for gamma in gamma_pairs],
                       c4_physical.reshape(-1, 3, 4, 2), expected[2].reshape(-1, 3, 4, 2))]
            for observable, channels, actual, native in groups:
                for channel, a, b in zip(channels, actual, native):
                    finite = np.isfinite(a).all() and np.isfinite(b).all()
                    close = np.isclose(a, b, rtol=1e-11, atol=1e-10)
                    norm = float(np.linalg.norm(b))
                    error = float(np.linalg.norm(a-b))
                    rows.append({'case': case, 'observable': observable, 'channel': channel,
                                 'values': a.size, 'max_abs_error': float(np.max(np.abs(a-b))),
                                 'relative_l2': error / norm if norm else error,
                                 'passed': bool(finite and close.all())})
            for name, value in [('2pt_pyquda', ss), ('2pt_gpt', expected[0]),
                                ('qTMDWF_pyquda_raw', sp), ('qTMDWF_pyquda_physical', sp_physical),
                                ('qTMDWF_gpt', expected[1]), ('C4_pyquda_raw', c4),
                                ('C4_pyquda_physical', c4_physical), ('C4_gpt', expected[2])]:
                arrays[f'{case}_{name}'] = value
            for prop, field in zip(props, fields):
                np.testing.assert_array_equal(prop.lexico(False), field)
            print(f'{case}: ' + ', '.join(f'{name} {len(channels)} channels' for name, channels, _, _ in groups), flush=True)

    source_files = [source, phase_source, Path(soft.__file__),
                    args.gpt_root / 'lib/gpt/core/tensor.py', args.gpt_root / 'lib/gpt/core/gamma.py',
                    args.gpt_root / 'lib/gpt/core/object_type/container.py',
                    args.gpt_root / 'lib/gpt/core/transform.py',
                    args.gpt_root / 'lib/cgpt/lib/slice_trace.h', args.gpt_root / 'lib/cgpt/lib/slice_trace_gpu.h',
                    args.pyquda_root / 'pyquda_utils/phase.py', args.pyquda_root / 'pyquda_core/pyquda/field.py']
    summary = {}
    for observable in ('2pt', 'qTMDWF', 'C4'):
        selected = [row for row in rows if row['observable'] == observable]
        summary[observable] = {'channels': len({row['channel'] for row in selected}),
                              'channel_case_checks': len(selected),
                              'values': sum(row['values'] for row in selected),
                              'max_abs_error': max(row['max_abs_error'] for row in selected),
                              'max_channel_relative_l2': max(row['relative_l2'] for row in selected),
                              'passed': all(row['passed'] for row in selected)}
    report = {'comparison': 'native_GPT_CPU_tensor_source_with_Python_site_loops',
              'compiled_cgpt_executed': False, 'mpi_executed': False,
              'shape_tzyx': [2, 2, 3, 4], 'seeds': [191, 192, 193],
              'momentum_pairs': [[[0]*3, [0]*3], [[1, -1, 2], [-1, 0, 1]]],
              'rtol': 1e-11, 'atol': 1e-10, 'gamma_basis_exact': True,
              'physical_sink_signs': dict(zip(GAMMA_LABELS, signs.tolist())),
              'summary': summary, 'channels': rows,
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'channels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / 'paired_arrays.npz', **arrays)
    print(json.dumps(summary, indent=2))
    return 0 if all(item['passed'] for item in summary.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
