import argparse
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser(description="Validate the completed strange HP512 benchmark")
parser.add_argument("--configuration", type=int, default=1050)
parser.add_argument("--n-base-noise", type=int, default=1)
args = parser.parse_args()
benchmark_dir = Path(__file__).resolve().parent
files = sorted((benchmark_dir / "data").rglob("*.bases.h5"))
matches = []
for path in files:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["config_num"]) == args.configuration:
            matches.append(path)
if len(matches) != 1:
    raise RuntimeError(f"expected one completed base file, found {len(matches)}")

path = matches[0]
with h5py.File(path, "r") as handle:
    expected_attrs = {
        "mass": -0.015,
        "noise_scheme": "hierarchical_probing",
        "vectors_per_base": 512,
        "hp_ordering": "nested_xyzt_checkerboard_v1",
        "spin_color_dilution": "point",
        "flavor_convention": "single_strange_flavor_trace_for_this_dirac_operator",
        "n_base_noise": args.n_base_noise,
        "effective_n_inversions": args.n_base_noise * 512 * 12,
        "flow_type": "symanzik",
        "flow_epsilon": 0.09,
        "flow_steps": 10,
        "raw_per_vector_stored": False,
    }
    for name, expected in expected_attrs.items():
        if handle.attrs[name] != expected:
            raise RuntimeError(f"unexpected {name}: {handle.attrs[name]!r}")
    np.testing.assert_array_equal(handle.attrs["global_lattice_size"], [64] * 4)
    expected_q = [
        [x, y, z, 0]
        for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)
    ]
    np.testing.assert_array_equal(handle.attrs["qext"], expected_q)
    np.testing.assert_array_equal(handle["base/base_noise_index"], np.arange(args.n_base_noise))
    expected_shapes = {
        "local_bilinear": (16, 125, 11, 64),
        "derivative_bilinear": (16, 4, 125, 11, 64),
        "flowed_noise_norm": (125, 11, 64),
        "ringed/kinetic": (11, 64),
        "ringed/kinetic_spacetime": (11,),
    }
    expected_shapes.update({
        f"Tmunu/T{mu}{nu}": (125, 11, 64)
        for mu in range(1, 5) for nu in range(mu, 5)
    })
    for name, shape in expected_shapes.items():
        values = handle[f"avg/{name}"][...]
        if values.shape != shape or not np.isfinite(values).all():
            raise RuntimeError(f"invalid avg/{name}: shape={values.shape}")
        bases = handle[f"base/{name}"]
        if bases.shape != (args.n_base_noise,) + shape:
            raise RuntimeError(f"invalid base/{name}: shape={bases.shape}")
        total = np.zeros(shape, dtype=np.complex128)
        for index in range(args.n_base_noise):
            base = bases[index]
            if not np.isfinite(base).all():
                raise RuntimeError(f"non-finite base/{name}[{index}]")
            total += base
        np.testing.assert_allclose(values, total / args.n_base_noise, rtol=1e-12, atol=1e-14)
    if not np.any(handle["avg/flowed_noise_norm"][...]):
        raise RuntimeError("flowed noise norm is identically zero")
    if bool(handle.attrs["sem_defined"]) != (args.n_base_noise > 1):
        raise RuntimeError("unexpected independent-base SEM status")

print(f"PASS strange HP512/SC12 benchmark: {path}", flush=True)
