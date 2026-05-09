import numpy as np


def spatial_phase(momentum, coords, lattice_size):
    momentum = np.asarray(momentum, dtype=int)
    coords = np.asarray(coords, dtype=int)
    lattice_size = np.asarray(lattice_size, dtype=int)
    angle = 2j * np.pi * np.sum(momentum * coords / lattice_size, axis=-1)
    return np.exp(angle)


def dominant_source_momentum_for_emff_phase(pf, qext, lattice_size):
    axes = [np.arange(size) for size in lattice_size]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    current_phase = spatial_phase(qext, coords, lattice_size)
    sink_line_phase = np.conjugate(spatial_phase(pf, coords, lattice_size))

    scan_ranges = [range(-size // 2, size // 2) for size in lattice_size]
    best_momentum = None
    best_overlap = -1.0
    for px in scan_ranges[0]:
        for py in scan_ranges[1]:
            for pz in scan_ranges[2]:
                source_momentum = [px, py, pz]
                source_phase = spatial_phase(source_momentum, coords, lattice_size)
                overlap = abs(np.sum(current_phase * sink_line_phase * source_phase))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_momentum = source_momentum
    return best_momentum, best_overlap


def run_free_field_momentum_flow_test(lattice_size=(8, 8, 8), temporal_size=16):
    cases = [
        ([0, 0, 1], [0, 0, 2]),
        ([0, 0, 1], [0, 0, -2]),
        ([0, 0, 2], [0, 0, 4]),
    ]

    all_passed = True
    print("[EMFF free-field phase test]")
    print(f"lattice spatial size = {list(lattice_size)}, temporal size = {temporal_size}")
    for pf, qext in cases:
        expected_pi = [pf_i - q_i for pf_i, q_i in zip(pf, qext)]
        observed_pi, overlap = dominant_source_momentum_for_emff_phase(pf, qext, lattice_size)
        passed = observed_pi == expected_pi
        all_passed = all_passed and passed
        print(f"pf = {pf}")
        print(f"qext = {qext}")
        print(f"inferred pi = pf - qext = {expected_pi}")
        print(f"observed dominant source momentum = {observed_pi}")
        print(f"dominant overlap = {overlap:.6e}")
        print(f"{'PASS' if passed else 'FAIL'}")
    return all_passed


def test_pion_emff_free_field_momentum_flow():
    assert run_free_field_momentum_flow_test()


if __name__ == "__main__":
    raise SystemExit(0 if run_free_field_momentum_flow_test() else 1)
