# 8^4 distance-6 linear multiplier search

Result: `colors=411`, `sigma=(1,8,64,102)`, `version=1`.
The plan was independently verified both by the production torus checker and
by enumeration of every same-color site pair on all 4096 sites. All 411 colors
are used. The shortest same-color distance is 7, for example between
`(0,0,0,0)` and `(3,0,0,4)`. This is a distance-6 plan, not distance-7.

## Meaning of optimal

The objective is the smallest modulus/color count in
`c(x)=sigma.x mod n`, with actual coordinates in `[0,7]^4` and periodic L1
distance. It is not minimum physical variance, wall time, or the chromatic
number over all nonlinear colorings. GPU/MPI tests have not been run.

A radius-3 ball has 129 sites and diameter 6, so it is a clique in the
distance-6 graph. Hence fewer than 129 colors are impossible for any coloring.
The saved search exhausts moduli **129 through 410** without a legal plan;
the first success is **411**. The single-thread search took approximately
4.3 seconds on the available Jupyter CPU; this is not EMT execution time.

## Search completeness and pruning

For each coordinate, actual differences are integers in `[-7,7]`; their torus
weight is `min(abs(delta),8-abs(delta))`. Coordinate differences can be realized
independently in the four axes.

After choosing a coefficient prefix, a dynamic-programming array stores the
minimum weight of a difference yielding each modular residue, capped at 7.
Adding a coefficient is rejected exactly when a nonzero difference of total
weight at most 6 reaches residue zero. The all-zero difference is not rejected.
Discarding weights above 6 is safe because all remaining coordinate weights
are nonnegative. The accepted prefix is extended by all 15 actual differences.

The exhaustive coefficient domain uses these safe reductions:

- Coefficients are residues modulo n. Each must have additive order at least
  8, since all distinct sites along one axis are within distance 4.
- Axis reflections change the corresponding coefficient sign plus an irrelevant
  constant. Equal extents allow axis permutations. Thus coefficients can be
  folded into `[1,floor(n/2)]` and sorted. Equal folded coefficients would
  produce a distance-2 collision and can be excluded.
- If any coefficient is coprime to n, multiplication by its modular inverse
  relabels colors bijectively; after reflections/permutation, the first
  coefficient can be fixed at 1. All possible remaining coefficients are tested.
- Tuples containing no coprime coefficient are also exhaustively tested; this
  case is not silently discarded for composite moduli.
- Tuples with `gcd(n,*sigma)>1` reduce to a smaller modulus. They cannot be the
  first solution in an ascending search starting at the proven clique bound.

## Reproduction

From the develop repository root:

```bash
g++ -O3 -std=c++17 tests/reference/search_m6_8torus.cpp -o /tmp/xgao_m6_search
/tmp/xgao_m6_search 129 411
python -m pytest tests/test_disconnected_probing_math.py -q
```

The program prints `found=0` for each modulus 129--410, followed by
`colors=411 found=1 ... 1 8 64 102`. The separate Python test checks the saved
JSON against actual site pairs and verifies the clique bound independently of
the dynamic-programming implementation. The search is an offline reference,
not part of the MPI measurement runtime.

After adding M6: independent mathematical tests **10 passed**; the full related
CPU/mock regression suite **108 passed**; `git diff --check` passed. The saved
C++ search was recompiled and reproduced the 129--410 exclusions and 411 result.
