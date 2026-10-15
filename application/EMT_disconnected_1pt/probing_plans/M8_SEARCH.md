# 8^4 distance-8 linear multiplier search

Result: `colors=829`, `sigma=(1,8,246,310)`, `version=1`.
One complete Fourier basis requires 829 inversions per base noise.

Both the production torus checker and an independent enumeration of every
same-color site pair on all 4096 sites accept the plan. All 829 colors are used.
The shortest same-color distance is 9, attained for example by `(0,0,0,0)` and
`(3,3,2,1)`. This is a distance-8 plan, not distance-9.

## Optimality scope and lower bound

The objective is the smallest modulus in `c(x)=sigma.x mod n` on `[0,7]^4`
with periodic L1 distance. This is not a claim about arbitrary nonlinear
colorings, stochastic variance, or physical EMT execution cost.

The radius-4 torus ball contains 317 points and has diameter 8, so it forms a
317-vertex clique in the distance-8 graph. Fewer than 317 colors are impossible.
The exhaustive search rejected **every modulus 317--828** and found its first
solution at **829**. The single-thread run took approximately 132 seconds on
the available Jupyter CPU; this is search time, not EMT execution time.

## Exhaustive search

The saved source uses the same algorithm and completeness reductions as
[the M6 search](M6_SEARCH.md), replacing target 6 with 8 and the clipped
minimum-cost sentinel 7 with 9. Actual differences in each axis are enumerated
over `[-7,7]`, weighted by `min(abs(delta),8-abs(delta))`.

For each coefficient prefix, dynamic programming records the minimum weight
for every modular residue. Adding a coordinate is rejected if a nonzero
difference of weight at most 8 reaches residue zero. Costs above 8 cannot
participate in such a collision and are safely clipped to 9.

Coefficient signs can be folded by lattice reflections, and equal-extent axes
can be permuted to sort coefficients. Tuples with a coefficient coprime to n
are normalized to have first coefficient 1 by a bijective color relabeling.
Tuples without any coprime coefficient are also exhaustively searched; they
are not discarded for composite moduli. Each coefficient must have additive
order at least 8. Repeated folded coefficients cause distance-2 collisions.
Tuples with a common divisor with n reduce to an already excluded smaller
modulus, so they cannot be the first solution in this ascending search.

## Reproduction

From the develop repository root:

```bash
g++ -O3 -std=c++17 tests/reference/search_m8_8torus.cpp -o /tmp/xgao_m8_search
/tmp/xgao_m8_search 317 829
python -m pytest tests/test_disconnected_probing_math.py -q
```

The search prints `found=0` for 317--828, then
`colors=829 found=1 ... 1 8 246 310`. The independent Python test reads the JSON,
checks every same-color pair, and checks the 317-point clique separately from
the dynamic-programming search. The source saved in the repository was checked
byte-for-byte against the source used by the completed search.

This is an offline reference search. No GPU/MPI EMT run has been performed,
and no production solver, flow, or contraction code is changed by this plan.

After adding M8: independent mathematical tests **11 passed**; the related
CPU/mock regression suite **109 passed**; `git diff --check` passed.
