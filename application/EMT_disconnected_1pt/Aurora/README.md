# Aurora: a076 strange disconnected quark EMT 1pt

This example follows the timed live benchmark in
`run/l64c64a076/EMT_proton_disconnected_strange/benchmark`, with the validated
B24 setting synchronized on 2026-10-16. It retains the HP512 production
estimator and coarse timing subclass, not the NP2 profiling workload or its
benchmark-only detailed contraction override. The launcher imports
the measurement repository containing this application. It does not depend on
files in the live production directory or modify the original measurement tree.

## Settings

| Setting | Default |
|---|---|
| Ensemble / configuration | l64c64a076, 64^4 / 1050 |
| Mass / csw / tolerance / maxiter | -0.015 / 1.0372 / 1e-10 / 5000 |
| Noise | One counter-based Z4 independent base, stream salt 0 |
| Probing | HP512, nested_xyzt_checkerboard_v1 |
| Spin-color dilution | point: all 4 spins x 3 colors |
| Solves per base | 512 x 12 = 6144 |
| Shard interval | 768 solves: 64 complete probes, 8 shards per base |
| Fermion-flow batch size | 24 diluted sources (48 flowed spinor fields) |
| Flow | Symanzik, epsilon 0.09, 10 intervals, 11 output times |
| Momentum | 125 vectors, spatial components -2 through 2 |
| Preprocessing / boundary | HYP(1,0.75,0.6,0.3), antiperiodic time |
| Nodes / MPI ranks / ranks per node | 8 / 64 / 8 |
| MPI geometry / MG | 2.2.4.4 / 4.4.4.4;4.4.2.2 |
| PBS queue / walltime | debug-scaling / 01:00:00 |

This is a single-flavor strange-quark loop, not a proton 2pt/3pt measurement.
The 12 spin-color contributions are summed, HP vectors are averaged, and only
independent Z4 bases are statistical samples. There is no low-mode deflation.

Configuration, base count, batch size, geometry and MG are near the top of
`run_quark_1pt.sh`; physics, HP and shard constants are at the top of
`Pyquda_EMT_disconnected_quark_1pt.py`. The mass is deliberately -0.015, not the
connected light-quark value -0.049. The validator is specific to these settings;
update it if changing the physics, probing, shape or output defaults.

## Recommended multi-source batching

Use multiple sources per flow batch when GPU memory permits. Both this shell
launcher and the Python driver's `--flow-batch-size` default are **24** for the
tested eight-node Aurora setup. One HP probe has 12 spin-color-diluted sources;
B24 groups two probes for flow, but still performs 24 sequential inversions.
The gauge-flow trajectory and resident flowed gauge are shared within a batch.
The current library uses host-array `Allreduce(SUM)` for quark-loop projection,
without a legacy gather/broadcast fallback.

B24 divides the 768-solve shard into 32 full batches and the HP512 base into
256 full batches. It does not change the estimator, spin-color sum, HP average,
base completion boundaries, or output format. It is a scheduling parameter,
not an independent-base count or a multi-right-hand-side inversion setting.

The live B4/B8/B12/B24 sweep (job 8932633) completed two NP2 bases per setting.
B24 was fastest. The subsequent B24 Allreduce test (8933007) measured about
4.23 seconds per diluted source after excluding its first batch, compared with
6.44 seconds for B24 with gather/broadcast and the historical 19.85 seconds for
B1. The full improvement combines batching and Allreduce, not batching alone.
All 31 base/average datasets agreed with the original B24 output within
`rtol=1e-10, atol=1e-12` (maximum absolute difference `1.33e-15`). These are
small-sample compute timings, excluding initialization and finalization, not a
complete HP512 runtime guarantee or an independent physics validation.

Flow requires multiple work buffers for each of the `2B` spinors. On a different
lattice, MPI geometry, MG hierarchy or accelerator, start smaller and increase
B while checking numerical agreement and peak memory; do not assume B24 is
portable to Perlmutter. Reduce B if memory is insufficient. There is no automatic
OOM fallback. Keep a warm QUDA cache across serial tests; new B values may still
require tuning. The application continues to use its own `cache/` directory.

## Environment and launch

The launcher sources
`/lus/flare/projects/StructNGB/xgao/software_gradientflow_261009/activate-pyquda-develop.sh`
in a fresh environment, then puts this checkout first on `PYTHONPATH`.
It uses the rebuilt QUDA ff4879bad stack with dpnp/SYCL, oneAPI 2026.1.0 and
MPICH prd/5.0.0.aurora_test.51a9474. Gauge files are read from:

```text
/lus/flare/projects/StructNGB/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.{config}.coulomb.1e-14
```

From this application's Aurora directory:

```bash
mkdir -p log
qsub submit_quark_1pt.pbs
```

Alternatively run `bash run_quark_1pt.sh` inside an allocation with at least
eight distinct hosts. The launcher selects its first eight hosts, checks tile
visibility on all 64 ranks, then runs the measurement and output validator.
PBS must be submitted from this directory because it uses `PBS_O_WORKDIR`.
Review account, queue and walltime before submitting; the one-hour request is a
diagnostic benchmark, not enough time promised for a full 6144-solve base.
No task was submitted when adding this example.

The local affinity wrapper assigns integer FLAT Level Zero tiles. The driver
passes its local `cache/` explicitly; no cache is copied from production.
Cold tuning and first-inversion warmup can add substantial startup time.

## Output, finalization and restart

- Mean-only shards: `data/EMTquarkLoop/shards/`.
- Complete-base averages: `data/EMTquarkLoop/*.bases.h5`.
- Base-level completion records: `data/sample_log_disconnected/`.
- Per-source timing and solver output: `log/strange_cfg1050.o/.e`.
- Tile precheck, validation and PBS output: `log/`.
- Allocation host list: `scripts/hosts`; QUDA cache: `cache/`.

Raw per-inversion arrays are disabled. After measurement returns, all ranks
enter a barrier and rank zero aggregates complete independent bases. The shell
then runs `validate_strange.py` to check attributes, shapes, finite values and
base/mean consistency. One base has undefined SEM, not a zero statistical error.
This is internal consistency checking, not comparison to an independent physics
result. Until a complete base exists, that validation cannot pass.

Resume uses the existing base-level sample log only. Completed bases are skipped;
an interrupted base is recomputed, even if some of its shards already exist.
A timeout before 768 solves produces no completed shard. Do not launch two jobs
in this directory concurrently or change physics/estimator settings while
reusing its data and sample logs. Logs are overwritten on rerun; preserve them
first if needed. Runtime artifacts are excluded by the local `.gitignore`.

## Timing and validation limits

The benchmark-local subclass prints `strange_source_compute`,
`strange_source_flow`, and `strange_source_contract` records after every batch.
Batch size one reports individual diluted-source time; larger batches report
per-source averages. `first_batch=True` marks warmup. Inversion, gauge restore,
flow, contraction and remaining batch overhead are separated. Disk writes,
noise preparation and MG initialization are outside the source compute timer;
the original shard timers still report I/O. Timing does not change the estimator.

The source benchmark completed NP2/SC12 bases and output/reference validation
with B24 and Allreduce on Aurora. That does not certify a complete HP512/SC12
base or independent physical correctness. The application copy is checked by CPU/mock wiring tests and syntax
checks; it has not been submitted as a new GPU validation job. The timing
subclass uses private measurement hooks, so recheck it when updating the library.
