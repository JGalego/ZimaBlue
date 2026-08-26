# The benchmark

`zimablue compare` answers "which planner wins, on my terms". The benchmark
answers a different question — **did anything change** — and that takes a suite
where nothing is negotiable:

```bash
zimablue bench --jobs 4 --out runs/bench
```

## What is frozen, and why

`zb-bench-v1` fixes the entries, the pools, the seeds and the duration, and it
names its entries as a literal list rather than asking the package what it
ships. A planner added later does not creep into the benchmark; it gets in
when the suite version is deliberately bumped, so a v1 number from March and a
v1 number from September measure the same thing.

| | |
|---|---|
| entries | every planner and reference controller shipped at v0.3.0, offline ones followed on odometry |
| pools | `rectangular`, `kidney`, `l_shaped` |
| seeds | 1, 2, 3 |
| duration | 15 simulated minutes per run |
| dirt | `autumn` on a `tracked` cleaner |

The whole suite is 189 runs. `--jobs 4` puts it well under an hour;
`--quick` runs a smoke tier that proves the pipeline in about a minute and
whose numbers mean nothing beyond that.

## What a result is

`run_bench()` returns the same `Comparison` the harness always produces —
median across seeds and pools, no scalar "best planner" — plus a header that
makes the run reproducible: suite name, ZimaBlue version, Python and numpy
versions, machine. `save()` writes it as JSON (per-trial scores, `null` where
a run never reached the mark), CSV (one row per trial) and a markdown
leaderboard.

Determinism is the package's standing contract: same ZimaBlue version, same
platform, same seed, same numbers. Two differing v1 results therefore mean a
real change — in the code or in the platform — not noise, which is what makes
the benchmark worth having.

## Reading it

The same warnings apply as in [planners](planners.md): the columns do not
collapse into one number, the truth-versus-odometry gap is a property of the
robot rather than the planner, and a middle-of-table ordering within a couple
of points is not a finding. The stable results are the ends of the table and
the column trade-offs.

In code:

```python
from zimablue.bench import BENCH_V1, run_bench

result = run_bench(BENCH_V1, jobs=4)
print(result.comparison.table())
result.save("runs/bench")
```

## Regression gates

A saved JSON result can be a CI baseline. Tolerances are explicit per metric;
the library does not hide a universal percentage that might be harmless for
energy and substantial for coverage.

```python
from pathlib import Path

from zimablue.bench import BenchTolerance, compare_benchmarks

current = run_bench(BENCH_V1, jobs=4)
gate = compare_benchmarks(
    current,
    "benchmarks/zb-bench-v1.json",
    {
        "coverage": BenchTolerance(absolute=0.01),
        "dirt": BenchTolerance(absolute=0.01),
        "energy": BenchTolerance(relative=0.03),
    },
)

gate.assert_passed()
Path("bench-gate.md").write_text(gate.to_markdown())
```

Checks use the median over the exact pool and seed trials named by the frozen
suite. Missing, duplicated or non-finite trial values fail completeness rather
than disappearing from an aggregate. The metric metadata determines direction:
lower energy and worst-gap values are improvements, while lower coverage is a
regression.

The baseline definition must equal the current definition. Comparing a quick
run with `zb-bench-v1`, or a v1 suite with a future v2, raises an error before
any score is interpreted.
