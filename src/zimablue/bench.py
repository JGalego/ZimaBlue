"""The benchmark: one command, the same numbers.

A comparison run ad hoc answers "which planner won today, on my machine, with
whatever I typed". A benchmark answers a different question -- "did anything
change" -- and for that every knob has to be frozen: which entries, which
pools, which seeds, how long. This module freezes them.

The suite is versioned, not living. ``BENCH_V1`` names its entries as a
literal tuple rather than calling :func:`~zimablue.planners.compare.default_entries`,
so a planner added to the package later does not silently change what the
benchmark measures; it gets in when the suite version is deliberately bumped,
and results from different suite versions are never comparable by accident.

Determinism is the package's standing contract -- same ZimaBlue version, same
platform, same seed, same numbers -- so a benchmark result is reproducible
from its JSON header alone::

    zimablue bench --jobs 4 --out runs/bench

or in code::

    from zimablue.bench import run_bench

    result = run_bench(jobs=4)
    print(result.comparison.table())
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zimablue._version import __version__
from zimablue.planners.compare import Comparison, compare

__all__ = ["BENCH_QUICK", "BENCH_V1", "BenchDefinition", "BenchResult", "run_bench"]


@dataclass(frozen=True)
class BenchDefinition:
    """Everything a benchmark run is allowed to know, spelled out."""

    name: str
    entries: tuple[str, ...]
    pools: tuple[str, ...]
    seeds: tuple[int, ...]
    minutes: float
    dirt: str = "autumn"
    robot: str = "tracked"

    @property
    def runs(self) -> int:
        return len(self.entries) * len(self.pools) * len(self.seeds)

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.entries)} entries on {', '.join(self.pools)}, "
            f"seeds {self.seeds[0]}..{self.seeds[-1]}, {self.minutes:g} min each "
            f"({self.runs} runs)"
        )


BENCH_V1 = BenchDefinition(
    name="zb-bench-v1",
    # The package's planners at v0.3.0, written out. default_entries() would
    # track additions and quietly redefine the benchmark; this does not.
    entries=(
        "baseline_coverage",
        "random_bounce",
        "systematic",
        "spiral_stc",
        "full_stc",
        "bsa",
        "ba_star",
        "brick_and_mortar",
        "binn",
        "epsilon_star",
        "ppcpp",
        "frontier",
        "smc",
        "boustrophedon@odometry",
        "sweep_optimal@odometry",
        "trapezoidal@odometry",
        "boustrophedon_cells@odometry",
        "morse@odometry",
        "contour@odometry",
        "wavefront@odometry",
        "spanning_tree@odometry",
    ),
    pools=("rectangular", "kidney", "l_shaped"),
    seeds=(1, 2, 3),
    minutes=15.0,
)

BENCH_QUICK = BenchDefinition(
    name="zb-bench-quick",
    # A smoke tier: proves the pipeline end to end in about a minute. Its
    # numbers mean nothing beyond "the machinery still runs".
    entries=("baseline_coverage", "random_bounce", "bsa"),
    pools=("rectangular",),
    seeds=(1,),
    minutes=2.0,
)


@dataclass
class BenchResult:
    """A benchmark run: the comparison plus enough header to reproduce it."""

    definition: BenchDefinition
    comparison: Comparison

    def header(self) -> dict[str, Any]:
        return {
            "bench": self.definition.name,
            "zimablue_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "definition": asdict(self.definition),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.header(),
            "trials": [
                {
                    "planner": trial.planner,
                    "pool": trial.pool,
                    "seed": trial.seed,
                    "scores": {k: _jsonable(v) for k, v in trial.scores.items()},
                }
                for trial in self.comparison.trials
            ],
        }

    def to_markdown(self) -> str:
        """The leaderboard as a pipe table, best value per column in bold."""
        comparison = self.comparison
        dims = comparison.dimensions
        lines = [
            f"<!-- {self.definition.name} · zimablue {__version__} · "
            f"reproduce with: zimablue bench -->",
            "",
            "| planner | " + " | ".join(d.label for d in dims) + " |",
            "|---|" + "|".join("---:" for _ in dims) + "|",
        ]
        raw = {
            planner: [comparison.score(planner, d.key) for d in dims]
            for planner in comparison.planners
        }
        winners = []
        for j, dim in enumerate(dims):
            finite = [v for p in comparison.planners if np.isfinite(v := raw[p][j])]
            winners.append((max(finite) if dim.better > 0 else min(finite)) if finite else None)
        for planner in comparison.planners:
            cells = []
            for j, dim in enumerate(dims):
                value = raw[planner][j]
                text = dim.format(value)
                best = winners[j]
                if best is not None and np.isfinite(value) and np.isclose(value, best):
                    text = f"**{text}**"
                cells.append(text)
            lines.append(f"| `{planner}` | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            f"Median over {len(comparison.pools)} pool(s) and "
            f"{len(self.definition.seeds)} seed(s); {len(comparison.trials)} runs of "
            f"{self.definition.minutes:g} simulated minutes each."
        )
        return "\n".join(lines)

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write the result as JSON, CSV and markdown under ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": directory / f"{self.definition.name}.json",
            "csv": directory / f"{self.definition.name}.csv",
            "markdown": directory / f"{self.definition.name}.md",
        }
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2))
        self.comparison.to_csv(paths["csv"])
        paths["markdown"].write_text(self.to_markdown() + "\n")
        return paths


def _jsonable(value: float) -> float | None:
    # JSON has no inf or nan; "never reached" serialises as null.
    return float(value) if np.isfinite(value) else None


def run_bench(
    definition: BenchDefinition = BENCH_V1,
    *,
    jobs: int = 1,
    on_result: Any = None,
) -> BenchResult:
    """Run a benchmark suite exactly as defined."""
    comparison = compare(
        definition.entries,
        pools=definition.pools,
        seeds=definition.seeds,
        minutes=definition.minutes,
        dirt=definition.dirt,
        robot=definition.robot,
        jobs=jobs,
        on_result=on_result,
    )
    comparison.label = definition.name
    return BenchResult(definition=definition, comparison=comparison)
