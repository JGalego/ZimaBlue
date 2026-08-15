"""Batch experiments: the same scenario, many seeds.

One run tells you what happened once. A hundred runs tell you whether it
happens.  The output is deliberately boring -- an aggregate table plus enough
per-episode metadata to reproduce any individual failure exactly, which is the
only reason to keep the metadata at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zimablue._version import __version__
from zimablue.metrics import Metrics
from zimablue.scenarios import Scenario

__all__ = ["BatchResult", "EpisodeResult", "run_batch"]


@dataclass
class EpisodeResult:
    """One episode's outcome, plus what it takes to re-run it."""

    seed: int
    metrics: Metrics
    recording_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "recording": str(self.recording_path) if self.recording_path else None,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BatchResult:
    """Aggregate results over a set of episodes."""

    scenario: Scenario
    episodes: list[EpisodeResult] = field(default_factory=list)

    # -- aggregation -------------------------------------------------------
    def values(self, metric: str) -> np.ndarray:
        return np.array(
            [getattr(e.metrics, metric) for e in self.episodes if hasattr(e.metrics, metric)],
            dtype=float,
        )

    def stats(self, metric: str) -> dict[str, float]:
        values = self.values(metric)
        if values.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that ran to completion rather than dying early.

        "Success" is finishing on the duration or a target -- not a quality
        judgement. A run that flattens its battery at 40% coverage did not
        succeed, and averaging it in with the rest silently flatters the mean.
        """
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.metrics.completed) / len(self.episodes)

    @property
    def stuck_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.metrics.stuck_events > 0) / len(self.episodes)

    def worst(self, metric: str = "coverage", count: int = 3) -> list[EpisodeResult]:
        """The worst episodes by a metric -- the ones worth actually watching."""
        return sorted(self.episodes, key=lambda e: getattr(e.metrics, metric))[:count]

    # -- reporting ----------------------------------------------------------
    def summary(self) -> str:
        coverage = self.stats("coverage")
        dirt = self.stats("dirt_removed_fraction")
        runtime = self.stats("runtime")
        energy = self.stats("energy_consumed")
        lines = [
            f"episodes           {len(self.episodes)}",
            f"success_rate       {self.success_rate * 100:.1f} %",
            f"mean_coverage      {coverage['mean'] * 100:.1f} %  "
            f"(sd {coverage['std'] * 100:.1f}, "
            f"min {coverage['min'] * 100:.1f}, max {coverage['max'] * 100:.1f})",
            f"mean_dirt_removed  {dirt['mean'] * 100:.1f} %  "
            f"(sd {dirt['std'] * 100:.1f}, "
            f"min {dirt['min'] * 100:.1f}, max {dirt['max'] * 100:.1f})",
            f"mean_runtime       {runtime['mean'] / 60:.1f} min",
            f"mean_energy        {energy['mean']:.1f} Wh",
            f"stuck_rate         {self.stuck_rate * 100:.1f} %",
            f"mean_collisions    {self.stats('collisions')['mean']:.0f}",
        ]
        worst = self.worst("coverage", 1)
        if worst:
            lines.append(
                f"worst_episode      seed {worst[0].seed} "
                f"({worst[0].metrics.coverage * 100:.1f} % coverage)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zimablue_version": __version__,
            "scenario": self.scenario.to_dict(),
            "aggregate": {
                "episodes": len(self.episodes),
                "success_rate": self.success_rate,
                "stuck_rate": self.stuck_rate,
                **{
                    f"{metric}": self.stats(metric)
                    for metric in (
                        "coverage",
                        "wall_coverage",
                        "dirt_removed_fraction",
                        "cleaning_uniformity",
                        "revisits",
                        "distance_traveled",
                        "runtime",
                        "energy_consumed",
                        "collisions",
                    )
                },
            },
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Flat per-episode CSV, for when you want a spreadsheet or pandas."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "seed": e.seed,
                **{k: v for k, v in e.metrics.to_dict().items() if not isinstance(v, dict)},
            }
            for e in self.episodes
        ]
        if not rows:
            path.write_text("")
            return path
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path


def run_batch(
    scenario: Scenario,
    *,
    episodes: int = 10,
    seeds: Iterable[int] | None = None,
    record_dir: str | Path | None = None,
    on_episode: Callable[[int, EpisodeResult], None] | None = None,
) -> BatchResult:
    """Run ``episodes`` seeded episodes of ``scenario``.

    Seeds default to ``scenario.seed + i``, so a batch is itself reproducible
    from the scenario file alone.  Pass ``record_dir`` to keep every episode's
    ``.zbr``; otherwise recording is skipped, which is meaningfully faster and
    is the right default for a sweep of hundreds.
    """
    seed_list = list(seeds) if seeds is not None else [scenario.seed + i for i in range(episodes)]
    record_path = Path(record_dir) if record_dir else None
    if record_path:
        record_path.mkdir(parents=True, exist_ok=True)

    result = BatchResult(scenario=scenario)
    for index, seed in enumerate(seed_list):
        run = scenario.run(seed=seed, record=record_path is not None)
        saved: Path | None = None
        if record_path is not None and run.recording is not None:
            saved = run.save(record_path / f"{scenario.name}_seed{seed:04d}.zbr")
        episode = EpisodeResult(seed=seed, metrics=run.metrics, recording_path=saved)
        result.episodes.append(episode)
        if on_episode is not None:
            on_episode(index, episode)
    return result
