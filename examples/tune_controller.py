#!/usr/bin/env python3
"""Tune the shipped controller, before reaching for anything that learns.

    python examples/tune_controller.py
    python examples/tune_controller.py --objective coverage --iterations 40
    python examples/tune_controller.py --episodes 5 --minutes 8

``BaselineCoverage`` has five numbers in it worth arguing about. This searches
them with a (1+1) evolution strategy -- propose a mutation, keep it if it
scores better, shrink the step when it does not -- scoring each candidate on a
batch of seeds so a lucky episode cannot win.

The point is the comparison. This is a handful of CPU-minutes against a
GPU-day, and any policy trained through ``zimablue.rl`` has to beat *this*
number, not the untuned default. Search is also honest about the reward the
way training is: ``--objective`` picks what is being maximised, and the answer
changes depending on which you ask for.

No dependency on an optimisation library: a (1+1) ES in twenty lines is enough
for five parameters, and CMA-ES would be the next step rather than a different
idea.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import zimablue as zb
from zimablue.controllers.baseline import BaselineCoverage, BaselineTuning

# Name, low, high. Bounds are what the parameter means, not what the search
# would like: a cruise speed above 1.0 is not a faster robot, it is a clipped
# command, and a perimeter pass over half the run leaves no time for lanes.
PARAMETERS = (
    ("cruise_speed", 0.4, 1.0),
    ("lane_overlap", 0.0, 0.5),
    ("perimeter_time", 0.0, 0.5),
    ("wall_threshold", 0.15, 0.6),
    ("wall_standoff", 0.2, 0.6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--dirt", default="autumn")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=3, help="seeds per evaluation")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--objective", default="dirt", choices=("dirt", "coverage"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def score(tuning: BaselineTuning, args: argparse.Namespace) -> float:
    """Mean over a fixed set of seeds. Fixed, so candidates are comparable."""
    values = []
    for episode in range(args.episodes):
        result = zb.Simulation(
            pool=args.pool,
            dirt=args.dirt,
            controller=BaselineCoverage(tuning),
            seed=args.seed + episode,
            record=False,
        ).run(minutes=args.minutes)
        metrics = result.metrics
        values.append(
            metrics.dirt_removed_fraction if args.objective == "dirt" else metrics.coverage
        )
    return float(np.mean(values))


def vector(tuning: BaselineTuning) -> np.ndarray:
    return np.array([getattr(tuning, name) for name, _, _ in PARAMETERS])


def build(values: np.ndarray) -> BaselineTuning:
    clipped = [np.clip(v, low, high) for v, (_, low, high) in zip(values, PARAMETERS, strict=True)]
    return replace(
        BaselineTuning(), **{n: float(v) for (n, _, _), v in zip(PARAMETERS, clipped, strict=True)}
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    spans = np.array([high - low for _, low, high in PARAMETERS])

    best = vector(BaselineTuning())
    best_score = score(build(best), args)
    baseline_score = best_score
    step = 0.25  # of each parameter's range

    print(f"{args.pool} pool, {args.dirt} dirt, {args.minutes:g} min x {args.episodes} seeds")
    print(f"maximising {args.objective}\n")
    print(f"{'iter':>4} {'step':>6} {args.objective:>9}  parameters")
    print(f"{'0':>4} {'-':>6} {best_score:9.3%}  (defaults)")

    for iteration in range(1, args.iterations + 1):
        candidate = best + rng.normal(0.0, step, len(PARAMETERS)) * spans
        candidate_score = score(build(candidate), args)
        if candidate_score > best_score:
            best, best_score = candidate, candidate_score
            step = min(step * 1.5, 0.4)  # it worked; try further afield
            marker = "  <-- kept"
        else:
            step = max(step * 0.85, 0.02)  # it did not; look closer to home
            marker = ""
        print(f"{iteration:>4} {step:6.3f} {candidate_score:9.3%}{marker}")

    tuned = build(best)
    gain = best_score - baseline_score
    print(f"\n{args.objective}: {baseline_score:.1%} -> {best_score:.1%}  ({gain:+.1%})")
    print("\ntuned:")
    for name, _, _ in PARAMETERS:
        print(f"  {name:16s} {getattr(BaselineTuning(), name):6.3f} -> {getattr(tuned, name):6.3f}")

    print(
        "\nThat took a few CPU-minutes. It is the number a trained policy has to\n"
        "beat -- and searching for coverage instead of dirt finds a different\n"
        "setting, which is the same disagreement the rest of the library is about."
    )


if __name__ == "__main__":
    main()
