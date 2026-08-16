#!/usr/bin/env python3
"""Run one scenario many times and ask whether the result holds up.

    python examples/batch_experiment.py
    python examples/batch_experiment.py --episodes 40 --scenario kidney

A single run tells you what happened once. A controller that scores 80% on
seed 42 and 45% on seed 43 has not been measured, it has been sampled -- so
the useful unit of evidence is a batch, and the useful output is a spread
rather than a number.

The last section is the point of the whole exercise: a batch keeps enough
metadata to reproduce its own worst episode exactly.
"""

from __future__ import annotations

import argparse
import statistics

import zimablue as zb
from zimablue.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="kidney", help="path or built-in name")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--minutes", type=float, default=15.0)
    args = parser.parse_args()

    # A scenario is a YAML file, or the name of one bundled with the package.
    # Loading gives a plain object you can edit before running.
    scenario = zb.load_scenario(args.scenario)
    scenario.duration = args.minutes * 60.0

    print(f"scenario   {scenario.name}")
    print(f"  pool     {scenario.pool}")
    print(f"  robot    {scenario.robot}")
    print(f"  dirt     {scenario.dirt}")
    print(f"  control  {scenario.controller}")
    print(f"  each episode is {scenario.duration / 60:.0f} simulated minutes\n")

    print(f"running {args.episodes} episodes, one per seed...")
    batch = run_batch(
        scenario,
        episodes=args.episodes,
        on_episode=lambda i, ep: print(
            f"  seed {ep.seed:3d}  coverage {ep.metrics.coverage:5.1%}"
            f"  dirt {ep.metrics.dirt_removed_fraction:5.1%}"
        ),
    )

    print()
    print(batch.summary())

    # ------------------------------------------------------------------
    # A mean on its own hides the thing you actually care about.
    # ------------------------------------------------------------------
    coverage = batch.values("coverage")
    spread = max(coverage) - min(coverage)
    print("\nspread")
    print(f"  coverage  {min(coverage):.1%} to {max(coverage):.1%}  (range {spread:.1%})")
    if len(coverage) > 1:
        print(f"  stdev     {statistics.stdev(coverage):.1%}")
    print(f"  success   {batch.success_rate:.0%}")
    print(f"  stuck     {batch.stuck_rate:.0%} of episodes")

    if spread > 0.15:
        print("\n  That range is wide enough that a single run would have told you")
        print("  very little. This is why the benchmark tables use batches.")

    # ------------------------------------------------------------------
    # Reproducing a failure. The seed is the whole story: same version, same
    # platform, same scenario, same seed gives a bit-identical run.
    # ------------------------------------------------------------------
    worst = batch.worst("coverage", count=1)[0]
    print(f"\nworst episode was seed {worst.seed} at {worst.metrics.coverage:.1%} coverage")
    print("re-running just that one, with a recording this time...")

    rerun = zb.Simulation(
        pool=scenario.pool,
        robot=scenario.robot,
        dirt=scenario.dirt,
        controller=scenario.controller,
        seed=worst.seed,
        scenario_name=f"{scenario.name}_worst",
    ).run(seconds=scenario.duration)

    match = rerun.metrics.coverage == worst.metrics.coverage
    print(f"  reproduced coverage {rerun.metrics.coverage:.4%}  identical: {match}")
    if not match:
        raise SystemExit("determinism contract violated -- please file a bug")

    path = rerun.save(f"runs/{scenario.name}_worst_seed{worst.seed}.zbr")
    print(f"  saved {path}")
    print(f"\n  watch what went wrong:  zimablue replay {path}")


if __name__ == "__main__":
    main()
