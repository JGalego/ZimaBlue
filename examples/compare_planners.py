#!/usr/bin/env python3
"""Run every coverage planner in the package and compare them on every axis.

    python examples/compare_planners.py                       # the default sweep
    python examples/compare_planners.py --minutes 30          # let them finish
    python examples/compare_planners.py --plots out/          # write the figures
    python examples/compare_planners.py --localisation both   # plan vs plan+odometry

Every planner, on every pool, measured every way the harness knows. The
point is not to crown one. It is that the ranking changes with the pool and with which
column you read, and that the two columns nobody reports -- how much of the
path was wasted overlap, and how much the robot had to turn -- separate
planners that look identical on coverage.

The run takes a while: it is one full simulation per planner per pool. Pass
``--jobs`` to spread it over cores, and ``--minutes`` low for a smoke test.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from zimablue.planners.compare import compare, default_entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", nargs="+", default=["rectangular", "kidney", "l_shaped"])
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--localisation",
        default="odometry",
        choices=("odometry", "truth", "both"),
        help="how the offline planners' paths are followed",
    )
    parser.add_argument("--only", nargs="+", default=None, help="just these planners")
    parser.add_argument("--plots", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None, help="pickle the comparison here")
    args = parser.parse_args(argv)

    entries = tuple(args.only) if args.only else default_entries(localisation=args.localisation)
    print(
        f"{len(entries)} planners x {len(args.pools)} pools x {len(args.seeds)} seeds, "
        f"{args.minutes:.0f} min each"
    )

    done = [0]

    def tick(trial):
        done[0] += 1
        total = len(entries) * len(args.pools) * len(args.seeds)
        print(
            f"  [{done[0]:>3}/{total}] {trial.planner:<26} {trial.pool:<12}"
            f" {trial.scores['coverage']:.1%}",
            flush=True,
        )

    result = compare(
        entries,
        pools=tuple(args.pools),
        seeds=tuple(args.seeds),
        minutes=args.minutes,
        jobs=args.jobs,
        on_result=tick,
    )

    for pool in result.pools:
        print(f"\n{pool}\n{result.table(pool)}")
    if len(result.pools) > 1:
        print(f"\nall pools\n{result.table()}")

    if args.csv:
        result.to_csv(args.csv)
        print(f"\n{args.csv}")
    if args.save:
        args.save.write_bytes(pickle.dumps(result))
        print(args.save)
    if args.plots:
        from zimablue.planners.plots import (
            plot_comparison,
            plot_curves,
            plot_matrix,
            plot_paths,
            plot_plans,
        )

        args.plots.mkdir(parents=True, exist_ok=True)
        plot_matrix(result).savefig(args.plots / "matrix.png", dpi=110)
        plot_comparison(result).savefig(args.plots / "comparison.png", dpi=110)
        for pool in result.pools:
            plot_paths(result, pool=pool).savefig(args.plots / f"paths_{pool}.png", dpi=110)
            plot_curves(result, pool=pool).savefig(args.plots / f"curves_{pool}.png", dpi=110)
            plot_plans(pool).savefig(args.plots / f"plans_{pool}.png", dpi=110)
        print(f"figures in {args.plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
