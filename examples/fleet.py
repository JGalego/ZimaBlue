#!/usr/bin/env python3
"""Several cleaners in one pool.

    python examples/fleet.py                              # three robots, one pool
    python examples/fleet.py --robots 4 --pool l_shaped
    python examples/fleet.py --controllers darp+sweep_optimal
    python examples/fleet.py --plots out/                 # the fleet views
    python examples/fleet.py --compare --jobs 4           # every method, scored
    python examples/fleet.py --scaling                    # 1, 2, 3, 4 robots

A fleet is not N runs added together: the robots share a dirt field, they get
in each other's way, and what they know about each other they had to say out
loud. The numbers that matter are therefore not coverage -- they are speedup
against the best single member, overlap, and how evenly the work fell.

``--controllers`` takes a controller name that every robot runs (``bsa``,
``auction``, ``binn_swarm``), a partition and a planner joined by ``+``
(``darp+sweep_optimal``), or ``mstc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zimablue.fleet import Fleet
from zimablue.planners.cooperative import mstc
from zimablue.planners.partition import partitioned


def build(spec: str):
    if spec == "mstc":
        return mstc(backtracking=True)
    if spec == "mstc_nobt":
        return mstc(backtracking=False)
    if "+" in spec:
        method, planner = spec.split("+", 1)
        return partitioned(method, planner)
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--controllers", default="darp+sweep_optimal")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dirt", default="autumn")
    parser.add_argument(
        "--comms-range", type=float, default=None, help="metres; unlimited by default"
    )
    parser.add_argument("--no-share", action="store_true", help="a fleet of strangers")
    parser.add_argument("--plots", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None, help="write a .zbr recording")
    parser.add_argument("--compare", action="store_true", help="score every method")
    parser.add_argument("--scaling", action="store_true", help="1..N robots, same pool")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(args)
    if args.scaling:
        return _scaling(args)

    import math

    fleet = Fleet(
        pool=args.pool,
        robots=args.robots,
        dirt=args.dirt,
        controllers=build(args.controllers),
        seed=args.seed,
        share=not args.no_share,
        comms_range=args.comms_range if args.comms_range is not None else math.inf,
    )
    result = fleet.run(minutes=args.minutes)
    print(f"\n{args.controllers} on {args.pool}, {args.robots} robots, {args.minutes:.0f} min")
    print(result.summary())

    if args.save:
        print(f"\n{result.save(args.save)}")
    if args.plots:
        from zimablue.fleetplots import plot_fleet, plot_overlap, plot_paths, plot_territory

        args.plots.mkdir(parents=True, exist_ok=True)
        plot_fleet(result).savefig(args.plots / "fleet.png", dpi=110)
        plot_paths(result).savefig(args.plots / "fleet_paths.png", dpi=110)
        plot_territory(result).savefig(args.plots / "fleet_territory.png", dpi=110)
        plot_overlap(result).savefig(args.plots / "fleet_overlap.png", dpi=110)
        print(f"figures in {args.plots}")
    return 0


def _scaling(args) -> int:
    """What the second, third and fourth robot are actually worth.

    The interesting column is not coverage, which of course goes up. It is
    speedup: if it grows more slowly than the robot count, the fleet is paying
    for machines that spend their time avoiding each other.
    """
    print(
        f"\n{'robots':>7}{'coverage':>10}{'dirt':>8}{'speedup':>9}{'overlap':>9}{'balance':>9}"
        f"{'bumps':>7}"
    )
    print("-" * 59)
    for count in range(1, args.robots + 1):
        result = Fleet(
            pool=args.pool,
            robots=count,
            dirt=args.dirt,
            controllers=build(args.controllers),
            seed=args.seed,
        ).run(minutes=args.minutes)
        m = result.metrics
        print(
            f"{count:>7}{m.team.coverage:>10.1%}{m.team.dirt_removed_fraction:>8.1%}"
            f"{m.speedup:>9.2f}{m.overlap:>9.1%}{m.balance:>9.2f}{m.encounters:>7}"
        )
    print("\n  speedup is team coverage over the best single member's, so its")
    print("  ceiling is the robot count. How far short it falls is the cost of")
    print("  sharing a pool.")
    return 0


def _compare(args) -> int:
    from zimablue.planners.compare import FLEET_ENTRIES, compare_fleets

    done = [0]

    def tick(trial):
        done[0] += 1
        print(
            f"  [{done[0]:>2}/{len(FLEET_ENTRIES)}] {trial.planner:<26} "
            f"team {trial.scores['coverage']:.1%} speedup {trial.scores['speedup']:.2f}",
            flush=True,
        )

    result = compare_fleets(
        robots=args.robots,
        pools=(args.pool,),
        seeds=(args.seed,),
        minutes=args.minutes,
        dirt=args.dirt,
        jobs=args.jobs,
        on_result=tick,
    )
    print(f"\n{result.table()}")
    if args.plots:
        from zimablue.planners.plots import plot_matrix, plot_paths

        args.plots.mkdir(parents=True, exist_ok=True)
        plot_matrix(result).savefig(args.plots / "fleet_matrix.png", dpi=110)
        plot_paths(result, columns=4).savefig(args.plots / "fleet_methods.png", dpi=110)
        print(f"figures in {args.plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
