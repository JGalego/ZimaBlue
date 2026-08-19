#!/usr/bin/env python3
"""Analyse how a controller behaves, not just what it achieved.

    python examples/analyse_dynamics.py                  # the whole report
    python examples/analyse_dynamics.py --pool mushroom  # the trap
    python examples/analyse_dynamics.py --plots out/     # write the figures

Coverage and dirt removed say what a run achieved. This asks different
questions: does the robot repeat itself, how fast does it forget where it
started, is it serving the distribution you meant, and how far ahead is a
prediction about it worth anything.

Two of the answers are not what you would guess, and both are printed below
rather than hidden in a docstring: the "random" controller is the *least*
sensitive to its initial conditions, and both ground-truth oracles score their
best distribution halfway through and then make it worse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import zimablue as zb
from zimablue.dynamics import (
    divergence,
    ergodic_score,
    forecast_cleaning,
    return_map,
    transfer_operator,
)

CONTROLLERS = ("baseline_coverage", "random_bounce", "systematic")


def run(pool: str, controller: str, seed: int, minutes: float):
    return (
        zb.Simulation(
            pool=pool,
            dirt="autumn",
            controller=controller,
            seed=seed,
            expose_truth=controller.endswith("oracle"),
        )
        .run(minutes=minutes)
        .recording
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="kidney")
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--plots", type=Path, default=None, help="write the figures here")
    args = parser.parse_args(argv)

    seeds = list(range(1, args.seeds + 1))
    runs = {c: [run(args.pool, c, s, args.minutes) for s in seeds] for c in CONTROLLERS}

    # -- 1. periodic orbits ------------------------------------------------
    print(f"\n{'':<20}{'contacts':>10}{'per min':>9}{'orbits':>8}{'trapped':>9}")
    print("-" * 56)
    for name, recordings in runs.items():
        section = return_map(recordings[0])
        orbits = section.periodic_orbits()
        print(
            f"{name:<20}{len(section):>10}{section.rate:>9.1f}{len(orbits):>8}"
            f"{section.trapped_fraction():>8.0%}"
        )
    print("  a trapped fraction near zero is the finding, not a null result:")
    print("  with this noise model the robot does not settle onto a closed loop.")

    # -- 2. mixing and almost-invariant sets -------------------------------
    print(f"\n{'':<20}{'cells':>7}{'gap':>8}{'mixing':>10}{'reliable':>10}   regions")
    print("-" * 78)
    for name, recordings in runs.items():
        operator = transfer_operator(recordings, cell=0.75, lag=10.0)
        labels = operator.almost_invariant_sets(2)
        leak = operator.leak_rate(labels)
        mixing = operator.mixing_time
        sizes = "/".join(str(n) for n in np.bincount(labels))
        print(
            f"{name:<20}{len(operator):>7}{operator.spectral_gap:>8.3f}"
            f"{('never' if not np.isfinite(mixing) else f'{mixing:.0f} s'):>10}"
            f"{operator.reliable!s:>10}   "
            f"{sizes} cells, leak " + "/".join(f"{v:.0%}" for _, v in sorted(leak.items()))
        )

    # -- 3. the ergodic metric ---------------------------------------------
    print(f"\n{'':<20}{'final':>9}{'best':>9}{'at':>8}{'wasted':>9}")
    print("-" * 55)
    scores = {}
    for name, recordings in runs.items():
        score = ergodic_score(recordings[0], target="dirt")
        scores[name] = score
        print(
            f"{name:<20}{score.value:>9.4f}{score.best:>9.4f}"
            f"{score.time_of_best:>7.0f}s{score.wasted:>9.0%}"
        )
    print("  'wasted' is the share of the run after the score stopped improving.")
    print("  Coverage cannot report it: coverage only ever goes up.")

    # -- 4. sensitivity ----------------------------------------------------
    print(f"\n{'':<20}{'lambda':>10}{'horizon':>10}{'diverged':>10}")
    print("-" * 51)
    for name in CONTROLLERS:
        result = divergence(
            controller=name, pool=args.pool, minutes=min(args.minutes, 10.0), runs=4, seed=3
        )
        horizon = result.horizon()
        print(
            f"{name:<20}{result.exponent():>10.4f}"
            f"{('inf' if not np.isfinite(horizon) else f'{horizon:.0f} s'):>10}"
            f"{result.diverged:>10.0%}"
        )
    print("  random_bounce is the least sensitive, not the most: its turn angles")
    print("  come from a seeded generator rather than from where the robot is.")

    # -- 5. forecasting ----------------------------------------------------
    print(f"\n{'':<20}{'rate':>10}{'fit window':>12}{'forecast err':>14}")
    print("-" * 57)
    forecasts = {}
    for name, recordings in runs.items():
        forecast = forecast_cleaning(recordings[0], fit_fraction=0.25)
        forecasts[name] = forecast
        print(
            f"{name:<20}{forecast.rate:>10.4f}{forecast.fitted_from / 60:>10.0f} min"
            f"{forecast.forecast_error:>13.1%}"
        )
    print("  a large error means the controller changed strategy partway through,")
    print("  which is what makes this a detector rather than only a predictor.")

    if args.plots is not None:
        from zimablue.dynamics.plots import (
            plot_ergodic,
            plot_forecast,
            plot_return_map,
            plot_transfer,
        )

        args.plots.mkdir(parents=True, exist_ok=True)
        plot_return_map(return_map(runs["random_bounce"][0])).savefig(
            args.plots / "section.png", dpi=110
        )
        plot_transfer(transfer_operator(runs["baseline_coverage"], cell=0.75)).savefig(
            args.plots / "transfer.png", dpi=110
        )
        plot_ergodic(scores).savefig(args.plots / "ergodic.png", dpi=110)
        plot_forecast(forecasts).savefig(args.plots / "forecast.png", dpi=110)
        print(f"\nfigures in {args.plots}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
