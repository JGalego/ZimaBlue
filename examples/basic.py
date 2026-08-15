#!/usr/bin/env python3
"""The smallest useful ZimaBlue program.

    python examples/basic.py

Builds a dirty kidney pool, runs the baseline cleaner for 20 simulated
minutes, prints both families of metrics, and saves a replayable recording.
"""

from __future__ import annotations

from pathlib import Path

import zimablue as zb


def main() -> None:
    sim = zb.Simulation(
        pool="kidney",
        robot="tracked",
        dirt="autumn",
        seed=42,
    )

    print(f"pool  {sim.pool}")
    print(f"robot {sim.robot.describe()}")
    print(f"dirt  {sim.world.dirt.initial_mass:.0f} g to remove\n")

    result = sim.run(minutes=20)
    print(result.metrics.summary())

    # Coverage and cleanliness are different questions. This is the whole point.
    print(
        f"\nThe robot drove over {result.metrics.coverage:.0%} of the floor "
        f"and removed {result.metrics.dirt_removed_fraction:.0%} of the dirt."
    )
    for name, remaining in sorted(result.metrics.dirt_by_type.items()):
        initial = result.world.dirt.field.initial_by_type.get(name, 0.0)
        if initial > 0:
            print(f"  {name:10s} {1 - remaining / initial:5.0%} removed")

    out = Path("runs/basic.zbr")
    result.save(out)
    print(f"\nsaved {out}\nwatch it:  zimablue replay {out}")


if __name__ == "__main__":
    main()
