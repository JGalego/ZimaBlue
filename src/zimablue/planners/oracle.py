"""An upper bound on the grams, so a score can become a regret.

``dirt_oracle`` is deliberately not this: it is the best *myopic* policy, and
its own docstring says it bounds nothing. What can be bounded is the physics.
In ``seconds`` of driving, a head ``swath`` wide moving at ``speed`` passes
over at most ``speed * seconds * swath`` of floor, and only mass that was under
the head was ever collectable. Grant a fictitious cleaner everything a real
one has to earn -- it teleports between the richest cells, never turns, never
revisits, lifts a cell's whole mass in one pass, and swallows every piece of
debris its intake can physically admit -- and the mass it ends with is the sum
of the heaviest cells that fit in that swept-area budget. Every real run
collects less, whatever the planner.

Two honesty notes. The bound is computed on the *initial* field, so dirt
drifting into the swept set during a run could in principle beat it; at the
package's drift rates the effect is well under the bound's own slack, and the
slack only ever points the safe way. And it caps at the collectable total --
oversize debris is outside any policy's reach and is not held against one.

The point of the bound is the column it buys. ``dirt`` says how much a
planner removed; ``of possible`` says how much of what was *reachable in the
time* it removed, which is the number that stays comparable when the pool,
the dirt or the duration changes. The distance from 100% is the regret, and
none of it is noise: it is travel, revisits, and not knowing where the dirt
is -- exactly the things a planner exists to manage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["collectable_bound", "dirt_bound"]


def collectable_bound(
    initial_grid: np.ndarray,
    navigable: np.ndarray,
    *,
    cell: float,
    speed: float,
    swath: float,
    seconds: float,
    collectable_debris: float = 0.0,
    collectable_total: float | None = None,
) -> float:
    """The most grams any policy could have collected, in grams.

    ``initial_grid`` is the dirt raster at t=0 (grams per cell), ``navigable``
    the boolean mask of floor the robot can reach. The relaxation ignores
    travel between cells and every revisit cost; see the module docstring for
    what that buys and what it forgives.
    """
    if seconds <= 0 or speed <= 0 or swath <= 0:
        return 0.0
    masses = np.sort(np.asarray(initial_grid, dtype=float)[np.asarray(navigable)].ravel())[::-1]
    budget_cells = int(speed * seconds * swath / (cell * cell))
    reachable = float(masses[:budget_cells].sum()) + float(collectable_debris)
    if collectable_total is not None:
        reachable = min(reachable, float(collectable_total))
    return reachable


def dirt_bound(recording: Any, pool: Any, robot: Any, *, seconds: float, cell: float) -> float:
    """The bound for one recorded run, from its own initial state."""
    initial = recording.dirt_at(0.0)
    if initial.size == 0:
        return 0.0
    navigable = pool.navigable_mask(cell)
    if navigable.shape != initial.shape:  # pragma: no cover - mismatched rasters
        return 0.0

    debris = recording.debris_at(0.0)
    limit = robot.cleaning.pump.max_debris_size
    collectable_debris = float(debris[debris[:, 3] <= limit, 2].sum()) if debris.size else 0.0
    total = float(initial.sum()) + float(debris[:, 2].sum() if debris.size else 0.0)
    oversize = float(debris[debris[:, 3] > limit, 2].sum()) if debris.size else 0.0

    return collectable_bound(
        initial,
        navigable,
        cell=cell,
        speed=robot.locomotion.max_speed,
        swath=robot.swath_width,
        seconds=seconds,
        collectable_debris=collectable_debris,
        collectable_total=total - oversize,
    )
