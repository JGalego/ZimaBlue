"""Metrics -- the two families, kept apart.

IEC 62929 evaluates a cleaning robot with a *coverage* test and a separate
*dust removal* test, and the research literature adds overlap rate, energy and
uniformity (``docs/research.md`` section 9).  ZimaBlue follows that split
literally, because collapsing it is how a testbed ends up rewarding a robot
that drives beautifully and cleans nothing.

* **Geometric** -- where the robot went: ``coverage``, ``floor_coverage``,
  ``wall_coverage``, ``revisits``, ``distance_traveled``.
* **Cleaning quality** -- what it removed: ``dirt_removed``,
  ``remaining_dirt``, per-type breakdown, ``cleaning_uniformity``.

Every scalar has a spatial companion (the visit grid, the remaining-dirt grid)
so replay can *show* the difference rather than assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.backends.base import Event, SimState
from zimablue.world import World

__all__ = ["Metrics", "SpatialMetrics", "compute_metrics"]

FloatArray = NDArray[np.float64]


@dataclass
class SpatialMetrics:
    """Per-cell arrays behind the scalars, for plotting and replay."""

    visits: NDArray[np.int32]
    """How many ticks the cleaning head covered each cell."""

    remaining_dirt: FloatArray
    """Grams per cell still on the floor at the end."""

    initial_dirt: FloatArray
    wall_visits: NDArray[np.int32]
    """Visit counts per 10 cm bin around the unrolled pool perimeter."""

    navigable: NDArray[np.bool_]

    @property
    def missed(self) -> NDArray[np.bool_]:
        """Navigable cells the robot never reached."""
        return self.navigable & (self.visits == 0)

    @property
    def dirt_removed_grid(self) -> FloatArray:
        return np.maximum(self.initial_dirt - self.remaining_dirt, 0.0)


@dataclass
class Metrics:
    """Scalar results of one run."""

    # Geometric
    coverage: float = 0.0
    """Fraction of navigable floor the cleaning head passed over, 0-1."""

    floor_coverage: float = 0.0
    """Alias of ``coverage``, named to contrast with ``wall_coverage``."""

    wall_coverage: float = 0.0
    """Fraction of the pool perimeter the robot ran alongside, 0-1."""

    revisits: float = 0.0
    """Mean extra passes over already-covered cells. 0 means a perfect
    non-overlapping path; 2 means each covered cell was hit three times."""

    distance_traveled: float = 0.0
    runtime: float = 0.0

    # Cleaning quality
    dirt_removed: float = 0.0
    """Grams permanently removed from the pool."""

    dirt_removed_fraction: float = 0.0
    remaining_dirt: float = 0.0
    dirt_collected: float = 0.0
    """Grams in the filter. Below ``dirt_removed`` only if debris was double
    counted; equal in the normal case."""

    dirt_by_type: dict[str, float] = field(default_factory=dict)
    """Remaining grams per dirt type -- shows *what* was left behind."""

    removed_by_type: dict[str, float] = field(default_factory=dict)
    cleaning_uniformity: float = 0.0
    """1 - (std/mean) of remaining dirt over navigable cells, clipped to 0-1.
    High means evenly clean; low means patchy."""

    debris_remaining: int = 0
    debris_collected: int = 0

    # Cost and failure
    energy_consumed: float = 0.0
    """Watt-hours."""

    battery_remaining: float = 0.0
    stuck_time: float = 0.0
    stuck_events: int = 0
    collisions: int = 0
    filter_full: bool = False
    completed: bool = False
    """Whether the run ended by finishing rather than by battery or timeout."""

    termination: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "floor_coverage": self.floor_coverage,
            "wall_coverage": self.wall_coverage,
            "revisits": self.revisits,
            "distance_traveled": self.distance_traveled,
            "runtime": self.runtime,
            "dirt_removed": self.dirt_removed,
            "dirt_removed_fraction": self.dirt_removed_fraction,
            "remaining_dirt": self.remaining_dirt,
            "dirt_collected": self.dirt_collected,
            "dirt_by_type": self.dirt_by_type,
            "removed_by_type": self.removed_by_type,
            "cleaning_uniformity": self.cleaning_uniformity,
            "debris_remaining": self.debris_remaining,
            "debris_collected": self.debris_collected,
            "energy_consumed": self.energy_consumed,
            "battery_remaining": self.battery_remaining,
            "stuck_time": self.stuck_time,
            "stuck_events": self.stuck_events,
            "collisions": self.collisions,
            "filter_full": self.filter_full,
            "completed": self.completed,
            "termination": self.termination,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metrics:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary(self) -> str:
        """A compact human-readable block, used by the CLI."""
        return "\n".join(
            [
                f"  coverage          {self.coverage * 100:6.1f} %   "
                f"(walls {self.wall_coverage * 100:.0f} %)",
                f"  dirt removed      {self.dirt_removed_fraction * 100:6.1f} %   "
                f"({self.dirt_removed:.0f} g of "
                f"{self.dirt_removed + self.remaining_dirt:.0f} g)",
                f"  uniformity        {self.cleaning_uniformity * 100:6.1f} %",
                f"  revisits          {self.revisits:6.2f}   extra passes/cell",
                f"  distance          {self.distance_traveled:6.1f} m",
                f"  runtime           {self.runtime / 60:6.1f} min",
                f"  energy            {self.energy_consumed:6.1f} Wh   "
                f"(battery {self.battery_remaining * 100:.0f} % left)",
                f"  collisions        {self.collisions:6d}",
                f"  stuck             {self.stuck_events:6d} events, {self.stuck_time:.1f} s",
                f"  termination       {self.termination}",
            ]
        )


def compute_metrics(
    world: World,
    state: SimState,
    events: list[Event],
    visits: NDArray[np.int32],
    wall_visits: NDArray[np.int32],
    initial_dirt: FloatArray,
    *,
    termination: str = "unknown",
) -> tuple[Metrics, SpatialMetrics]:
    """Score a finished run."""
    pool = world.pool
    navigable = pool.navigable_mask(world.cell)
    navigable_cells = int(navigable.sum())

    covered = navigable & (visits > 0)
    covered_cells = int(covered.sum())
    coverage = covered_cells / navigable_cells if navigable_cells else 0.0

    # Revisits: extra passes beyond the first, averaged over covered cells.
    # Measured only where the robot actually went -- averaging over the whole
    # pool would let poor coverage disguise itself as an efficient path.
    if covered_cells:
        extra = visits[covered].astype(float) - 1.0
        revisits = float(extra.mean())
    else:
        revisits = 0.0

    wall_coverage = float((wall_visits > 0).sum() / len(wall_visits)) if len(wall_visits) else 0.0

    remaining_grid = world.dirt.field.total_grid()
    remaining_total = world.dirt.total_mass
    initial_total = world.dirt.initial_mass

    removed = max(0.0, initial_total - remaining_total)
    removed_fraction = removed / initial_total if initial_total > 0 else 1.0

    initial_by_type = world.dirt.field.initial_by_type
    remaining_by_type = world.dirt.field.by_type()
    removed_by_type = {
        name: max(0.0, initial_by_type.get(name, 0.0) - remaining_by_type.get(name, 0.0))
        for name in initial_by_type
    }

    # Uniformity: how *evenly* the robot cleaned, not how evenly dirty the pool
    # ended up. Measuring the spread of remaining dirt would mostly measure how
    # patchy the dirt was to begin with -- a perfect cleaner working on a patchy
    # pool would score badly through no fault of its own. So it is the spread of
    # the per-cell *removal fraction*, over cells that had dirt worth removing.
    #
    # 1 means every dirty cell was cleaned to the same degree; 0 means the robot
    # stripped its lanes and left everything between them untouched.
    had_dirt = navigable & (initial_dirt > 1e-6)
    if had_dirt.any():
        fractions = np.clip(
            (initial_dirt[had_dirt] - remaining_grid[had_dirt]) / initial_dirt[had_dirt], 0.0, 1.0
        )
        mean = float(fractions.mean())
        uniformity = float(np.clip(1.0 - fractions.std() / mean, 0.0, 1.0)) if mean > 1e-9 else 0.0
    else:
        uniformity = 1.0

    stuck_events = sum(1 for e in events if e.kind == "stuck")
    collisions = sum(1 for e in events if e.kind == "collision")
    filter_full = any(e.kind == "filter_full" for e in events)

    metrics = Metrics(
        coverage=coverage,
        floor_coverage=coverage,
        wall_coverage=wall_coverage,
        revisits=revisits,
        distance_traveled=state.distance,
        runtime=state.time,
        dirt_removed=removed,
        dirt_removed_fraction=removed_fraction,
        remaining_dirt=remaining_total,
        dirt_collected=state.dirt_collected,
        dirt_by_type=remaining_by_type,
        removed_by_type=removed_by_type,
        cleaning_uniformity=uniformity,
        debris_remaining=len(world.dirt.debris) - world.dirt.debris.collected_count,
        debris_collected=world.dirt.debris.collected_count,
        energy_consumed=state.energy_used_wh,
        battery_remaining=state.battery_fraction,
        stuck_time=state.stuck_time,
        stuck_events=stuck_events,
        collisions=collisions,
        filter_full=filter_full,
        completed=termination in ("duration", "target_reached"),
        termination=termination,
    )
    spatial = SpatialMetrics(
        visits=visits.copy(),
        remaining_dirt=remaining_grid,
        initial_dirt=initial_dirt,
        wall_visits=wall_visits.copy(),
        navigable=navigable,
    )
    return metrics, spatial
