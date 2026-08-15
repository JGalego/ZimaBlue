"""Things in a pool that are not just floor.

Features fall into two groups, and the split matters to the simulator:

* **Blocking** features (obstacles, un-climbable stairs) remove area from the
  navigable set and are collided against.
* **Hydraulic** features (drains, returns, skimmers) do not block the robot but
  perturb the water, which moves fine dirt around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

__all__ = [
    "Drain",
    "Obstacle",
    "PoolFeature",
    "Return",
    "Skimmer",
    "Stairs",
    "flow_field",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PoolFeature:
    """Base for anything placed in a pool."""

    name: str

    @property
    def blocking(self) -> bool:
        """Whether the robot is excluded from this feature's footprint."""
        return False

    @property
    def footprint(self) -> Polygon | None:
        """Planar area occupied, if any."""
        return None


@dataclass(frozen=True)
class Obstacle(PoolFeature):
    """A solid the robot must drive around (ladder foot, light niche, bench)."""

    polygon: Polygon = field(default_factory=Polygon)
    height: float = 0.5

    @property
    def blocking(self) -> bool:
        return True

    @property
    def footprint(self) -> Polygon:
        return self.polygon


@dataclass(frozen=True)
class Stairs(PoolFeature):
    """A stepped entry.

    Most floor-crawling cleaners cannot reliably climb pool steps, so stairs
    default to blocking.  Set ``climbable=True`` to model a robot that can, in
    which case the footprint stays navigable and cleanable.
    """

    polygon: Polygon = field(default_factory=Polygon)
    steps: int = 3
    top_depth: float = 0.2
    bottom_depth: float = 1.2
    climbable: bool = False

    @property
    def blocking(self) -> bool:
        return not self.climbable

    @property
    def footprint(self) -> Polygon:
        return self.polygon

    def step_depths(self) -> FloatArray:
        """Water depth at each tread, shallow to deep."""
        return np.linspace(self.top_depth, self.bottom_depth, self.steps)


@dataclass(frozen=True)
class Drain(PoolFeature):
    """A main drain: a sink that pulls water (and suspended fines) inward."""

    position: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.25
    flow_rate: float = 0.15
    """Nominal inflow speed at the drain rim, m/s."""


@dataclass(frozen=True)
class Return(PoolFeature):
    """A return jet: a source that pushes water along ``direction``."""

    position: tuple[float, float] = (0.0, 0.0)
    direction: tuple[float, float] = (1.0, 0.0)
    flow_rate: float = 0.4
    """Jet speed at the outlet, m/s."""

    reach: float = 3.0
    """Distance over which the jet decays to negligible, m."""


@dataclass(frozen=True)
class Skimmer(PoolFeature):
    """A surface skimmer: removes *floating* debris that drifts within reach."""

    position: tuple[float, float] = (0.0, 0.0)
    width: float = 0.35
    capture_radius: float = 0.6


def flow_field(
    features: tuple[PoolFeature, ...],
    xs: FloatArray,
    ys: FloatArray,
    circulation: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Steady-state water velocity from drains and returns.

    A deliberately crude superposition: each drain contributes an inward
    :math:`1/r`-ish pull and each return a directional jet decaying with
    distance.  This is not a flow solution -- it is a cheap, smooth,
    divergence-ignoring field whose only job is to nudge fine sediment toward
    drains and away from returns.  See ``docs/research.md`` (section 8) for why
    CFD is out of scope.
    """
    vx = np.zeros_like(xs, dtype=float)
    vy = np.zeros_like(ys, dtype=float)
    if circulation <= 0:
        return vx, vy

    for feature in features:
        if isinstance(feature, Drain):
            dx = feature.position[0] - xs
            dy = feature.position[1] - ys
            r = np.hypot(dx, dy)
            r = np.maximum(r, feature.radius)
            strength = circulation * feature.flow_rate * (feature.radius / r) ** 2
            vx += strength * dx / r
            vy += strength * dy / r
        elif isinstance(feature, Return):
            dx = xs - feature.position[0]
            dy = ys - feature.position[1]
            r = np.hypot(dx, dy)
            decay = np.exp(-r / max(feature.reach, 1e-6))
            norm = float(np.hypot(*feature.direction)) or 1.0
            vx += circulation * feature.flow_rate * decay * feature.direction[0] / norm
            vy += circulation * feature.flow_rate * decay * feature.direction[1] / norm
    return vx, vy
