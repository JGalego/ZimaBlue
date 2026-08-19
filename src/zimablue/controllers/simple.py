"""Reference points for benchmarking, at opposite ends of the scale.

None of these is meant to be used in anger. They exist so that a number from
:class:`~zimablue.controllers.baseline.BaselineCoverage` -- or from your own
controller -- can be read against something: a floor and two ceilings.

The two ceilings are the point. :class:`LawnmowerOracle` is optimal at driving
and :class:`DirtOracle` is greedy about cleaning, and they do not agree about
which is the better run. A controller sitting below both is losing on
navigation; one that beats the lawnmower on dirt removed while scoring worse
on coverage is not broken, it is doing the job.
"""

from __future__ import annotations

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.geometry import wrap_angle
from zimablue.rng import RngTree
from zimablue.robot import Cleaner, DriveCommand

__all__ = ["DirtOracle", "LawnmowerOracle", "RandomBounce"]


class RandomBounce:
    """Drive straight; on contact, turn a random amount and continue.

    The floor. This is what the cheapest cleaners effectively do, and it is
    the number any real coverage strategy must beat. It is also a useful
    smoke test: it explores without ever needing a working heading estimate.
    """

    name = "random_bounce"

    def __init__(self, seed: int = 0, *, turn_range: tuple[float, float] = (0.6, 2.4)) -> None:
        self.seed = seed
        self.turn_range = turn_range
        self._rng = RngTree(seed).stream("controller:random_bounce")
        self._turning_until = 0.0
        self._turn_sign = 1.0

    def reset(self, robot: Cleaner) -> None:
        self._rng = RngTree(self.seed).stream("controller:random_bounce")
        self._turning_until = 0.0
        self._turn_sign = 1.0

    def step(self, ctl: ControlInput) -> DriveCommand:
        top = ctl.robot.locomotion.max_speed
        if ctl.battery <= ctl.robot.power.battery.cutoff:
            return DriveCommand.stop()

        contact = ctl.reading("contact")
        bumped = bool(contact is not None and contact.valid and np.any(contact.values > 0.5))
        stuck = ctl.extras.get("stuck", 0.0) > 0.5

        if (bumped or stuck) and ctl.time >= self._turning_until:
            self._turning_until = ctl.time + float(self._rng.uniform(*self.turn_range))
            self._turn_sign = 1.0 if self._rng.random() < 0.5 else -1.0

        if ctl.time < self._turning_until:
            turn = top * 0.6 * self._turn_sign
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)
        return DriveCommand(left=top * 0.85, right=top * 0.85, brush=True, pump=1.0)


class LawnmowerOracle:
    """A perfect boustrophedon path, driven from ground-truth pose.

    The ceiling, and **not a legitimate controller**: it requires
    ``Simulation(expose_truth=True)`` and would be impossible on hardware,
    where pose has to be estimated from drifting sensors.

    Its value is as an upper bound. If the oracle reaches 96% coverage on a
    pool, then a real controller scoring 78% has 18 points of *navigation* to
    win -- and if the oracle itself only reaches 80%, the gap is the pool
    shape or the robot's geometry, and no amount of planning will fix it.
    """

    name = "lawnmower_oracle"
    needs_truth = True

    def __init__(self, *, spacing_factor: float = 0.85, margin: float = 0.05) -> None:
        self.spacing_factor = spacing_factor
        self.margin = margin
        self._waypoints: list[tuple[float, float]] = []
        self._index = 0

    def reset(self, robot: Cleaner) -> None:
        self._waypoints = []
        self._index = 0

    def _plan(self, ctl: ControlInput) -> None:
        """Build the lane waypoints once, from the pool's true geometry."""
        from shapely.geometry import LineString, MultiLineString

        truth = ctl.truth
        pool = truth.pool  # attached by Simulation when expose_truth is on
        robot = ctl.robot
        spacing = robot.swath_width * self.spacing_factor
        inset = robot.radius + self.margin
        region = pool.navigable.buffer(-inset)
        if region.is_empty:
            region = pool.navigable

        minx, miny, maxx, maxy = region.bounds
        points: list[tuple[float, float]] = []
        for i, y in enumerate(np.arange(miny + spacing * 0.5, maxy, spacing)):
            cut = region.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
            if cut.is_empty:
                continue
            parts = list(cut.geoms) if isinstance(cut, MultiLineString) else [cut]
            # Visit every span on this row, not just the longest: that is the
            # difference between an oracle and the baseline it bounds.
            parts = sorted(
                (p for p in parts if p.length > spacing * 0.4),
                key=lambda p: p.coords[0][0],
                reverse=bool(i % 2),
            )
            for part in parts:
                (x0, y0), (x1, y1) = part.coords[0], part.coords[-1]
                if (x0 > x1) != bool(i % 2):
                    x0, x1, y0, y1 = x1, x0, y1, y0
                points += [(x0, y0), (x1, y1)]
        self._waypoints = points

    def step(self, ctl: ControlInput) -> DriveCommand:
        if ctl.truth is None:
            raise RuntimeError(
                "LawnmowerOracle needs ground truth; construct the run with "
                "Simulation(..., expose_truth=True). It is a benchmark bound, "
                "not a deployable controller."
            )
        if not self._waypoints:
            self._plan(ctl)
        top = ctl.robot.locomotion.max_speed
        if ctl.battery <= ctl.robot.power.battery.cutoff or self._index >= len(self._waypoints):
            return DriveCommand.stop()

        truth = ctl.truth
        target = self._waypoints[self._index]
        dx, dy = target[0] - truth.x, target[1] - truth.y
        distance = float(np.hypot(dx, dy))
        if distance < 0.12:
            self._index += 1
            return DriveCommand(left=top * 0.4, right=top * 0.4, brush=True, pump=1.0)

        error = float(wrap_angle(np.arctan2(dy, dx) - truth.heading))
        if abs(error) > np.deg2rad(35.0):
            turn = top * 0.5 * float(np.sign(error))
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)
        correction = float(np.clip(error * 1.1, -0.5, 0.5))
        speed = top * 0.9
        return DriveCommand(
            left=speed * (1.0 - correction),
            right=speed * (1.0 + correction),
            brush=True,
            pump=1.0,
        )


class DirtOracle:
    """Drive at whatever is dirtiest, reading the dirt field directly.

    The other reference point. :class:`LawnmowerOracle` is optimal at
    *covering* a pool and turns out to be a poor cleaner, because it visits
    every cell once and adhered dirt needs several passes. This one ignores
    coverage entirely and goes where the grams are.

    Not deployable either, and less so than the lawnmower: it needs
    ``Simulation(expose_truth=True)`` and reads the remaining mass per cell,
    which no cleaner can measure.

    **Greedy is not optimal.** It is worth being clear about this, because the
    name invites the other reading. On a kidney pool with autumn dirt it does
    remove more than the baseline at both horizons -- 37% against 16% in ten
    minutes, 59% against 39% in thirty -- but it gets there having driven over
    a quarter of the pool, against the baseline's four fifths. It works the
    richest patch until the easy mass is gone and the returns flatten, then
    crosses to the next one, and the floor it leaves is clean in blotches. So
    it is an upper bound on nothing: it is the best myopic policy, which makes
    it a good teacher for imitation early in a run and a bad one to copy for a
    long one.

    ``travel_cost`` is the exponent distance is discounted by, and it defaults
    to 0: go to the dirtiest cell in the pool, however far away it is. That
    reads as reckless and measures as best, on every pool tried -- adhered dirt
    needs repeated passes, so the grams are concentrated and worth crossing a
    pool for. Raising it to 0.5 saves about a third of the driving and costs
    one to five points of dirt removed.
    """

    name = "dirt_oracle"
    needs_truth = True

    def __init__(
        self,
        *,
        travel_cost: float = 0.0,
        replan_interval: float = 4.0,
        arrive: float = 0.15,
    ) -> None:
        self.travel_cost = travel_cost
        self.replan_interval = replan_interval
        self.arrive = arrive
        self._target: tuple[float, float] | None = None
        self._last_plan = -1e9

    def reset(self, robot: Cleaner) -> None:
        self._target = None
        self._last_plan = -1e9

    def _choose(self, ctl: ControlInput) -> tuple[float, float] | None:
        """The best cell to go to next, by grams discounted for the trip."""
        truth = ctl.truth
        world = truth.world  # attached by Simulation when expose_truth is on
        mass = world.dirt.field.total_grid()
        if not mass.any():
            return None

        grid = world.dirt.field.grid
        xs, ys = grid.cell_centers()
        # Only where the robot can actually get to, or it will drive at a
        # patch of wall dirt forever.
        navigable = truth.pool.navigable_mask(grid.cell)
        usable = navigable & (mass > 0.0)
        if not usable.any():
            return None

        distance = np.hypot(xs - truth.x, ys - truth.y)
        # Half a swath, so "the cell under the robot" is never the best target
        # and it cannot deadlock on the spot it is standing on.
        floor = max(0.5 * ctl.robot.swath_width, ctl.robot.radius)
        score = np.where(usable, mass / (distance + floor) ** self.travel_cost, -np.inf)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        return (float(xs[row, col]), float(ys[row, col]))

    def step(self, ctl: ControlInput) -> DriveCommand:
        if ctl.truth is None:
            raise RuntimeError(
                "DirtOracle needs ground truth; construct the run with "
                "Simulation(..., expose_truth=True). It reads the dirt field, "
                "so it is a benchmark bound rather than a deployable controller."
            )
        if ctl.battery <= ctl.robot.power.battery.cutoff:
            return DriveCommand.stop()

        truth = ctl.truth
        stale = ctl.time - self._last_plan >= self.replan_interval
        arrived = self._target is not None and (
            float(np.hypot(self._target[0] - truth.x, self._target[1] - truth.y)) < self.arrive
        )
        if self._target is None or stale or arrived:
            self._target = self._choose(ctl)
            self._last_plan = ctl.time
        if self._target is None:
            return DriveCommand.stop()

        top = ctl.robot.locomotion.max_speed
        dx, dy = self._target[0] - truth.x, self._target[1] - truth.y
        error = float(wrap_angle(np.arctan2(dy, dx) - truth.heading))
        if abs(error) > np.deg2rad(35.0):
            turn = top * 0.5 * float(np.sign(error))
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)
        correction = float(np.clip(error * 1.1, -0.5, 0.5))
        speed = top * 0.9
        return DriveCommand(
            left=speed * (1.0 - correction),
            right=speed * (1.0 + correction),
            brush=True,
            pump=1.0,
        )


@CONTROLLERS.register("random_bounce")
def _make_random(**kwargs: object) -> RandomBounce:
    return RandomBounce(**kwargs)  # type: ignore[arg-type]


@CONTROLLERS.register("dirt_oracle")
def _make_dirt_oracle(**kwargs: object) -> DirtOracle:
    return DirtOracle(**kwargs)  # type: ignore[arg-type]


@CONTROLLERS.register("lawnmower_oracle")
def _make_oracle(**kwargs: object) -> LawnmowerOracle:
    return LawnmowerOracle(**kwargs)  # type: ignore[arg-type]
