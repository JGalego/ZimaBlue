"""Coverage path planning: the interface, and the follower that executes it.

The coverage-path-planning literature splits cleanly in two, and the split is
not academic -- it decides what a method is allowed to know.

**Offline planners** are given the map and compute a whole path before the
robot moves. Cellular decompositions, sweep-direction optimisation,
contour-parallel offsets, spanning trees. They produce a :class:`CoveragePath`,
and a :class:`PathFollower` drives it.

**Online planners** are handed sensor readings and decide the next move. They
are :class:`~zimablue.controllers.base.Controller` implementations directly and
live in :mod:`zimablue.planners.online`.

An offline planner is not an oracle. Reading the pool's outline is not
cheating -- you could survey a pool once and load the result -- whereas reading
the dirt field, as ``dirt_oracle`` does, is information no machine can have.
But it is not free either, and :class:`PathFollower` is where the cost shows
up: a planned path has to be *executed*, which needs to know where the robot
is. Follow it on the true pose and you have measured the plan. Follow it on
the pose estimate and you have measured the plan plus the localisation, which
is the thing that actually ships.

That switch is one argument, and running the same planner both ways is the
most informative comparison in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from zimablue.registry import Registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.pool import Pool
    from zimablue.robot import Cleaner

__all__ = ["PLANNERS", "CoveragePath", "CoveragePlanner", "PathFollower", "make_planner"]

FloatArray = NDArray[np.float64]


@dataclass
class CoveragePath:
    """A planned route through a pool, plus how it was arrived at."""

    waypoints: FloatArray
    """``(n, 2)`` metres, in order. The robot is asked to pass through each."""

    planner: str = ""
    cells: tuple[Any, ...] = ()
    """The decomposition the plan came from, if it used one. Kept for drawing:
    the cells are usually more informative about a method than its path is."""

    notes: dict[str, Any] = field(default_factory=dict)
    """Anything the planner wants to report -- sweep angle, cell count, the
    tour length it settled for."""

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def length(self) -> float:
        """Total path length in metres, before any following error."""
        if len(self) < 2:
            return 0.0
        return float(np.hypot(*np.diff(self.waypoints, axis=0).T).sum())

    @property
    def turns(self) -> float:
        """Total absolute heading change in radians.

        The other half of the cost of a plan, and the half that coverage
        metrics ignore. Two paths of equal length are not equally good if one
        of them turns twice as often -- turning is slow, it slips, and on a
        tracked machine it is where the localisation error comes from.
        """
        if len(self) < 3:
            return 0.0
        deltas = np.diff(self.waypoints, axis=0)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        change = np.diff(headings)
        return float(np.abs(np.arctan2(np.sin(change), np.cos(change))).sum())

    @property
    def sharp_turns(self) -> int:
        """Waypoints where the path turns more than 45 degrees."""
        if len(self) < 3:
            return 0
        deltas = np.diff(self.waypoints, axis=0)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        change = np.diff(headings)
        return int((np.abs(np.arctan2(np.sin(change), np.cos(change))) > np.pi / 4).sum())

    def describe(self) -> str:
        return (
            f"{self.planner}: {len(self)} waypoints, {self.length:.0f} m, "
            f"{np.degrees(self.turns):.0f} deg of turning, {self.sharp_turns} sharp"
        )


@runtime_checkable
class CoveragePlanner(Protocol):
    """Plans a covering path over a known pool.

    One method. The planner is handed the geometry and the robot's dimensions
    -- the swath width is what sets the lane spacing, and no planner should be
    hard-coding it -- and returns an ordered list of waypoints.
    """

    name: str

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath: ...


PLANNERS: Registry[CoveragePlanner] = Registry("planner", entry_point_group="zimablue.planners")


def make_planner(planner: CoveragePlanner | str, **kwargs: Any) -> CoveragePlanner:
    """Resolve a planner from a name or pass one through."""
    if isinstance(planner, str):
        return PLANNERS.create(planner, **kwargs)
    return planner


# ----------------------------------------------------------------------
class PathFollower:
    """Drives a planned path. Pure pursuit, with the usual honesty switch.

    A :class:`~zimablue.controllers.base.Controller`, so a planned path is
    scored, recorded, batched and replayed exactly like a reactive controller.

    ``localisation`` decides where the robot thinks it is:

    ``"truth"``
        The simulator's true pose. This measures *the plan* -- how good the
        route is, given perfect execution. It needs ``expose_truth`` and is an
        upper bound, not a controller you could ship.
    ``"odometry"``
        Dead reckoning from the wheel encoders and gyro, through the same EKF
        the ``systematic`` controller uses. This measures the plan *plus* the
        localisation, which is what a real machine gets.

    The gap between the two is the point. A plan that looks excellent on truth
    and falls apart on odometry is telling you the bottleneck is not the
    planner, and that is a result rather than a disappointment.
    """

    name = "path_follower"

    def __init__(
        self,
        planner: CoveragePlanner | str,
        *,
        localisation: str = "odometry",
        lookahead: float = 0.45,
        arrive: float = 0.22,
        speed: float = 1.0,
        turn_gain: float = 2.2,
        stall: float = 12.0,
        estimator: Any = None,
        loop: bool = False,
    ) -> None:
        if localisation not in ("truth", "odometry"):
            raise ValueError(f"localisation must be 'truth' or 'odometry', got {localisation!r}")
        self.planner = make_planner(planner)
        self.localisation = localisation
        self.lookahead = float(lookahead)
        self.arrive = float(arrive)
        self.speed = float(speed)
        self.turn_gain = float(turn_gain)
        self.stall = float(stall)
        self.loop = loop
        self._estimator_config = estimator

        self.name = f"{getattr(self.planner, 'name', 'planned')}"
        # Reading the pool geometry is what an offline planner is *for*, and it
        # is a different thing from reading the dirt field. Both need the flag;
        # only one of them is undeployable.
        self.needs_truth = True

        self.index = 0
        self.blackboard: Any = None
        self.fleet_size = 1
        self.give_way = 1.1
        """Metres ahead at which a lower-numbered team-mate makes this one
        wait. Right of way by index is arbitrary and that is the point: it is
        a total order, so two robots meeting head-on cannot both defer and
        deadlock."""

        self.done = False
        """Whether the plan has been driven to its end."""

        self.partition: Any = None
        """The :class:`~zimablue.planners.partition.Partition` this follower's
        share was cut from, when one was. ``partitioned`` sets it so a finished
        fleet can be asked what the cut was, rather than only what happened."""

        self.path: CoveragePath | None = None
        self.target = 0
        self._pose = (0.0, 0.0, 0.0)
        self._estimator: Any = None
        self._last_time = 0.0
        self._planned_for: Any = None
        self._target_since = 0.0
        self._last_target = -1
        self._closest = np.inf
        self._skipped = 0
        self._waited = 0.0

    # ------------------------------------------------------------------
    def attach_fleet(
        self,
        *,
        index: int,
        blackboard: Any,
        origin: tuple[float, float, float],
        fleet_size: int = 1,
        share: bool = True,
    ) -> None:
        """Join a fleet: publish where we are, and give way to lower numbers.

        A follower has nothing to coordinate -- its route was decided before
        the run and a partitioner already made sure two robots are not sweeping
        the same floor. What it still has to do is not drive into anybody, and
        a plan cannot help with that because the other robot is not in it.
        """
        self.index = int(index)
        self.blackboard = blackboard
        self.fleet_size = int(fleet_size)

    def reset(self, robot: Cleaner) -> None:
        self.robot = robot
        self.done = False
        self.target = 0
        self.path = None
        self._planned_for = None
        self._last_time = 0.0
        self._estimator = None
        self._target_since = 0.0
        self._last_target = -1
        self._closest = np.inf
        self._skipped = 0
        self._waited = 0.0

    def telemetry(self) -> dict[str, float]:
        x, y, heading = self._pose
        return {
            "est_x": x,
            "est_y": y,
            "est_heading": heading,
            "waypoint": float(self.target),
            "waypoints": float(len(self.path) if self.path else 0),
            "progress": float(self.target / len(self.path))
            if self.path and len(self.path)
            else 0.0,
            "skipped": float(self._skipped),
            "waited": float(self._waited),
        }

    # ------------------------------------------------------------------
    def step(self, control_input: Any) -> Any:
        truth = control_input.truth
        if truth is None or getattr(truth, "pool", None) is None:
            raise RuntimeError(
                f"{self.name} plans against the pool's geometry, so it needs "
                "Simulation(expose_truth=True). Running it from a scenario or the CLI "
                "sets that automatically."
            )

        if self.path is None or self._planned_for is not truth.pool:
            self.path = self.planner.plan(truth.pool, control_input.robot)
            self._planned_for = truth.pool
            self.target = 0
            if len(self.path) == 0:
                raise RuntimeError(f"{self.planner.name} planned an empty path for this pool")

        self._pose = self._locate(control_input, truth)
        if self.blackboard is not None:
            x, y, heading = self._pose
            self.blackboard.publish(
                self.index,
                x,
                y,
                heading,
                time=control_input.time,
                extras={"waypoint": float(self.target)},
            )
            if self._must_wait():
                from zimablue.robot import DriveCommand

                # Waiting is not stalling. Without this the give-way rule and
                # the stall guard fight each other: a robot held at a
                # territory border for twelve seconds has its waypoint
                # confiscated by its own watchdog, and a fleet on a shared
                # border chops its plans to pieces that way.
                self._target_since = control_input.time
                self._waited += control_input.dt
                return DriveCommand(0.0, 0.0, brush=True, pump=1.0)
        return self._pursue(control_input)

    def _must_wait(self) -> bool:
        """Whether a lower-numbered robot is close enough and in front."""
        x, y, heading = self._pose
        for peer in self.blackboard.peers(self.index):
            if peer.index > self.index:
                continue
            gap = float(np.hypot(peer.x - x, peer.y - y))
            if gap > self.give_way:
                continue
            bearing = float(np.arctan2(peer.y - y, peer.x - x))
            ahead = abs(float(np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading))))
            if ahead < np.pi / 3:
                return True
        return False

    def _locate(self, control_input: Any, truth: Any) -> tuple[float, float, float]:
        """Where the follower believes the robot is."""
        if self.localisation == "truth":
            return (float(truth.x), float(truth.y), float(truth.heading))

        from zimablue.estimation import EstimatorConfig, PoseEstimator

        if self._estimator is None:
            config = self._estimator_config or EstimatorConfig()
            self._estimator = PoseEstimator(config, origin=(truth.x, truth.y, truth.heading))
            self._last_time = control_input.time

        dt = max(control_input.time - self._last_time, 0.0)
        self._last_time = control_input.time
        encoder = control_input.reading("encoder")
        imu = control_input.reading("imu")
        speed = 0.5 * (encoder[0] + encoder[1]) if encoder is not None and encoder.valid else 0.0
        gyro = imu[2] if imu is not None and imu.valid else 0.0
        if dt > 0:
            self._estimator.predict(float(speed), float(gyro), dt)
            self._estimator.zero_velocity_update(float(gyro), dt, moving=abs(speed) > 0.02)
        pose = self._estimator.estimate
        return (pose.x, pose.y, pose.heading)

    def _pursue(self, control_input: Any) -> Any:
        """Steer towards the first waypoint more than ``lookahead`` metres away.

        Pure pursuit on a polyline. Every waypoint inside the lookahead circle
        is consumed, and the robot aims at the first one outside it, so
        corners get cut slightly and the robot never has to stop, turn and go.

        The consumption rule is the whole of it, and getting it wrong is
        subtle. An earlier version advanced the waypoint index only on
        *arrival* -- within 18 cm -- while separately aiming a lookahead
        distance further along. On the open floor those agree. Against a wall
        they do not: the plan's first waypoint sits in a corner the hull
        cannot quite reach, the robot closes to 40 cm, starts aiming at the
        far end of the lane instead, reverses out, finds itself more than a
        lookahead from the corner again, and turns back. It paced a 15 cm
        stretch of tile for the entire run and covered 4% of the pool.
        """
        from zimablue.robot import DriveCommand

        assert self.path is not None
        x, y, heading = self._pose
        waypoints = self.path.waypoints
        here = np.array([x, y])

        while self.target < len(waypoints) - 1:
            waypoint = waypoints[self.target]
            if float(np.hypot(*(waypoint - here))) < self.lookahead:
                self.target += 1
                continue
            # Also drop a waypoint the robot has driven past. Cutting a corner
            # wide can leave one behind and outside the circle, and chasing it
            # backwards would undo the corner.
            reference = waypoints[self.target - 1] if self.target else None
            if reference is not None and float(np.dot(here - waypoint, waypoint - reference)) > 0:
                self.target += 1
                continue
            break

        # Give up on a waypoint nothing is making progress towards. Both
        # halves of the consumption rule above are geometric, and geometry
        # cannot see a robot wedged in a corner or orbiting a point the plan
        # put inside a wall -- ``spanning_tree`` drew a perfect one-metre
        # circle for the last four minutes of a run.
        #
        # *Progress*, not elapsed time. A boustrophedon lane is nine metres
        # long and takes forty seconds to drive, all of it on one waypoint,
        # so a plain timer skips three quarters of the plan while the robot
        # is doing exactly what it was told.
        gap = float(np.hypot(*(waypoints[self.target] - here)))
        if self.target != self._last_target:
            self._last_target = self.target
            self._closest = gap
            self._target_since = control_input.time
        elif gap < self._closest - 0.05:
            self._closest = gap
            self._target_since = control_input.time
        elif control_input.time - self._target_since > self.stall:
            self._target_since = control_input.time
            self._skipped += 1
            if self.target >= len(waypoints) - 1:
                # The plan is over and the last waypoint is out of reach.
                # Counting a skip every twelve seconds from here on would say
                # a 22-waypoint plan skipped 29 of them.
                self._skipped -= 1
                self.done = True
                return DriveCommand.stop()
            self.target += 1

        goal = waypoints[self.target]
        if self.target >= len(waypoints) - 1 and float(np.hypot(*(goal - here))) <= self.arrive:
            if not self.loop:
                # The plan is finished. Stopping is the honest thing to do, and
                # the ergodic metric will mark it down for exactly that -- see
                # docs/dynamics.md. Looping instead is one argument away.
                return DriveCommand.stop()
            self.target = 0
            goal = waypoints[0]

        bearing = float(np.arctan2(goal[1] - y, goal[0] - x))
        error = float(np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading)))

        limit = control_input.robot.locomotion.max_speed
        # Slow down for a big heading error rather than trying to drive and
        # turn at once; a differential drive turns fastest on the spot.
        forward = self.speed * limit * float(np.clip(np.cos(error), 0.0, 1.0))
        return DriveCommand.from_body(
            forward, self.turn_gain * error, control_input.robot.locomotion
        )
