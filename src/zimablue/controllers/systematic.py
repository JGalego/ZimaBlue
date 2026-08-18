"""A controller that builds a map and uses it.

The baseline is a reflex agent: it steers on bump switches and hardly reads its
sensors, which is why sensor faults barely change its behaviour and why random
bounce can out-cover it given enough time. This controller is the answer to
that -- and, being map-based, it is also the one that *degrades* when a sensor
lies, which is the point of having the fault machinery.

Parts:

* :class:`~zimablue.estimation.PoseEstimator` -- an EKF over encoders and the
  IMU, with zero-velocity updates to pin down the gyro bias.
* :class:`OccupancyMap` -- a grid in the *estimated* frame, filled in from wall
  contacts and sonar returns. The robot does not get the pool handed to it; it
  finds the walls by hitting them and by pinging them.
* :class:`SystematicCoverage` -- boustrophedon lanes while there is room, then
  a nearest-frontier search over the map when the local sweep runs out.

Everything is in the estimated frame, so estimation error and mapping error
compound exactly as they would on hardware. When the map is wrong, the plan is
wrong, and the coverage metric shows it.

A measured result worth knowing before you use this
---------------------------------------------------

Calibrating the odometry makes the *estimate* much better and the *coverage*
worse. On the kidney pool over 25 minutes, seed 42:

===============  ==============  ============  ========
encoder_scale    position error  filter sigma  coverage
===============  ==============  ============  ========
1.00 (default)         13.67 m        4.54 m      73.9%
0.96                   12.45 m        3.70 m      67.1%
0.94                    3.84 m        1.51 m      52.6%
0.92                    2.66 m        0.99 m      35.9%
===============  ==============  ============  ========

The estimator is not at fault -- 2.7 m after 340 m of travel with no absolute
reference is respectable dead reckoning. The planner is. With a poor estimate
the robot's lane plan is effectively randomised and it wanders widely, which
covers ground the way ``random_bounce`` does; with a good estimate it executes
disciplined short lanes and spends its time turning instead of sweeping
(182 m travelled against 340 m). Coverage here is being won by accident.

So this controller is an alternative to ``baseline_coverage``, not yet a
replacement for it, and the honest headline is that **better localisation does
not help until the planner can spend it.** That is the next roadmap item, and
it is now backed by a number rather than an intuition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.estimation import EstimatorConfig, PoseEstimator
from zimablue.geometry import wrap_angle
from zimablue.robot import Cleaner, DriveCommand

__all__ = ["MapCell", "OccupancyMap", "SystematicCoverage", "SystematicTuning"]


class MapCell:
    """Cell states. Plain ints: this grid is indexed in the hot loop."""

    UNKNOWN = 0
    FREE = 1
    WALL = 2


class OccupancyMap:
    """A coarse grid the robot fills in as it drives.

    Anchored at the robot's start, sized generously enough for a domestic pool
    and clipped at the edges -- a growable array would be tidier but this runs
    inside the control loop and a fixed allocation keeps it predictable.

    Resolution is deliberately coarser than the simulator's dirt raster. The
    map is only as good as the pose estimate feeding it, and recording it at a
    finer resolution than the estimate's own error would be false precision.
    """

    def __init__(self, cell: float = 0.25, extent: float = 30.0) -> None:
        self.cell = cell
        self.size = int(extent / cell)
        self.origin = self.size // 2
        self.grid = np.full((self.size, self.size), MapCell.UNKNOWN, dtype=np.uint8)
        self.covered = np.zeros((self.size, self.size), dtype=bool)

    # -- indexing ---------------------------------------------------------
    def to_index(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(np.clip(round(y / self.cell) + self.origin, 0, self.size - 1)),
            int(np.clip(round(x / self.cell) + self.origin, 0, self.size - 1)),
        )

    def to_world(self, row: int, col: int) -> tuple[float, float]:
        return ((col - self.origin) * self.cell, (row - self.origin) * self.cell)

    def state_at(self, x: float, y: float) -> int:
        row, col = self.to_index(x, y)
        return int(self.grid[row, col])

    # -- updating ---------------------------------------------------------
    def mark_free(self, x: float, y: float, radius: float) -> None:
        """The robot is here, so this is floor."""
        self._disk(x, y, radius, MapCell.FREE, overwrite_walls=False)

    def mark_covered(self, x: float, y: float, radius: float) -> None:
        rows, cols = self._disk_indices(x, y, radius)
        if rows.size:
            self.covered[rows, cols] = True

    def mark_wall(self, x: float, y: float) -> None:
        row, col = self.to_index(x, y)
        self.grid[row, col] = MapCell.WALL

    def observe_ray(self, x: float, y: float, angle: float, distance: float, hit: bool) -> None:
        """Carve free space along a sonar beam, and mark the endpoint if it hit.

        Standard inverse sensor model, minus the probabilities: at this
        resolution a binary map is enough for planning, and a log-odds grid
        would imply a confidence the pose estimate does not support.

        A non-finite distance carves nothing. An ultrasonic rangefinder that
        gets no echo back -- off the end of a pool, or into a surface angled
        away from it -- has not measured a long distance, it has failed to
        measure, and real drivers report that as NaN or inf. The simulated
        sensor never produces one, so this used to raise on the first beam
        with no return.
        """
        if not np.isfinite(distance) or distance <= 0.0:
            return
        steps = max(1, int(distance / self.cell))
        for i in range(steps):
            t = (i + 0.5) * self.cell
            self._set_free(x + np.cos(angle) * t, y + np.sin(angle) * t)
        if hit:
            self.mark_wall(x + np.cos(angle) * distance, y + np.sin(angle) * distance)

    def _set_free(self, x: float, y: float) -> None:
        row, col = self.to_index(x, y)
        if self.grid[row, col] == MapCell.UNKNOWN:
            self.grid[row, col] = MapCell.FREE

    def _disk_indices(self, x: float, y: float, radius: float):
        span = max(1, int(radius / self.cell))
        row0, col0 = self.to_index(x, y)
        rows, cols = np.mgrid[
            max(row0 - span, 0) : min(row0 + span + 1, self.size),
            max(col0 - span, 0) : min(col0 + span + 1, self.size),
        ]
        if rows.size == 0:
            return rows.ravel(), cols.ravel()
        wx = (cols - self.origin) * self.cell
        wy = (rows - self.origin) * self.cell
        inside = (wx - x) ** 2 + (wy - y) ** 2 <= radius * radius
        return rows[inside], cols[inside]

    def _disk(self, x: float, y: float, radius: float, value: int, *, overwrite_walls: bool):
        rows, cols = self._disk_indices(x, y, radius)
        if not rows.size:
            return
        if overwrite_walls:
            self.grid[rows, cols] = value
        else:
            mask = self.grid[rows, cols] != MapCell.WALL
            self.grid[rows[mask], cols[mask]] = value

    # -- planning ---------------------------------------------------------
    def nearest_frontier(self, x: float, y: float, *, min_distance: float = 0.0):
        """Breadth-first search for the nearest cell worth driving to.

        A cell qualifies if it is known floor and either **uncovered** or
        **adjacent to unknown space**. That second clause is the whole game.

        An earlier version demanded only "free and uncovered", and it produced
        a result that took a while to understand: improving the pose estimate
        made *coverage worse*. With a poor estimate the map smears, inventing
        free cells all over the place for the robot to chase; with a good one
        the known-free region is tight and quickly covered, so the search came
        up empty and the robot declared a mostly-unexplored pool finished.

        Targeting the free/unknown boundary is what frontier exploration
        actually means, and it makes a better estimate help rather than hurt.

        Returns world coordinates, or ``None`` when everything reachable really
        has been covered. BFS over non-wall cells, so the target is reachable
        rather than merely close.
        """
        start = self.to_index(x, y)
        seen = np.zeros_like(self.covered)
        queue = deque([start])
        seen[start] = True
        min_cells = min_distance / self.cell

        while queue:
            row, col = queue.popleft()
            far_enough = np.hypot(row - start[0], col - start[1]) >= min_cells
            wanted = not self.covered[row, col] or self._touches_unknown(row, col)
            if far_enough and wanted and self.grid[row, col] == MapCell.FREE:
                return self.to_world(row, col)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and not seen[nr, nc]:
                    seen[nr, nc] = True
                    if self.grid[nr, nc] != MapCell.WALL:
                        queue.append((nr, nc))
        return None

    def _touches_unknown(self, row: int, col: int) -> bool:
        """Whether this cell sits on the boundary of explored space."""
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            inside = 0 <= nr < self.size and 0 <= nc < self.size
            if inside and self.grid[nr, nc] == MapCell.UNKNOWN:
                return True
        return False

    def blocked_ahead(self, x: float, y: float, heading: float, distance: float) -> bool:
        """Whether a known wall lies within ``distance`` along ``heading``."""
        steps = max(1, int(distance / self.cell))
        for i in range(1, steps + 1):
            t = i * self.cell
            if self.state_at(x + np.cos(heading) * t, y + np.sin(heading) * t) == MapCell.WALL:
                return True
        return False

    @property
    def explored_cells(self) -> int:
        return int((self.grid != MapCell.UNKNOWN).sum())

    @property
    def covered_cells(self) -> int:
        return int((self.covered & (self.grid == MapCell.FREE)).sum())


class Phase(Enum):
    """The U-turn is spelled out as two distinct turns.

    Collapsing them into one TURN state and remembering "what comes next" in a
    separate field is how the earlier version of this machine ended up
    re-entering a turn with a target recomputed from the live heading, chasing
    itself. Two states cost one enum member and remove the whole class of bug.
    """

    LANE = "lane"
    TURN_OUT = "turn_out"
    """First quarter of the U-turn, off the lane."""

    SHIFT = "shift"
    """Advance one swath across to the next lane."""

    TURN_IN = "turn_in"
    """Second quarter, onto the new lane heading."""

    SEEK = "seek"
    """Drive to a frontier cell when the local sweep is exhausted."""

    RECOVER = "recover"
    DONE = "done"


@dataclass
class SystematicTuning:
    cruise_speed: float = 0.9
    turn_speed: float = 0.6
    wall_threshold: float = 0.32
    lane_overlap: float = 0.15
    recover_time: float = 2.4
    wedged_time: float = 2.2
    arrive_radius: float = 0.28
    """How close to a frontier target counts as arrived, m."""

    settle_time: float = 0.6
    """Seconds to hold still at each lane end. Pauses cost a little coverage
    and buy a zero-velocity update, which is the only thing that observes the
    gyro bias -- without them the lanes fan out over a long run."""

    zupt_interval: float = 0.0
    """Seconds between deliberate stop-and-check pauses. 0 disables them.

    The idea was sound and the measurement said no. In a curved pool the robot
    spends most of its time seeking and recovering, so lane-end pauses are
    rare -- a few hundred zero-velocity updates across a half-hour run against
    thirteen thousand in a rectangle -- and the gyro bias random-walks
    unobserved. Stopping on a timer fixes that, but the stops cost more
    coverage than the steadier heading wins back: turning this on at 20 s took
    the rectangle from 79% to 65% and the L-shape from 72% to 48%.

    Kept, defaulted off, because it is the right lever for a robot that pauses
    cheaply -- and because the negative result is worth recording rather than
    deleting."""

    zupt_pause: float = 0.5
    """How long each deliberate pause lasts."""


class SystematicCoverage:
    """Map-building boustrophedon coverage with frontier recovery."""

    name = "systematic"

    def __init__(
        self,
        tuning: SystematicTuning | None = None,
        *,
        map_cell: float = 0.25,
        estimator: EstimatorConfig | None = None,
        pause_for_zupt: bool = True,
    ) -> None:
        self.tuning = tuning or SystematicTuning()
        self.map_cell = map_cell
        # Uncalibrated by default, which is what an unknown robot on an unknown
        # surface actually gives you. See the class docstring for the measured
        # trade -- calibrating is a five-fold improvement in the estimate and
        # currently costs coverage, for reasons that are the planner's fault.
        self.estimator_config = estimator
        self.pause_for_zupt = pause_for_zupt
        self.estimator = PoseEstimator(self.estimator_config)
        self.map = OccupancyMap(cell=map_cell)
        self.phase = Phase.LANE

    # ------------------------------------------------------------------
    def reset(self, robot: Cleaner) -> None:
        self.estimator = PoseEstimator(self.estimator_config)
        self.map = OccupancyMap(cell=self.map_cell)
        self.phase = Phase.LANE
        self.swath = robot.swath_width
        self.radius = robot.radius
        self._lane_heading = 0.0
        self._lane_direction = 1
        self._target_heading = 0.0
        self._shift_from = (0.0, 0.0)
        self._phase_started = 0.0
        self._last_time = 0.0
        self._contact_since: float | None = None
        self._recover_turn = 1.0
        self._target: tuple[float, float] | None = None
        self._frontier_failures = 0
        self._empty_frontiers = 0
        self._last_replan = -1e9
        self._shift_heading = 0.0
        self._pause_until = -1e9
        self._last_zupt_at = 0.0

    # ------------------------------------------------------------------
    def step(self, ctl: ControlInput) -> DriveCommand:
        dt = max(ctl.time - self._last_time, 0.0)
        self._last_time = ctl.time
        top = ctl.robot.locomotion.max_speed

        speed, gyro, wheel_speed = self._read_proprioception(ctl)
        self.estimator.predict(speed, gyro, dt)
        moving = wheel_speed > self.estimator.config.zupt_speed
        self.estimator.zero_velocity_update(gyro, dt, moving=moving)
        pose = self.estimator.estimate

        self._update_map(ctl, pose)

        if ctl.battery <= ctl.robot.power.battery.cutoff:
            self.phase = Phase.DONE
            return DriveCommand.stop()

        # A deliberate stop-and-check, on a timer. Held above the state machine
        # so it cannot be starved by whatever phase the robot is in.
        if ctl.time >= self._pause_until > -1e8:
            self._pause_until = -1e9
            self._last_zupt_at = ctl.time
        if self._pause_until > -1e8:
            # Freeze the phase clock so a pause cannot time a phase out.
            self._phase_started += dt
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)
        if (
            self.pause_for_zupt
            and self.tuning.zupt_interval > 0.0
            and ctl.time - self._last_zupt_at > self.tuning.zupt_interval
            and self.phase in (Phase.LANE, Phase.SEEK)
        ):
            self._pause_until = ctl.time + self.tuning.zupt_pause
            self._phase_started += dt
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)

        blocked, wedged = self._obstruction(ctl, dt)
        if wedged and self.phase is not Phase.RECOVER:
            self._recover_turn *= -1.0
            self._contact_since = None
            self._enter(Phase.RECOVER, ctl.time)

        elapsed = ctl.time - self._phase_started
        handler = {
            Phase.LANE: self._do_lane,
            Phase.TURN_OUT: self._do_turn_out,
            Phase.SHIFT: self._do_shift,
            Phase.TURN_IN: self._do_turn_in,
            Phase.SEEK: self._do_seek,
            Phase.RECOVER: self._do_recover,
            Phase.DONE: lambda *_: DriveCommand.stop(),
        }[self.phase]
        return handler(ctl, pose, top, elapsed, blocked)

    # ------------------------------------------------------------------
    def _read_proprioception(self, ctl: ControlInput) -> tuple[float, float, float]:
        """Forward speed, gyro rate, and the fastest wheel.

        The third value is what decides whether the robot is standing still.
        Using the forward speed instead would call a robot spinning on the spot
        "stationary" -- its wheel speeds cancel -- and the zero-velocity update
        would then charge the whole rotation to the gyro bias.
        """
        encoder = ctl.reading("encoder")
        if encoder is not None and encoder.valid:
            left, right = float(encoder[0]), float(encoder[1])
        else:
            left = right = 0.0
        imu = ctl.reading("imu")
        gyro = float(imu[2]) if imu is not None and imu.valid else 0.0
        return 0.5 * (left + right), gyro, max(abs(left), abs(right))

    def _update_map(self, ctl: ControlInput, pose) -> None:
        self.map.mark_free(pose.x, pose.y, self.radius)
        self.map.mark_covered(pose.x, pose.y, 0.5 * self.swath)

        contact = ctl.reading("contact")
        if contact is not None and contact.valid:
            # A closed bump switch is a wall just beyond the hull, in the
            # direction of that switch.
            for index, offset in enumerate((0.0, np.pi / 2, -np.pi / 2, np.pi)):
                if contact[index] > 0.5:
                    angle = pose.heading + offset
                    reach = self.radius + self.map.cell
                    self.map.mark_wall(
                        pose.x + np.cos(angle) * reach, pose.y + np.sin(angle) * reach
                    )

        sonar = ctl.reading("sonar")
        if sonar is not None and sonar.valid:
            sensor = ctl.robot.sensors.get("sonar")
            angles = getattr(sensor, "beam_angles", ())
            max_range = getattr(sensor, "max_range", 3.0)
            for index, offset in enumerate(angles):
                distance = float(sonar[index])
                if not np.isfinite(distance):
                    continue
                # A max-range return means "nothing seen", not "wall there".
                self.map.observe_ray(
                    pose.x,
                    pose.y,
                    pose.heading + offset,
                    min(distance, max_range),
                    hit=distance < max_range - 1e-3,
                )

    def _obstruction(self, ctl: ControlInput, dt: float) -> tuple[bool, bool]:
        contact = ctl.reading("contact")
        front = side = False
        if contact is not None and contact.valid:
            front = bool(contact[0] > 0.5)
            side = bool(contact[1] > 0.5 or contact[2] > 0.5)

        sonar = ctl.reading("sonar")
        ahead = float(sonar[0]) if sonar is not None and sonar.valid else float("inf")
        if not np.isfinite(ahead):
            # No echo is not "the way is clear"; it is "no information". Treat
            # it as clear for steering -- the bump switches are the backstop --
            # but do not let a NaN propagate into the comparisons below.
            ahead = float("inf")
        blocked = front or ahead <= self.tuning.wall_threshold

        if not (front or side):
            self._contact_since = None
        elif self._contact_since is None:
            self._contact_since = ctl.time
        wedged = (
            self._contact_since is not None
            and (ctl.time - self._contact_since) > self.tuning.wedged_time
        ) or ctl.extras.get("stuck", 0.0) > 0.5
        return blocked, wedged

    # -- phases ------------------------------------------------------------
    def _do_lane(self, ctl, pose, top, elapsed, blocked):
        if blocked:
            self._begin_turn(ctl, pose, self._lane_direction * np.pi / 2, Phase.TURN_OUT)
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)
        return self._drive(top, pose, self._lane_heading)

    def _do_turn_out(self, ctl, pose, top, elapsed, blocked):
        # Hold still first: this is the zero-velocity window that keeps the
        # gyro bias observable, and it is the difference between parallel lanes
        # and a slowly widening fan.
        if self.pause_for_zupt and elapsed < self.tuning.settle_time:
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)
        if self._turn_complete(pose, elapsed):
            self._shift_from = (pose.x, pose.y)
            self._enter(Phase.SHIFT, ctl.time)
            return self._drive(top, pose, self._target_heading)
        return self._rotate(top, pose)

    def _do_shift(self, ctl, pose, top, elapsed, blocked):
        travelled = float(np.hypot(pose.x - self._shift_from[0], pose.y - self._shift_from[1]))
        target_shift = self.swath * (1.0 - self.tuning.lane_overlap)

        if blocked and travelled < target_shift * 0.5:
            # Cornered: no room to step across, so this sweep is finished.
            return self._start_seek(ctl, pose, top)
        if travelled >= target_shift or elapsed > 8.0:
            self._begin_turn(ctl, pose, self._lane_direction * np.pi / 2, Phase.TURN_IN)
            return self._rotate(top, pose)
        return self._drive(top, pose, self._shift_heading)

    def _do_turn_in(self, ctl, pose, top, elapsed, blocked):
        if self._turn_complete(pose, elapsed):
            self._lane_heading = self._target_heading
            # Alternate the U-turn direction so the lanes stack rather than
            # spiralling in one direction.
            self._lane_direction *= -1
            self._enter(Phase.LANE, ctl.time)
            return self._drive(top, pose, self._lane_heading)
        return self._rotate(top, pose)

    def _do_seek(self, ctl, pose, top, elapsed, blocked):
        """Drive to a frontier cell, then resume lane sweeping from there."""
        if self._target is None or elapsed > 45.0:
            return self._start_seek(ctl, pose, top)

        dx, dy = self._target[0] - pose.x, self._target[1] - pose.y
        if float(np.hypot(dx, dy)) < self.tuning.arrive_radius:
            self._frontier_failures = 0
            self._resume_lanes(ctl, pose)
            return self._drive(top, pose, self._lane_heading)

        if blocked:
            # The map was wrong or something is in the way. Turn away for a
            # moment, then re-plan -- re-planning every tick just burns the
            # give-up budget without moving.
            if ctl.time - self._last_replan > 2.0:
                self._last_replan = ctl.time
                self._frontier_failures += 1
                return self._start_seek(ctl, pose, top)
            return self._rotate_away(top)

        return self._drive_to(top, pose, self._target)

    def _do_recover(self, ctl, pose, top, elapsed, blocked):
        if elapsed < self.tuning.recover_time * 0.4:
            back = -top * 0.6
            return DriveCommand(back, back, brush=True, pump=1.0)
        if elapsed < self.tuning.recover_time:
            turn = top * self.tuning.turn_speed * self._recover_turn
            return DriveCommand(-turn, turn, brush=True, pump=1.0)
        self._resume_lanes(ctl, pose)
        return self._drive(top, pose, self._lane_heading)

    # -- helpers -----------------------------------------------------------
    def _start_seek(self, ctl, pose, top) -> DriveCommand:
        """Pick a frontier and head for it, or declare the pool finished."""
        target = self.map.nearest_frontier(pose.x, pose.y, min_distance=self.swath)
        if target is None:
            self._empty_frontiers += 1
            # Two consecutive empty searches, not one: a single miss can happen
            # while the robot is wedged against a wall with a stale map.
            if self._empty_frontiers >= 2:
                self._enter(Phase.DONE, ctl.time)
                return DriveCommand.stop()
            self._resume_lanes(ctl, pose)
            return self._drive(top, pose, self._lane_heading)
        if self._frontier_failures > 25:
            self._enter(Phase.DONE, ctl.time)
            return DriveCommand.stop()
        self._empty_frontiers = 0
        self._target = target
        self._enter(Phase.SEEK, ctl.time)
        return self._drive_to(top, pose, target)

    def _turn_complete(self, pose, elapsed: float) -> bool:
        error = float(wrap_angle(self._target_heading - pose.heading))
        return abs(error) < np.deg2rad(7.0) or elapsed > 8.0

    def _resume_lanes(self, ctl, pose) -> None:
        self._lane_heading = pose.heading
        self._target = None
        self._enter(Phase.LANE, ctl.time)

    def _begin_turn(self, ctl, pose, delta: float, phase: Phase) -> None:
        self._target_heading = float(wrap_angle(pose.heading + delta))
        self._shift_heading = self._target_heading
        self._enter(phase, ctl.time)

    def _enter(self, phase: Phase, time: float) -> None:
        self.phase = phase
        self._phase_started = time

    def _rotate(self, top: float, pose) -> DriveCommand:
        error = float(wrap_angle(self._target_heading - pose.heading))
        turn = top * self.tuning.turn_speed * float(np.sign(error) or 1.0)
        return DriveCommand(-turn, turn, brush=True, pump=1.0)

    def _rotate_away(self, top: float) -> DriveCommand:
        turn = top * self.tuning.turn_speed * self._recover_turn
        return DriveCommand(-turn, turn, brush=True, pump=1.0)

    def _drive(self, top: float, pose, heading: float) -> DriveCommand:
        error = float(wrap_angle(heading - pose.heading))
        if abs(error) > np.deg2rad(50.0):
            self._target_heading = heading
            return self._rotate(top, pose)
        correction = float(np.clip(error * 0.9, -0.4, 0.4))
        speed = top * self.tuning.cruise_speed
        return DriveCommand(
            speed * (1.0 - correction), speed * (1.0 + correction), brush=True, pump=1.0
        )

    def _drive_to(self, top: float, pose, target: tuple[float, float]) -> DriveCommand:
        heading = float(np.arctan2(target[1] - pose.y, target[0] - pose.x))
        return self._drive(top, pose, heading)

    # ------------------------------------------------------------------
    def telemetry(self) -> dict[str, float]:
        """Per-tick channels recorded alongside the simulation's own.

        Recording the estimate next to ground truth is what makes estimation
        error visible in replay instead of merely asserted in a test.
        """
        pose = self.estimator.estimate
        return {
            "est_x": pose.x,
            "est_y": pose.y,
            "est_heading": pose.heading,
            "est_bias": pose.gyro_bias,
            "est_sigma": pose.position_sigma,
            "est_heading_sigma": pose.heading_sigma,
            "zupts": float(pose.zupt_count),
            "phase": float(list(Phase).index(self.phase)),
            "mapped": float(self.map.explored_cells),
            "map_covered": float(self.map.covered_cells),
        }


@CONTROLLERS.register("systematic")
def _make_systematic(**kwargs: object) -> SystematicCoverage:
    return SystematicCoverage(**kwargs)  # type: ignore[arg-type]
