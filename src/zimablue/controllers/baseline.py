"""The baseline coverage controller.

Deliberately simple, and deliberately *not good*.  Its job is to make ZimaBlue
demonstrate useful behaviour out of the box and to be an easy target for
anything better -- not to be a contribution.  It is a behaviour-based stack of
the kind that actually shipped in mid-range cleaners before gyro-guided
systematic navigation: drive a lane, detect the wall, turn, offset, repeat,
with a perimeter pass and a stuck-recovery reflex.

What it does *not* have, on purpose:

* no map, no SLAM, no cell decomposition -- so it strands area behind the
  concave side of a kidney or L-shaped pool, which the coverage metric will
  show you
* no absolute position -- heading comes from integrating the gyro, so it
  drifts, and the lanes drift with it
* no dirt sensing -- it cannot prioritise, so it cleans a clean pool exactly
  as hard as a filthy one

Every one of those is a good first exercise for a replacement controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.geometry import wrap_angle
from zimablue.robot import Cleaner, DriveCommand

__all__ = ["BaselineCoverage", "HeadingEstimator", "Phase"]


class Phase(Enum):
    """What the controller is currently trying to do."""

    PERIMETER = "perimeter"
    """Follow the wall once, to clean the edge band a lawnmower path misses."""

    LANE = "lane"
    """Drive straight along the current lane."""

    TURN = "turn"
    """Rotate onto the next lane heading."""

    SHIFT = "shift"
    """Advance one swath width between lanes."""

    RECOVER = "recover"
    """Back off and rotate after getting stuck."""

    DONE = "done"


class HeadingEstimator:
    """Integrates gyro rate into a heading estimate.

    This is the honest thing to do with a gyro and no compass: it works, and it
    drifts.  The drift is real -- the IMU carries a turn-on bias and a random
    walk -- so lanes gradually skew over a long run, exactly as they do on
    hardware.  A better controller would bound this with wall contacts or a
    depth prior; that is left as an exercise.
    """

    def __init__(self) -> None:
        self.heading = 0.0
        self._last_time: float | None = None

    def reset(self) -> None:
        self.heading = 0.0
        self._last_time = None

    def update(self, time: float, yaw_rate: float) -> float:
        if self._last_time is None:
            self._last_time = time
            return self.heading
        dt = time - self._last_time
        self._last_time = time
        if dt > 0:
            self.heading = float(wrap_angle(self.heading + yaw_rate * dt))
        return self.heading


@dataclass(frozen=True)
class _Sense:
    """This tick's readings, reduced to what the phase handlers ask about."""

    heading: float
    ahead: float
    left: float
    right: float
    front_contact: bool
    side_contact: bool
    blocked: bool
    moving: bool


@dataclass
class BaselineTuning:
    """Knobs, gathered in one place so they can be swept in experiments."""

    cruise_speed: float = 0.85
    """Fraction of the robot's top speed used on a lane."""

    turn_speed: float = 0.55
    wall_threshold: float = 0.30
    """Sonar range at which a wall counts as "ahead", m."""

    lane_overlap: float = 0.12
    """Fraction of the swath re-covered on the next lane. A little overlap
    beats the gaps that heading drift would otherwise leave."""

    perimeter_time: float = 0.25
    """Fraction of the run budget spent on the perimeter pass."""

    recover_time: float = 2.6
    stuck_backup_speed: float = 0.6

    wall_standoff: float = 0.34
    """Distance to hold from the wall during the perimeter pass, m."""

    wall_seek_range: float = 1.2
    """Beyond this the wall is considered out of reach and the robot goes
    looking for one instead of steering at a saturated error."""

    wedged_time: float = 2.5
    """Seconds of unbroken contact after which the robot assumes it is wedged
    and backs out, regardless of what the wheels report."""


class BaselineCoverage:
    """Boustrophedon coverage with wall following and stuck recovery."""

    name = "baseline_coverage"

    def __init__(self, tuning: BaselineTuning | None = None, *, run_duration: float = 1800.0):
        self.tuning = tuning or BaselineTuning()
        self.run_duration = run_duration
        self.heading = HeadingEstimator()
        self.phase = Phase.PERIMETER
        self._robot: Cleaner | None = None
        self._phase_started = 0.0
        self._target_heading = 0.0
        self._lane_direction = 1
        self._shift_distance = 0.0
        self._shift_travelled = 0.0
        self._odometer = 0.0
        self._last_time = 0.0
        self._recover_turn = 1.0
        self._after_turn = Phase.LANE
        self._contact_since: float | None = None

    # ------------------------------------------------------------------
    def reset(self, robot: Cleaner) -> None:
        self._robot = robot
        self.heading.reset()
        self.phase = Phase.PERIMETER
        self._phase_started = 0.0
        self._target_heading = 0.0
        self._lane_direction = 1
        self._shift_distance = robot.swath_width * (1.0 - self.tuning.lane_overlap)
        self._shift_travelled = 0.0
        self._odometer = 0.0
        self._last_time = 0.0
        self._recover_turn = 1.0
        self._after_turn = Phase.LANE
        self._contact_since = None

    # ------------------------------------------------------------------
    def step(self, ctl: ControlInput) -> DriveCommand:
        """One tick.

        Structured as a strict dispatch: read sensors, run exactly one phase
        handler, apply at most one transition.  An earlier version let phases
        fall through to each other within a tick, and near a wall that produced
        a TURN -> SHIFT -> TURN cascade in which the turn target was recomputed
        from the live heading every tick -- so the target chased the robot and
        it span on the spot indefinitely.  One transition per tick makes the
        machine's behaviour reviewable.
        """
        robot = ctl.robot
        top = robot.locomotion.max_speed
        sense = self._perceive(ctl)

        if ctl.battery <= robot.power.battery.cutoff:
            self.phase = Phase.DONE
            return DriveCommand.stop()

        # Watchdog. Two things the obvious checks miss:
        #
        # The simulator's stuck flag only fires when the robot achieves neither
        # translation nor rotation, so one grinding along a wall while spinning
        # freely never trips it. And odometry cannot detect it either -- the
        # encoders measure wheel speed, and a pinned robot's wheels keep
        # turning. That is true of real hardware too.
        #
        # Sustained contact is the signal that does work: if something has been
        # touching us for seconds, we are wedged, whatever the wheels say.
        if not (sense.front_contact or sense.side_contact):
            self._contact_since = None
        elif self._contact_since is None:
            self._contact_since = ctl.time
        wedged = (
            self._contact_since is not None
            and (ctl.time - self._contact_since) > self.tuning.wedged_time
        )
        if (ctl.extras.get("stuck", 0.0) > 0.5 or wedged) and self.phase is not Phase.RECOVER:
            self._recover_turn *= -1.0
            self._contact_since = None
            self._transition(Phase.RECOVER, ctl.time)

        elapsed = ctl.time - self._phase_started
        handler = self._HANDLERS[self.phase]
        return handler(self, ctl, sense, top, elapsed)

    # ------------------------------------------------------------------
    def _perceive(self, ctl: ControlInput) -> _Sense:
        """Fold this tick's readings into the few facts the phases need."""
        imu = ctl.reading("imu")
        yaw_rate = float(imu[2]) if imu is not None and imu.valid else 0.0
        heading = self.heading.update(ctl.time, yaw_rate)

        travelled = 0.0
        encoder = ctl.reading("encoder")
        if encoder is not None and encoder.valid:
            speed = 0.5 * (encoder[0] + encoder[1])
            dt = max(ctl.time - self._last_time, 0.0)
            travelled = abs(speed) * dt
            self._odometer += travelled
            self._shift_travelled += travelled
        self._last_time = ctl.time

        contact = ctl.reading("contact")
        if contact is not None and contact.valid:
            front = bool(contact[0] > 0.5)
            side = bool(contact[1] > 0.5 or contact[2] > 0.5)
        else:
            front = side = False

        sonar = ctl.reading("sonar")
        if sonar is not None and sonar.valid:
            ahead, left, right = float(sonar[0]), float(sonar[1]), float(sonar[2])
        else:
            ahead = left = right = float("inf")

        return _Sense(
            heading=heading,
            ahead=ahead,
            left=left,
            right=right,
            front_contact=front,
            side_contact=side,
            # Rear contact deliberately excluded: backing into a wall during a
            # recovery must not read as "wall ahead" and cancel the escape.
            blocked=front or ahead <= self.tuning.wall_threshold,
            moving=travelled > 1e-4,
        )

    # -- phase handlers ---------------------------------------------------
    def _do_perimeter(self, ctl, sense, top, elapsed):
        if ctl.time >= self.run_duration * self.tuning.perimeter_time:
            self._target_heading = sense.heading
            self._transition(Phase.LANE, ctl.time)
        return self._follow_wall(top, sense)

    def _do_lane(self, ctl, sense, top, elapsed):
        if sense.blocked:
            # End of the lane: first quarter of the U-turn, then shift across.
            self._begin_turn(ctl, sense, self._lane_direction * np.pi / 2, Phase.SHIFT)
            return self._rotate_toward(top, sense.heading)
        return self._drive_straight(top, sense.heading)

    def _do_turn(self, ctl, sense, top, elapsed):
        error = float(wrap_angle(self._target_heading - sense.heading))
        if abs(error) < np.deg2rad(8.0) or elapsed > 6.0:
            self._shift_travelled = 0.0
            self._transition(self._after_turn, ctl.time)
            return self._drive_straight(top, sense.heading)
        return self._rotate_toward(top, sense.heading)

    def _do_shift(self, ctl, sense, top, elapsed):
        if sense.blocked:
            # No room to offset: this is a corner, not a lane end. A quarter
            # turn here just faces the adjoining wall and the machine ping-pongs
            # between TURN and SHIFT forever, which is exactly what an earlier
            # version did. Turn right around, reverse the sweep, resume lanes.
            self._lane_direction *= -1
            self._begin_turn(ctl, sense, np.pi, Phase.LANE)
            return self._rotate_toward(top, sense.heading)
        if self._shift_travelled >= self._shift_distance or elapsed > 8.0:
            # Second quarter of the U-turn: back onto a lane, reversed.
            self._begin_turn(ctl, sense, self._lane_direction * np.pi / 2, Phase.LANE)
            return self._rotate_toward(top, sense.heading)
        return self._drive_straight(top, sense.heading)

    def _do_recover(self, ctl, sense, top, elapsed):
        if elapsed < self.tuning.recover_time * 0.4:
            back = -top * self.tuning.stuck_backup_speed
            return DriveCommand(left=back, right=back, brush=True, pump=1.0)
        if elapsed < self.tuning.recover_time:
            turn = top * self.tuning.turn_speed * self._recover_turn
            return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)
        self._target_heading = sense.heading
        self._transition(Phase.LANE, ctl.time)
        return self._drive_straight(top, sense.heading)

    def _do_done(self, ctl, sense, top, elapsed):
        return DriveCommand.stop()

    _HANDLERS = {
        Phase.PERIMETER: _do_perimeter,
        Phase.LANE: _do_lane,
        Phase.TURN: _do_turn,
        Phase.SHIFT: _do_shift,
        Phase.RECOVER: _do_recover,
        Phase.DONE: _do_done,
    }

    # ------------------------------------------------------------------
    def _transition(self, phase: Phase, time: float) -> None:
        self.phase = phase
        self._phase_started = time

    def _begin_turn(self, ctl, sense, delta: float, after: Phase) -> None:
        """Start a turn of ``delta`` radians, entering ``after`` when it lands.

        The follow-on phase is named explicitly rather than inferred from a turn
        counter: the counter version could be reset mid-sequence by the corner
        case and leave the machine cycling.
        """
        self._target_heading = float(wrap_angle(sense.heading + delta))
        self._after_turn = after
        self._transition(Phase.TURN, ctl.time)

    def _rotate_toward(self, top: float, heading: float) -> DriveCommand:
        """Spin in place toward ``_target_heading``."""
        error = float(wrap_angle(self._target_heading - heading))
        turn = top * self.tuning.turn_speed * float(np.sign(error) or 1.0)
        return DriveCommand(left=-turn, right=turn, brush=True, pump=1.0)

    def _drive_straight(self, top: float, heading: float) -> DriveCommand:
        """Hold the lane heading with a proportional correction."""
        error = float(wrap_angle(self._target_heading - heading))
        correction = float(np.clip(error * 0.8, -0.35, 0.35))
        speed = top * self.tuning.cruise_speed
        return DriveCommand(
            left=speed * (1.0 - correction),
            right=speed * (1.0 + correction),
            brush=True,
            pump=1.0,
        )

    def _follow_wall(self, top: float, sense: _Sense) -> DriveCommand:
        """Keep the wall on the left at roughly a fixed standoff.

        Three cases, and the middle one is the one that is easy to get wrong:

        * wall ahead -> rotate right, away from it
        * no wall within reach -> drive straight and go and find one. A pure
          proportional law here saturates on the max-range reading and drives
          the robot in circles in open water, which is precisely what an
          earlier version did.
        * wall alongside -> proportional standoff correction

        Steering sign: omega = (v_right - v_left) / track_width, so turning
        *toward* a wall on the left means making the right track the faster one.
        """
        speed = top * 0.7
        if sense.blocked:
            turn = top * self.tuning.turn_speed
            return DriveCommand(left=turn, right=-turn, brush=True, pump=1.0)

        if not np.isfinite(sense.left) or sense.left > self.tuning.wall_seek_range:
            return DriveCommand(left=speed, right=speed, brush=True, pump=1.0)

        error = float(np.clip((sense.left - self.tuning.wall_standoff) * 1.5, -0.45, 0.45))
        return DriveCommand(
            left=speed * (1.0 - error),
            right=speed * (1.0 + error),
            brush=True,
            pump=1.0,
        )


@CONTROLLERS.register("baseline_coverage")
def _make_baseline(**kwargs: object) -> BaselineCoverage:
    return BaselineCoverage(**kwargs)  # type: ignore[arg-type]
