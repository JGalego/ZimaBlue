"""Turning a :class:`DriveCommand` into something a motor driver accepts.

A controller asks for track speeds in metres per second.  A motor driver takes
a duty cycle.  In simulation the gap between those two does not exist: the
backend reads the commanded speed, applies a slip model and integrates, so the
requested speed is achieved by construction.  On hardware, closing that gap is
a control loop, and it is the first piece of the port that can be wrong in a
way the simulator cannot show you.

:class:`WheelSpeedLoop` is a per-side PI controller with velocity feedforward.
It is not clever and does not need to be -- a geared DC drivetrain under a
constant load is close enough to first order that PI is the standard answer.
What matters is the parts that are easy to leave out and painful to debug on a
robot: feedforward so the integrator is not doing all the work, anti-windup so
a jammed track does not store up a lurch, a slew limit so a step command does
not brown out the supply, and per-side symmetry so a saturating turn does not
change its own radius.

The loop is testable without hardware, because the simulator will happily act
as the plant::

    loop = WheelSpeedLoop.for_robot(robot)
    effort = loop(command, measured=(enc.left, enc.right), dt=dt)

:meth:`for_robot` derives the limits from the robot's own specification, which
means the gains are the only numbers left to tune on the real machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zimablue.robot import Cleaner, DriveCommand

__all__ = ["MotorEffort", "WheelSpeedLoop"]


@dataclass(frozen=True)
class MotorEffort:
    """What to send to the driver: normalised duty per side, plus the rest.

    ``left`` and ``right`` are in ``[-1, 1]`` and mean "fraction of full
    forward effort". They are deliberately not volts, amps or PWM counts: the
    mapping to those is the driver's business and the one part of this that
    cannot be written without the hardware in front of you.
    """

    left: float
    right: float
    brush: bool = True
    pump: float = 1.0
    saturated: bool = False
    """True when either side hit its limit this tick.

    Worth recording. A loop that spends its time saturated is not controlling
    anything, and the symptom on a robot -- a turn that comes out wider than
    commanded -- looks like a planner bug.
    """

    def as_tuple(self) -> tuple[float, float]:
        return (self.left, self.right)


@dataclass
class WheelSpeedLoop:
    """Per-side PI speed control with feedforward.

    ``kp`` and ``ki`` are in effort per (m/s) and effort per (m/s * s). The
    defaults are a starting point for a small geared drive and are meant to be
    replaced by numbers measured on the real drivetrain; see
    :doc:`../docs/hardware` for the step-response procedure.
    """

    max_speed: float
    """Track speed at full effort, m/s. Sets the feedforward scale."""

    kp: float = 1.5
    ki: float = 4.0
    max_effort: float = 1.0
    slew: float = 8.0
    """Maximum change in effort per second. 0 disables the limit."""

    integral_limit: float = 0.5
    """Cap on the integral term's contribution, in effort units."""

    deadband: float = 0.01
    """Commanded speeds below this are treated as a stop, m/s.

    Without it the integrator winds up against static friction whenever the
    robot is asked to hold still, and the first command after that arrives as
    a kick.
    """

    def __post_init__(self) -> None:
        if self.max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {self.max_speed}")
        if self.max_effort <= 0:
            raise ValueError(f"max_effort must be positive, got {self.max_effort}")
        self.reset()

    @classmethod
    def for_robot(cls, robot: Cleaner, **kwargs: float) -> WheelSpeedLoop:
        """Build a loop whose limits come from the robot's own specification."""
        return cls(max_speed=robot.locomotion.max_speed, **kwargs)

    def reset(self) -> None:
        self._integral = np.zeros(2)
        self._effort = np.zeros(2)

    @property
    def integral(self) -> tuple[float, float]:
        """The accumulated error per side, for telemetry and for debugging.

        A steadily growing integral on one side is a dragging track or a
        miscalibrated encoder, and it is visible here long before it is
        visible in the trajectory.
        """
        return (float(self._integral[0]), float(self._integral[1]))

    def __call__(
        self,
        command: DriveCommand,
        measured: tuple[float, float],
        dt: float,
    ) -> MotorEffort:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        target = np.array([command.left, command.right], dtype=float)
        target[np.abs(target) < self.deadband] = 0.0
        actual = np.asarray(measured, dtype=float)

        feedforward = target / self.max_speed
        error = target - actual
        proposed = feedforward + self.kp * error + self._integral

        # Anti-windup, the conditional-integration kind: stop accumulating when
        # the output is already against its limit and the error would push it
        # further. Clamping the integral alone is not enough -- it still fills
        # up during a stall and then has to be unwound, which is the lurch when
        # a stuck robot comes free.
        limit = self.max_effort
        pushing_out = (np.abs(proposed) >= limit) & (np.sign(error) == np.sign(proposed))
        self._integral = np.where(
            pushing_out,
            self._integral,
            np.clip(
                self._integral + self.ki * error * dt, -self.integral_limit, self.integral_limit
            ),
        )
        # A commanded stop should not be held up by history.
        self._integral[target == 0.0] = 0.0

        effort = feedforward + self.kp * error + self._integral
        saturated = bool(np.any(np.abs(effort) > limit))
        # Scale both sides together rather than clipping them independently:
        # clipping one side of a turn changes the turn radius, which is the
        # same mistake DriveCommand.from_body is careful to avoid upstream.
        peak = float(np.max(np.abs(effort)))
        if peak > limit:
            effort = effort * (limit / peak)

        if self.slew > 0:
            # Ramp up gently, cut immediately. A slew limit exists to stop a
            # step command browning out the supply or snapping the drivetrain,
            # and neither happens on the way down -- removing power is the one
            # move that is always safe. Limiting the fall as well would put a
            # delay between "stop" and stopping, which is exactly the delay a
            # watchdog exists to avoid.
            step = self.slew * dt
            rising = np.abs(effort) > np.abs(self._effort)
            crossing = np.sign(effort) * np.sign(self._effort) < 0
            limited = np.clip(effort, self._effort - step, self._effort + step)
            effort = np.where(rising | crossing, limited, effort)
        self._effort = effort

        return MotorEffort(
            left=float(effort[0]),
            right=float(effort[1]),
            brush=command.brush,
            pump=command.pump,
            saturated=saturated,
        )
