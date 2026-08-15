"""The actuation command -- the whole interface between control and physics.

Keeping this tiny is deliberate.  A controller cannot reach into the simulator,
and the simulator cannot ask a controller for anything it did not send, so
swapping in a different autonomy stack is a local change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.robot.components import Locomotion

__all__ = ["DriveCommand"]


@dataclass(frozen=True)
class DriveCommand:
    """What the controller asks the robot to do this tick."""

    left: float = 0.0
    """Target left track speed, m/s."""

    right: float = 0.0
    """Target right track speed, m/s."""

    brush: bool = True
    """Whether the agitation brush is running."""

    pump: float = 1.0
    """Pump duty, 0-1."""

    @classmethod
    def stop(cls) -> DriveCommand:
        """Everything off -- used at battery cutoff and on termination."""
        return cls(left=0.0, right=0.0, brush=False, pump=0.0)

    @classmethod
    def from_body(
        cls,
        v: float,
        omega: float,
        locomotion: Locomotion,
        *,
        brush: bool = True,
        pump: float = 1.0,
    ) -> DriveCommand:
        """Build a command from a body-frame velocity, honouring motor limits.

        When the requested twist needs more speed than the motors have, both
        sides are scaled by the same factor.  Clipping them independently would
        change the turn radius, which is a classic and surprisingly common
        source of drift in differential-drive code.
        """
        left, right = locomotion.to_wheel_speeds(v, omega)
        limit = locomotion.max_speed
        peak = max(abs(left), abs(right))
        if peak > limit and peak > 0:
            scale = limit / peak
            left *= scale
            right *= scale
        return cls(left=float(left), right=float(right), brush=brush, pump=float(pump))

    def body_velocity(self, locomotion: Locomotion) -> tuple[float, float]:
        """The ``(v, omega)`` this command represents."""
        return locomotion.to_body_velocity(self.left, self.right)

    def clamped(self, locomotion: Locomotion) -> DriveCommand:
        """A copy with speeds inside motor limits and pump duty in ``[0, 1]``."""
        return DriveCommand(
            left=locomotion.left.motor.clamp_speed(self.left),
            right=locomotion.right.motor.clamp_speed(self.right),
            brush=self.brush,
            pump=float(np.clip(self.pump, 0.0, 1.0)),
        )

    def as_array(self) -> np.ndarray:
        """``[left, right, brush, pump]`` -- the recorded representation."""
        return np.array([self.left, self.right, 1.0 if self.brush else 0.0, self.pump], dtype=float)
