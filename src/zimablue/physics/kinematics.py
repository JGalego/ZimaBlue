"""Differential-drive motion with slip.

Small enough to read in one sitting, which is the point: navigation bugs in a
testbed must be attributable to the controller, not to the integrator.
"""

from __future__ import annotations

import numpy as np

from zimablue.robot.components import Locomotion

__all__ = ["exact_arc_step", "slip_factors"]


def exact_arc_step(
    x: float, y: float, heading: float, v: float, omega: float, dt: float
) -> tuple[float, float, float]:
    """Integrate a constant-twist motion exactly over ``dt``.

    Euler integration of a turning differential drive accumulates a systematic outward
    error on every arc, which over a 30-minute run looks exactly like odometry
    drift and would be indistinguishable from the drift the sensors are
    *supposed* to model.  Closed-form arc integration removes that confound:
    the pose is exact for the constant-twist assumption, so all remaining drift
    comes from slip and sensor noise, which is where it belongs.
    """
    if abs(omega) < 1e-9:
        return (x + v * np.cos(heading) * dt, y + v * np.sin(heading) * dt, heading)
    radius = v / omega
    new_heading = heading + omega * dt
    nx = x + radius * (np.sin(new_heading) - np.sin(heading))
    ny = y - radius * (np.cos(new_heading) - np.cos(heading))
    return (float(nx), float(ny), float(new_heading))


def slip_factors(
    locomotion: Locomotion,
    surface_friction: float,
    v_left: float,
    v_right: float,
    *,
    noise_left: float = 0.0,
    noise_right: float = 0.0,
) -> tuple[float, float]:
    """Fraction of commanded track speed *lost* to slip on each side, in ``[0, 1)``.

    Two contributions, both grounded in how a tracked cleaner behaves:

    * **Baseline slip** rises as available grip falls. Grip is the pool
      surface's friction times the drivetrain's traction coefficient, so the
      same robot slips more on smooth tile than on rough plaster.
    * **Turn slip** rises with the speed difference between the tracks. A track
      with a long contact patch has to scrub sideways to yaw, and
      ``turn_resistance`` sets how much that costs.

    The noise arguments accept pre-drawn samples rather than an RNG so the
    caller controls which stream they come from -- keeping this function pure
    and the determinism contract auditable.
    """
    grip = max(surface_friction * locomotion.traction, 1e-3)
    base = np.clip(0.02 / grip, 0.0, 0.25)

    differential = abs(v_right - v_left)
    scrub = locomotion.turn_resistance * differential / max(locomotion.max_speed, 1e-6)
    turn = np.clip(scrub * 0.5, 0.0, 0.4)

    left = float(np.clip(base + turn + noise_left, 0.0, 0.95))
    right = float(np.clip(base + turn + noise_right, 0.0, 0.95))
    return (left, right)
