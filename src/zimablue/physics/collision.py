"""Contact with walls and obstacles.

Penetration-based resolution against the pool's segment geometry.  The robot is
treated as a disc for contact purposes: at the raster resolutions ZimaBlue runs
at, a disc and the true rectangle differ by less than a cell, and a disc costs
one distance query instead of a polygon intersection on every tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zimablue.pool import Pool

__all__ = ["Contact", "resolve"]


@dataclass(frozen=True)
class Contact:
    """The outcome of one collision query."""

    touching: bool
    x: float
    y: float
    """Corrected position."""

    penetration: float
    """How far the robot had to be pushed out, m."""

    normal: tuple[float, float] = (0.0, 0.0)
    """Unit surface normal pointing away from the wall, into the water."""

    flags: tuple[bool, bool, bool, bool] = (False, False, False, False)
    """Front, left, right, rear bump switches."""

    is_obstacle: bool = False

    @property
    def any(self) -> bool:
        return self.touching


def _bump_flags(bearing: float) -> tuple[bool, bool, bool, bool]:
    """Map a body-frame bearing to the wall into four bump switches."""
    quarter = np.pi / 4
    if -quarter <= bearing <= quarter:
        return (True, False, False, False)
    if quarter < bearing <= 3 * quarter:
        return (False, True, False, False)
    if -3 * quarter <= bearing < -quarter:
        return (False, False, True, False)
    return (False, False, False, True)


def resolve(pool: Pool, x: float, y: float, heading: float, radius: float) -> Contact:
    """Push the robot out of any surface it overlaps and report the contact.

    Handles both cases that matter: the robot is inside the pool but too close
    to a wall, and the robot has been pushed *through* a wall by a large step.
    The second is rare at 50 Hz but must not silently teleport the robot
    outside the pool, which would corrupt every downstream metric.
    """
    inside = bool(pool.contains(x, y))
    distance, wall_x, wall_y, is_obstacle = pool.nearest_wall(x, y)

    if inside and distance >= radius:
        return Contact(touching=False, x=x, y=y, penetration=0.0)

    dx, dy = x - wall_x, y - wall_y
    norm = float(np.hypot(dx, dy))
    if norm < 1e-12:
        # Exactly on the surface: fall back to the direction of the pool
        # centroid so the push has a defined direction.
        centre = pool.navigable.centroid
        dx, dy = centre.x - x, centre.y - y
        norm = max(float(np.hypot(dx, dy)), 1e-9)

    if inside:
        nx, ny = dx / norm, dy / norm
        penetration = radius - distance
    else:
        # Outside: the outward vector points the wrong way.
        nx, ny = -dx / norm, -dy / norm
        penetration = radius + distance

    corrected_x = wall_x + nx * radius
    corrected_y = wall_y + ny * radius

    bearing = float(
        (np.arctan2(wall_y - corrected_y, wall_x - corrected_x) - heading + np.pi) % (2 * np.pi)
        - np.pi
    )
    return Contact(
        touching=True,
        x=corrected_x,
        y=corrected_y,
        penetration=float(penetration),
        normal=(nx, ny),
        flags=_bump_flags(bearing),
        is_obstacle=bool(is_obstacle),
    )
