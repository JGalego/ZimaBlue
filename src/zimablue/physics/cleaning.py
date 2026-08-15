"""How the robot actually removes dirt.

This is where ZimaBlue's central claim is implemented, so it is worth being
explicit about the model rather than burying it in coefficients.

Removal of a continuous dirt layer over one tick is exponential::

    fraction = 1 - exp(-rate * dt)
    rate     = BASE_RATE * suction * releasable / pickup_difficulty

with the interesting term being ``releasable``: the fraction of the dirt that
is *available* to be sucked up at all.

    releasable = (1 - bond) + bond * min(agitation / (3*bond + 0.3), 1)

Loose dirt (``bond ~ 0``) is fully available to suction alone.  Adhered dirt
is not: the bonded share only becomes available in proportion to how hard the
brush is working, and if the brush is off, ``agitation`` is zero and that share
stays on the wall no matter how long the robot sits there.

That single expression produces the behaviour the whole project exists to
measure: a cleaner with a weak or disabled brush drives over algae, reports
excellent coverage, and removes almost none of it.

No CFD. See ``docs/research.md`` section 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from zimablue.dirt.field import DirtState
from zimablue.geometry import Window
from zimablue.pool import Pool
from zimablue.robot import Cleaner

__all__ = ["CleaningOutcome", "apply_cleaning"]

FloatArray = NDArray[np.float64]

BASE_RATE = 6.0
"""Removal rate constant, 1/s, at full suction on fully-releasable dirt.

Calibrated so that one pass at the reference robot's cruise speed removes most
of the loose dirt beneath it: at 0.25 m/s a 10 cm cell is under the head for
0.4 s, giving ``1 - exp(-6 * 0.4) ~= 91%``.
"""

PASSIVE_RELEASE = 0.05
"""Share of bonded dirt that comes free without any agitation at all.

Not zero: water shear and the hull scraping past do something. Small enough
that a brush-less robot cannot clean an algae bloom by persistence.
"""


@dataclass
class CleaningOutcome:
    """What one tick of cleaning accomplished."""

    removed: dict[str, float] = field(default_factory=dict)
    """Mass lifted off the surface per dirt type, grams."""

    captured: float = 0.0
    """Mass actually retained by the filter, grams."""

    passed_through: float = 0.0
    """Mass lifted but too fine for the mesh; it re-settles nearby."""

    debris_collected: int = 0
    debris_blocked: int = 0
    """Items too large for the intake; bumped aside instead of swallowed."""

    filter_load: float = 0.0
    filter_full: bool = False

    @property
    def total_removed(self) -> float:
        return float(sum(self.removed.values()))


def apply_cleaning(
    pool: Pool,
    dirt: DirtState,
    robot: Cleaner,
    *,
    x: float,
    y: float,
    heading: float,
    speed: float,
    brush_on: bool,
    pump_duty: float,
    filter_load: float,
    dt: float,
    cell: float,
) -> CleaningOutcome:
    """Remove dirt under the cleaning head for one tick."""
    cleaning = robot.cleaning
    outcome = CleaningOutcome(filter_load=filter_load)

    clog = cleaning.filter.clog_fraction(filter_load)
    suction = cleaning.pump.suction(pump_duty, clog)
    if suction <= 0.0 and not brush_on:
        return outcome

    agitation = cleaning.agitation(abs(speed)) * pool.material.brush_gain if brush_on else 0.0

    window = pool.grid(cell).window(x, y, 0.5 * robot.swath_width)
    if window is None or window.count == 0:
        return outcome

    # --- continuous layers ------------------------------------------------
    fractions: dict[str, float] = {}
    for name, dirt_type in dirt.field.types.items():
        bond = float(np.clip(dirt_type.adhesion * pool.material.adhesion_factor, 0.0, 1.0))
        loosened = min(agitation / (3.0 * bond + 0.3), 1.0) if bond > 0 else 1.0
        releasable = (1.0 - bond) + bond * max(loosened, PASSIVE_RELEASE)
        rate = BASE_RATE * suction * releasable / dirt_type.pickup_difficulty
        fractions[name] = float(1.0 - np.exp(-rate * dt))

    removed = dirt.field.remove_window(window, fractions)
    outcome.removed = dict(removed)

    # --- filtration -------------------------------------------------------
    # Anything finer than the mesh is lifted but not kept: it goes back into
    # suspension and settles again nearby. This is why a coarse filter leaves a
    # pool that looks cleaned but measures dirty.
    for name, mass in removed.items():
        dirt_type = dirt.field.types[name]
        retained = cleaning.filter.retains(dirt_type.particle_size)
        kept = mass * retained
        escaped = mass - kept
        outcome.captured += kept
        outcome.passed_through += escaped
        if escaped > 0:
            _resettle(dirt, name, window, escaped)

    # --- discrete debris --------------------------------------------------
    reach = 0.5 * cleaning.pump.intake_width + 0.5 * robot.chassis.length
    nose_x = x + np.cos(heading) * robot.chassis.length * 0.35
    nose_y = y + np.sin(heading) * robot.chassis.length * 0.35
    near = dirt.debris.near(nose_x, nose_y, reach)
    if near.any() and suction > 0.2:
        swallowable = near & (dirt.debris.size <= cleaning.pump.max_debris_size)
        blocked = near & ~swallowable
        mass, count = dirt.debris.collect(swallowable)
        outcome.captured += mass
        outcome.removed["debris"] = outcome.removed.get("debris", 0.0) + mass
        outcome.debris_collected = count
        if blocked.any():
            # Too big for the intake: shoved along instead of collected.
            outcome.debris_blocked = int(blocked.sum())
            dirt.debris.nudge(
                blocked,
                float(np.cos(heading) * speed * dt),
                float(np.sin(heading) * speed * dt),
                inside=pool.contains,
            )

    outcome.filter_load = filter_load + outcome.captured
    outcome.filter_full = outcome.filter_load >= cleaning.filter.capacity
    return outcome


def _resettle(dirt: DirtState, layer: str, window: Window, mass: float) -> None:
    """Put mass that slipped the filter back into the water column nearby.

    Deposition only.  Spreading those fines outward is a diffusion step, and
    diffusing every 20 ms over a patch that moves 5 mm in that time is wasted
    work -- the backend batches it onto the same slow cadence as the flow
    drift, which is where the rest of the water transport happens.
    """
    count = window.count
    if count <= 0:
        return
    patch = window.view(dirt.field.layers[layer])
    patch[window.mask] += mass / count
