"""The backend boundary.

A backend owns **dynamics and sensing** and nothing else.  Dirt accounting,
metrics, recording and replay are computed by shared code from the state a
backend returns, so a new backend inherits all of them for free -- and cannot
quietly change how they are measured.

This is the interface a future Isaac Sim backend has to satisfy; see
``docs/architecture.md`` for the intended shape of that implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from zimablue.registry import Registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.rng import RngTree
    from zimablue.robot import Cleaner, DriveCommand
    from zimablue.sensors import Reading
    from zimablue.world import World

__all__ = ["BACKENDS", "Event", "SimState", "SimulationBackend", "StepResult"]


@dataclass(frozen=True)
class Event:
    """Something discrete worth recording.

    Events are sparse and carry a payload, which is why they are stored
    separately from the dense per-frame columns in a recording.
    """

    time: float
    kind: str
    """``collision`` | ``stuck`` | ``unstuck`` | ``filter_full`` | ``battery_low``
    | ``battery_empty`` | ``fault`` | ``debris_collected`` | ``debris_blocked``"""

    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "kind": self.kind, "detail": self.detail}


@dataclass
class SimState:
    """Everything that changes as a run proceeds.

    Deliberately flat and primitive: a frame of a recording is very nearly a
    copy of this struct, which keeps the recording layer simple and makes it
    obvious what is and is not reproducible.
    """

    time: float = 0.0
    step: int = 0

    # Pose and motion (ground truth)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0
    """Body-frame forward ground speed, m/s."""

    omega: float = 0.0
    accel_forward: float = 0.0
    accel_lateral: float = 0.0

    # Drivetrain
    wheel_left: float = 0.0
    wheel_right: float = 0.0
    """Actual track-surface speeds, m/s. Differ from ground speed under slip."""

    slip_left: float = 0.0
    slip_right: float = 0.0

    # Environment coupling
    depth: float = 0.0
    contacts: tuple[bool, bool, bool, bool] = (False, False, False, False)
    collided: bool = False
    stuck: bool = False
    stuck_time: float = 0.0

    # Consumables
    battery_wh: float = 0.0
    energy_used_wh: float = 0.0
    power_w: float = 0.0
    filter_load: float = 0.0

    # Odometers
    distance: float = 0.0
    dirt_collected: float = 0.0
    """Mass in the filter, grams. Not the same as dirt removed from the pool:
    fines pass through the mesh and settle again."""

    dirt_removed: float = 0.0
    """Mass taken out of the pool's dirt field this run, grams."""

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.heading)

    @property
    def battery_fraction(self) -> float:
        return self._battery_fraction

    def copy(self) -> SimState:
        from dataclasses import replace

        return replace(self)

    # Set by the backend at reset so battery_fraction has a denominator.
    _battery_capacity: float = 1.0
    _battery_fraction: float = 1.0


@dataclass
class StepResult:
    """One tick's output."""

    state: SimState
    events: list[Event] = field(default_factory=list)
    observations: dict[str, Reading] = field(default_factory=dict)


@runtime_checkable
class SimulationBackend(Protocol):
    """Dynamics and sensing for one robot in one world."""

    name: str

    def reset(self, world: World, robot: Cleaner, rng: RngTree) -> SimState:
        """Place the robot and return the initial state."""
        ...

    def step(self, state: SimState, command: DriveCommand, dt: float) -> StepResult:
        """Advance by ``dt`` and return the new state plus any events."""
        ...

    def sense(self, state: SimState) -> dict[str, Reading]:
        """Poll the robot's sensors against the current ground truth."""
        ...

    def close(self) -> None:
        """Release any resources. A no-op for the 2D backend."""
        ...


BACKENDS: Registry[SimulationBackend] = Registry("backend")


def wrap_angle(theta: float) -> float:
    """Wrap a scalar angle into ``(-pi, pi]``."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)
