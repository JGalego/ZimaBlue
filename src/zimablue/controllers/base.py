"""The controller interface.

A controller sees **sensor readings only** -- never ground-truth pose.  That is
the single most important property of this interface: a controller that could
read `state.x` would be solving a different problem from the one a real cleaner
faces, and every coverage number it produced would be a lie.

Replacing the autonomy stack means writing one class with one method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from zimablue.registry import Registry
from zimablue.robot import Cleaner, DriveCommand
from zimablue.sensors import Reading

__all__ = ["CONTROLLERS", "ControlInput", "Controller"]


@dataclass
class ControlInput:
    """Everything a controller is allowed to know this tick."""

    time: float
    """Seconds since the run started. A clock is not cheating -- every real
    cleaner has one."""

    dt: float
    readings: dict[str, Reading]
    """Latest reading per sensor, already noisy, delayed and possibly dropped."""

    battery: float
    """State of charge, 0-1. Available on the real hardware's own bus."""

    filter_load: float
    """Grams in the filter. A real cleaner does not measure this directly; it
    is exposed because suction loss is observable, and a controller that uses
    it should say so."""

    robot: Cleaner
    """The robot's own specification -- its dimensions and limits. Self
    knowledge, not world knowledge."""

    truth: Any = None
    """Ground-truth :class:`SimState`, or ``None``.

    Off by default and only populated when a run explicitly opts in via
    ``Simulation(expose_truth=True)``.  Useful for building an oracle baseline
    or debugging a planner; a controller that needs it is not deployable.
    """

    extras: dict[str, float] = field(default_factory=dict)

    def reading(self, name: str) -> Reading | None:
        return self.readings.get(name)


@runtime_checkable
class Controller(Protocol):
    """Turns sensor readings into drive commands.

    One optional attribute, not in the protocol body because adding a data
    member to a ``runtime_checkable`` Protocol makes every class without it
    fail ``isinstance``: set ``needs_truth = True`` on a controller that reads
    :attr:`ControlInput.truth`, and running it from a scenario or the CLI will
    turn ``expose_truth`` on for you. Both shipped oracles do. Without it the
    controller raises on its first tick, which is the right failure but a
    tedious one to hit from a YAML file.
    """

    name: str

    def reset(self, robot: Cleaner) -> None:
        """Called once before a run begins."""
        ...

    def step(self, control_input: ControlInput) -> DriveCommand:
        """Return this tick's command."""
        ...


CONTROLLERS: Registry[Controller] = Registry("controller", entry_point_group="zimablue.controllers")
