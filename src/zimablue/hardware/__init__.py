"""Running a ZimaBlue controller on something that is not a simulator.

The whole port fits behind two dataclasses that already existed.  A controller
is handed a :class:`~zimablue.controllers.base.ControlInput` -- a clock, the
latest reading per sensor, battery, filter load, its own specification -- and
returns a :class:`~zimablue.robot.DriveCommand`.  Nothing in either of them is
simulator-specific, which was the point of writing them that way, and it means
the code that turns those two into a working robot is small:

:mod:`~zimablue.hardware.sources`
    Readings from a real device, or replayed from a recording with jitter and
    dropouts on top.

:mod:`~zimablue.hardware.motors`
    A PI speed loop, because a controller asks for metres per second and a
    driver takes duty cycle.

:mod:`~zimablue.hardware.safety`
    A watchdog for the failures the simulator is not obliged to have.

:mod:`~zimablue.hardware.runtime`
    The loop: sense, decide, actuate, record. Writes the same ``.zbr`` the
    simulator writes, so a real run replays in the ordinary viewer.

:mod:`~zimablue.hardware.logs`
    Reading trajectories logged off other people's robots, so some of this
    library's numbers can be checked against motion it did not generate.

Nothing here needs an extra installed. It is all standard library and NumPy,
because a robot is the last place you want a dependency tree.

What it does not give you is cleaning metrics. Coverage and dirt removed are
computed from the true pose against the true dirt field, and a robot has
neither -- see :meth:`~zimablue.hardware.runtime.HardwareRun.metrics` and
``docs/hardware.md``.
"""

from __future__ import annotations

from zimablue.hardware.logs import Trajectory, read_tum_trajectory
from zimablue.hardware.motors import MotorEffort, WheelSpeedLoop
from zimablue.hardware.runtime import HardwareRun, HardwareRuntime, Tick
from zimablue.hardware.safety import SafetyLimits, Watchdog
from zimablue.hardware.sources import (
    DeviceSource,
    ReadingSource,
    RecordedSource,
    TrajectorySource,
)

__all__ = [
    "DeviceSource",
    "HardwareRun",
    "HardwareRuntime",
    "MotorEffort",
    "ReadingSource",
    "RecordedSource",
    "SafetyLimits",
    "Tick",
    "Trajectory",
    "TrajectorySource",
    "Watchdog",
    "WheelSpeedLoop",
    "read_tum_trajectory",
]
