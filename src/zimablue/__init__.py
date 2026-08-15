"""ZimaBlue -- simulate, test, and replay robotic pool cleaners.

The public API is the domain model.  Backends, integrators and renderers are
implementation details reached through it, never around it.

    >>> import zimablue as zb
    >>> pool = zb.make_pool("kidney")
    >>> robot = zb.make_robot("tracked")
"""

from __future__ import annotations

from zimablue._version import __version__
from zimablue.geometry import Grid
from zimablue.pool import (
    MATERIALS,
    POOL_PRESETS,
    CompositeDepth,
    ConstantDepth,
    DepthModel,
    Drain,
    Obstacle,
    PlaneSlopeDepth,
    Pool,
    PoolFeature,
    Return,
    Skimmer,
    Stairs,
    SurfaceMaterial,
    Water,
    make_pool,
)
from zimablue.registry import Registry
from zimablue.rng import RngTree
from zimablue.robot import (
    ROBOT_PRESETS,
    Battery,
    Brush,
    Chassis,
    Cleaner,
    CleaningSystem,
    DriveCommand,
    DriveUnit,
    Filter,
    Locomotion,
    Motor,
    PowerSystem,
    Pump,
    make_robot,
)
from zimablue.sensors import (
    IMU,
    ContactSensor,
    Encoder,
    PressureSensor,
    Reading,
    Sensor,
    SensorConfig,
    SensorContext,
    SensorFault,
    SensorSuite,
    Sonar,
)

__all__ = [
    "IMU",
    "MATERIALS",
    "POOL_PRESETS",
    "ROBOT_PRESETS",
    "Battery",
    "Brush",
    "Chassis",
    "Cleaner",
    "CleaningSystem",
    "CompositeDepth",
    "ConstantDepth",
    "ContactSensor",
    "DepthModel",
    "Drain",
    "DriveCommand",
    "DriveUnit",
    "Encoder",
    "Filter",
    "Grid",
    "Locomotion",
    "Motor",
    "Obstacle",
    "PlaneSlopeDepth",
    "Pool",
    "PoolFeature",
    "PowerSystem",
    "PressureSensor",
    "Pump",
    "Reading",
    "Registry",
    "Return",
    "RngTree",
    "Sensor",
    "SensorConfig",
    "SensorContext",
    "SensorFault",
    "SensorSuite",
    "Skimmer",
    "Sonar",
    "Stairs",
    "SurfaceMaterial",
    "Water",
    "__version__",
    "make_pool",
    "make_robot",
]
