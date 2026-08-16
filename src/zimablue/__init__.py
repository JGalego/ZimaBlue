"""ZimaBlue -- simulate, test, and replay robotic pool cleaners.

The public API is the domain model.  Backends, integrators and renderers are
implementation details reached through it, never around it.

    >>> import zimablue as zb
    >>> pool = zb.make_pool("kidney")
    >>> robot = zb.make_robot("tracked")
"""

from __future__ import annotations

from zimablue._version import __version__
from zimablue.batch import BatchResult, run_batch
from zimablue.controllers import (
    CONTROLLERS,
    ControlInput,
    Controller,
    OccupancyMap,
    SystematicCoverage,
)
from zimablue.dirt import (
    DIRT_PRESETS,
    DIRT_TYPES,
    DebrisSpec,
    DirtField,
    DirtSpec,
    DirtState,
    DirtType,
    LayerSpec,
    make_dirt,
)
from zimablue.estimation import EstimatorConfig, PoseEstimate, PoseEstimator
from zimablue.geometry import Grid
from zimablue.imaging import PoolTrace, pool_from_image, trace_pool
from zimablue.metrics import Metrics, SpatialMetrics
from zimablue.notebook import PoolPreview, preview
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
    get_material,
    make_pool,
)
from zimablue.recording import Recording
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
from zimablue.scenarios import Scenario, load_scenario
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
from zimablue.simulation import RunResult, Simulation
from zimablue.world import World

__all__ = [
    "CONTROLLERS",
    "DIRT_PRESETS",
    "DIRT_TYPES",
    "IMU",
    "MATERIALS",
    "POOL_PRESETS",
    "ROBOT_PRESETS",
    "BatchResult",
    "Battery",
    "Brush",
    "Chassis",
    "Cleaner",
    "CleaningSystem",
    "CompositeDepth",
    "ConstantDepth",
    "ContactSensor",
    "ControlInput",
    "Controller",
    "DebrisSpec",
    "DepthModel",
    "DirtField",
    "DirtSpec",
    "DirtState",
    "DirtType",
    "Drain",
    "DriveCommand",
    "DriveUnit",
    "Encoder",
    "EstimatorConfig",
    "Filter",
    "Grid",
    "LayerSpec",
    "Locomotion",
    "Metrics",
    "Motor",
    "Obstacle",
    "OccupancyMap",
    "PlaneSlopeDepth",
    "Pool",
    "PoolFeature",
    "PoolPreview",
    "PoolTrace",
    "PoseEstimate",
    "PoseEstimator",
    "PowerSystem",
    "PressureSensor",
    "Pump",
    "Reading",
    "Recording",
    "Registry",
    "Return",
    "RngTree",
    "RunResult",
    "Scenario",
    "Sensor",
    "SensorConfig",
    "SensorContext",
    "SensorFault",
    "SensorSuite",
    "Simulation",
    "Skimmer",
    "Sonar",
    "SpatialMetrics",
    "Stairs",
    "SurfaceMaterial",
    "SystematicCoverage",
    "Water",
    "World",
    "__version__",
    "get_material",
    "load_scenario",
    "make_dirt",
    "make_pool",
    "make_robot",
    "pool_from_image",
    "preview",
    "run_batch",
    "trace_pool",
]
