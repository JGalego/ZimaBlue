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
from zimablue.calibration import (
    CalibrationResult,
    CalibrationStep,
    Parameter,
    TwinCalibrator,
    trajectory_loss,
)
from zimablue.controllers import (
    CONTROLLERS,
    ControlInput,
    Controller,
    OccupancyMap,
    SystematicCoverage,
)
from zimablue.counterfactual import CounterfactualResult, run_counterfactual
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
from zimablue.experiments import (
    AutonomousExperiment,
    CandidateResult,
    ExperimentGeneration,
    ExperimentObjective,
    ExperimentResult,
)
from zimablue.fleet import Blackboard, Fleet, FleetMember, FleetMetrics, FleetResult, spread_poses
from zimablue.geometry import Grid
from zimablue.imaging import PoolTrace, pool_from_image, trace_pool
from zimablue.metrics import Metrics, SpatialMetrics
from zimablue.notebook import PoolPreview, preview
from zimablue.phone import (
    DepthObservation,
    PhoneReconstruction,
    PhoneView,
    fit_phone_depth,
    fuse_phone_traces,
    pool_from_phones,
)
from zimablue.physics import (
    DifferentiableTrajectory,
    DriveJacobians,
    differentiable_drive_step,
    differentiable_rollout,
)
from zimablue.planners import (
    PARTITIONS,
    PLANNERS,
    CoveragePath,
    CoveragePlanner,
    OnlineCoverage,
    Partition,
    PathFollower,
    Territory,
    make_partition,
    make_planner,
    mstc,
    partitioned,
)
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
from zimablue.shadow import ResidualStats, ShadowHealth, ShadowTwin
from zimablue.simulation import RunResult, Simulation
from zimablue.sketch import SketchSegmenter, pool_from_sketch, trace_sketch
from zimablue.world import World

__all__ = [
    "CONTROLLERS",
    "DIRT_PRESETS",
    "DIRT_TYPES",
    "IMU",
    "MATERIALS",
    "PARTITIONS",
    "PLANNERS",
    "POOL_PRESETS",
    "ROBOT_PRESETS",
    "AutonomousExperiment",
    "BatchResult",
    "Battery",
    "Blackboard",
    "Brush",
    "CalibrationResult",
    "CalibrationStep",
    "CandidateResult",
    "Chassis",
    "Cleaner",
    "CleaningSystem",
    "CompositeDepth",
    "ConstantDepth",
    "ContactSensor",
    "ControlInput",
    "Controller",
    "CounterfactualResult",
    "CoveragePath",
    "CoveragePlanner",
    "DebrisSpec",
    "DepthModel",
    "DepthObservation",
    "DifferentiableTrajectory",
    "DirtField",
    "DirtSpec",
    "DirtState",
    "DirtType",
    "Drain",
    "DriveCommand",
    "DriveJacobians",
    "DriveUnit",
    "Encoder",
    "EstimatorConfig",
    "ExperimentGeneration",
    "ExperimentObjective",
    "ExperimentResult",
    "Filter",
    "Fleet",
    "FleetMember",
    "FleetMetrics",
    "FleetResult",
    "Grid",
    "LayerSpec",
    "Locomotion",
    "Metrics",
    "Motor",
    "Obstacle",
    "OccupancyMap",
    "OnlineCoverage",
    "Parameter",
    "Partition",
    "PathFollower",
    "PhoneReconstruction",
    "PhoneView",
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
    "ResidualStats",
    "Return",
    "RngTree",
    "RunResult",
    "Scenario",
    "Sensor",
    "SensorConfig",
    "SensorContext",
    "SensorFault",
    "SensorSuite",
    "ShadowHealth",
    "ShadowTwin",
    "Simulation",
    "SketchSegmenter",
    "Skimmer",
    "Sonar",
    "SpatialMetrics",
    "Stairs",
    "SurfaceMaterial",
    "SystematicCoverage",
    "Territory",
    "TwinCalibrator",
    "Water",
    "World",
    "__version__",
    "differentiable_drive_step",
    "differentiable_rollout",
    "fit_phone_depth",
    "fuse_phone_traces",
    "get_material",
    "load_scenario",
    "make_dirt",
    "make_partition",
    "make_planner",
    "make_pool",
    "make_robot",
    "mstc",
    "partitioned",
    "pool_from_image",
    "pool_from_phones",
    "pool_from_sketch",
    "preview",
    "run_batch",
    "run_counterfactual",
    "spread_poses",
    "trace_pool",
    "trace_sketch",
    "trajectory_loss",
]
