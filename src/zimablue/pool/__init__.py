"""Pool geometry, floor, features and water."""

from __future__ import annotations

from zimablue.pool.depth import (
    CompositeDepth,
    ConstantDepth,
    DepthModel,
    PlaneSlopeDepth,
)
from zimablue.pool.features import (
    Drain,
    Obstacle,
    PoolFeature,
    Return,
    Skimmer,
    Stairs,
    flow_field,
)
from zimablue.pool.materials import MATERIALS, SurfaceMaterial, get_material
from zimablue.pool.pool import DEFAULT_CELL, Pool, Water
from zimablue.pool.presets import POOL_PRESETS, make_pool

__all__ = [
    "DEFAULT_CELL",
    "MATERIALS",
    "POOL_PRESETS",
    "CompositeDepth",
    "ConstantDepth",
    "DepthModel",
    "Drain",
    "Obstacle",
    "PlaneSlopeDepth",
    "Pool",
    "PoolFeature",
    "Return",
    "Skimmer",
    "Stairs",
    "SurfaceMaterial",
    "Water",
    "flow_field",
    "get_material",
    "make_pool",
]
