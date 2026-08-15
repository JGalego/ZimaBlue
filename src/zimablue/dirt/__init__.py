"""Dirt: types, spatial state, and deterministic generation."""

from __future__ import annotations

from zimablue.dirt.field import DebrisSet, DirtField, DirtState
from zimablue.dirt.generators import (
    DIRT_PRESETS,
    PATTERNS,
    DebrisSpec,
    DirtSpec,
    LayerSpec,
    make_dirt,
)
from zimablue.dirt.types import (
    DIRT_TYPES,
    DirtType,
    get_dirt_type,
    settling_velocity,
    stokes_settling_velocity,
)

__all__ = [
    "DIRT_PRESETS",
    "DIRT_TYPES",
    "PATTERNS",
    "DebrisSet",
    "DebrisSpec",
    "DirtField",
    "DirtSpec",
    "DirtState",
    "DirtType",
    "LayerSpec",
    "get_dirt_type",
    "make_dirt",
    "settling_velocity",
    "stokes_settling_velocity",
]
