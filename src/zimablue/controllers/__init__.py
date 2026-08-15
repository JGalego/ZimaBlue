"""Autonomy. Replaceable by design -- see :mod:`zimablue.controllers.base`."""

from __future__ import annotations

from zimablue.controllers.base import CONTROLLERS, ControlInput, Controller
from zimablue.controllers.baseline import (
    BaselineCoverage,
    BaselineTuning,
    HeadingEstimator,
    Phase,
)
from zimablue.controllers.simple import LawnmowerOracle, RandomBounce

__all__ = [
    "CONTROLLERS",
    "BaselineCoverage",
    "BaselineTuning",
    "ControlInput",
    "Controller",
    "HeadingEstimator",
    "LawnmowerOracle",
    "Phase",
    "RandomBounce",
]
