"""Simulation backends.

``SimulationBackend`` is the seam that keeps heavy engines out of the domain
model.  Only ``Fast2DBackend`` exists today; see
``docs/architecture.md`` for the intended Isaac Sim implementation.
"""

from __future__ import annotations

from zimablue.backends.base import (
    BACKENDS,
    Event,
    SimState,
    SimulationBackend,
    StepResult,
)
from zimablue.backends.fast2d import Fast2DBackend

__all__ = [
    "BACKENDS",
    "Event",
    "Fast2DBackend",
    "SimState",
    "SimulationBackend",
    "StepResult",
]
