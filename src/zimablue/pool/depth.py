"""Depth models.

A pool's floor is described by a ``DepthModel``: anything that can answer
"how deep is the water at (x, y)?" for whole arrays at once.  Keeping this
behind a small protocol is what stops pool presets turning into a pile of
conditionals -- a sloped pool and a flat pool differ by which model they hold,
not by a branch in ``Pool``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

__all__ = [
    "CompositeDepth",
    "ConstantDepth",
    "DepthModel",
    "PlaneSlopeDepth",
]

FloatArray = NDArray[np.float64]


@runtime_checkable
class DepthModel(Protocol):
    """Water depth as a function of planar position."""

    @property
    def max_depth(self) -> float:
        """Deepest water the model can return, in metres."""

    def depth_at(self, x: FloatArray | float, y: FloatArray | float) -> FloatArray:
        """Depth in metres at each ``(x, y)``; shape follows the inputs."""


@dataclass(frozen=True)
class ConstantDepth:
    """Flat floor."""

    depth: float

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")

    @property
    def max_depth(self) -> float:
        return self.depth

    def depth_at(self, x: FloatArray | float, y: FloatArray | float) -> FloatArray:
        return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, self.depth, dtype=float)


@dataclass(frozen=True)
class PlaneSlopeDepth:
    """Floor that ramps linearly along a direction.

    ``shallow`` applies at ``origin``; depth increases along the unit vector
    ``(dx, dy)`` reaching ``deep`` after ``length`` metres, then stays flat.
    """

    shallow: float
    deep: float
    origin: tuple[float, float]
    direction: tuple[float, float]
    length: float

    def __post_init__(self) -> None:
        if self.shallow <= 0 or self.deep <= 0:
            raise ValueError("shallow and deep depths must be positive")
        if self.length <= 0:
            raise ValueError(f"slope length must be positive, got {self.length}")
        norm = float(np.hypot(*self.direction))
        if norm == 0:
            raise ValueError("slope direction must be a non-zero vector")

    @property
    def max_depth(self) -> float:
        return max(self.shallow, self.deep)

    def depth_at(self, x: FloatArray | float, y: FloatArray | float) -> FloatArray:
        dx, dy = self.direction
        norm = float(np.hypot(dx, dy))
        ux, uy = dx / norm, dy / norm
        s = (np.asarray(x, dtype=float) - self.origin[0]) * ux + (
            np.asarray(y, dtype=float) - self.origin[1]
        ) * uy
        t = np.clip(s / self.length, 0.0, 1.0)
        return np.asarray(self.shallow + t * (self.deep - self.shallow), dtype=float)


@dataclass(frozen=True)
class CompositeDepth:
    """A base model overridden inside named regions.

    Regions are tested in order, first match wins, so a caller can layer a
    shallow ledge over a sloped basin without either model knowing about the
    other.
    """

    base: DepthModel
    regions: tuple[tuple[Polygon, DepthModel], ...] = ()

    @property
    def max_depth(self) -> float:
        return max([self.base.max_depth, *(m.max_depth for _, m in self.regions)])

    def depth_at(self, x: FloatArray | float, y: FloatArray | float) -> FloatArray:
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        ya = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.asarray(self.base.depth_at(xa, ya), dtype=float).copy()
        out = np.broadcast_to(out, xa.shape).copy()
        for polygon, model in self.regions:
            minx, miny, maxx, maxy = polygon.bounds
            # Bounding-box prefilter first: the point-in-polygon test below is
            # the expensive part and most cells miss every region.
            candidate = (xa >= minx) & (xa <= maxx) & (ya >= miny) & (ya <= maxy)
            if not candidate.any():
                continue
            from shapely import contains_xy  # local import: keeps module import cheap

            inside = np.zeros_like(candidate)
            inside[candidate] = contains_xy(polygon, xa[candidate], ya[candidate])
            if inside.any():
                out[inside] = np.asarray(model.depth_at(xa[inside], ya[inside]), dtype=float)
        return out
