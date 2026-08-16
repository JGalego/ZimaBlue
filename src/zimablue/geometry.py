"""Shared 2D geometry helpers.

Shapely owns *authoring* geometry (building pool shapes, boolean ops, area).
This module owns *hot-loop* geometry: rasterisation and vectorised ray casts
against a flat segment array, which run every simulation tick and would be far
too slow through per-call Shapely objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

__all__ = [
    "Grid",
    "Window",
    "closest_point_on_segments",
    "polygon_segments",
    "raycast",
    "wrap_angle",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Window:
    """A rectangular block of cells plus a boolean mask within it.

    Lets hot-loop code touch only the cells under the robot::

        w = grid.window(x, y, radius)
        layer[w.rows, w.cols][w.mask] -= removed
    """

    rows: slice
    cols: slice
    mask: NDArray[np.bool_]

    def view(self, array: NDArray) -> NDArray:
        """The sub-array this window covers (a view, not a copy)."""
        return array[self.rows, self.cols]

    @property
    def count(self) -> int:
        return int(self.mask.sum())


def wrap_angle(theta: FloatArray | float) -> FloatArray | float:
    """Wrap angles into ``[-pi, pi)``.

    Note the half-open end: exactly +pi comes back as -pi. Every caller
    compares magnitudes or feeds the result to a trig function, so the choice
    does not matter to them -- but the docstring used to claim the opposite
    interval, which is the sort of thing someone eventually relies on.
    """
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Grid:
    """A uniform axis-aligned raster over a world-space rectangle.

    Cell ``(row, col)`` covers ``[minx + col*cell, minx + (col+1)*cell) x
    [miny + row*cell, miny + (row+1)*cell)``.  Row-major, origin bottom-left,
    which matches ``matplotlib.imshow(origin="lower")`` so replay rendering
    needs no flips.
    """

    minx: float
    miny: float
    nrows: int
    ncols: int
    cell: float

    @classmethod
    def covering(cls, bounds: tuple[float, float, float, float], cell: float) -> Grid:
        """Smallest grid with cell size ``cell`` that covers ``bounds``."""
        if cell <= 0:
            raise ValueError(f"cell size must be positive, got {cell}")
        minx, miny, maxx, maxy = bounds
        ncols = max(1, int(np.ceil((maxx - minx) / cell)))
        nrows = max(1, int(np.ceil((maxy - miny) / cell)))
        return cls(minx=float(minx), miny=float(miny), nrows=nrows, ncols=ncols, cell=float(cell))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def cell_area(self) -> float:
        return self.cell * self.cell

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` in the order matplotlib wants."""
        return (
            self.minx,
            self.minx + self.ncols * self.cell,
            self.miny,
            self.miny + self.nrows * self.cell,
        )

    def cell_centers(self) -> tuple[FloatArray, FloatArray]:
        """``(xs, ys)`` grids of cell-centre coordinates, both ``(nrows, ncols)``.

        Cached: this is called from setup code on every raster query, and
        rebuilding two full meshgrids each time showed up clearly in profiles.
        The arrays are marked read-only so a caller cannot corrupt the cache.
        """
        cached = _CENTER_CACHE.get(self)
        if cached is None:
            xs = self.minx + (np.arange(self.ncols) + 0.5) * self.cell
            ys = self.miny + (np.arange(self.nrows) + 0.5) * self.cell
            cached = np.meshgrid(xs, ys)
            cached[0].setflags(write=False)
            cached[1].setflags(write=False)
            _CENTER_CACHE[self] = cached
        return cached

    def window(self, cx: float, cy: float, radius: float) -> Window | None:
        """The smallest cell block containing the disc at ``(cx, cy)``.

        The cleaning head covers a ~34 cm disc inside a pool raster of several
        thousand cells.  Operating on the whole raster every tick costs three
        orders of magnitude more work than the disc does, so every hot-loop
        consumer works through a window instead.  Returns ``None`` when the
        disc misses the grid entirely.
        """
        col0 = int(np.floor((cx - radius - self.minx) / self.cell))
        col1 = int(np.ceil((cx + radius - self.minx) / self.cell))
        row0 = int(np.floor((cy - radius - self.miny) / self.cell))
        row1 = int(np.ceil((cy + radius - self.miny) / self.cell))
        col0, col1 = max(col0, 0), min(col1, self.ncols)
        row0, row1 = max(row0, 0), min(row1, self.nrows)
        if col0 >= col1 or row0 >= row1:
            return None

        # Broadcast a row vector against a column vector rather than building
        # two full 2D meshgrids: same mask, two small 1D allocations.
        xs = self.minx + (np.arange(col0, col1) + 0.5) * self.cell - cx
        ys = self.miny + (np.arange(row0, row1) + 0.5) * self.cell - cy
        mask = (xs[None, :] ** 2 + ys[:, None] ** 2) <= radius * radius
        return Window(rows=slice(row0, row1), cols=slice(col0, col1), mask=mask)

    def index_of(
        self, x: FloatArray | float, y: FloatArray | float
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Cell indices containing ``(x, y)``, clipped to the grid."""
        col = np.floor((np.asarray(x, dtype=float) - self.minx) / self.cell).astype(int)
        row = np.floor((np.asarray(y, dtype=float) - self.miny) / self.cell).astype(int)
        return np.clip(row, 0, self.nrows - 1), np.clip(col, 0, self.ncols - 1)

    def contains(self, x: FloatArray | float, y: FloatArray | float) -> NDArray[np.bool_]:
        """Whether ``(x, y)`` falls inside the grid rectangle."""
        left, right, bottom, top = self.extent
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        return (xa >= left) & (xa < right) & (ya >= bottom) & (ya < top)

    def disk_mask(self, cx: float, cy: float, radius: float) -> NDArray[np.bool_]:
        """Boolean mask of cells whose centre lies within ``radius`` of ``(cx, cy)``."""
        xs, ys = self.cell_centers()
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius


_CENTER_CACHE: dict[Grid, tuple[FloatArray, FloatArray]] = {}
"""Grid is a frozen dataclass, so it hashes by value: two grids with the same
geometry share one cached meshgrid."""


def polygon_segments(polygon: Polygon) -> FloatArray:
    """Flatten a polygon (exterior plus holes) into an ``(n, 4)`` segment array.

    Each row is ``(x0, y0, x1, y1)``.
    """
    rings = [polygon.exterior, *polygon.interiors]
    segments: list[tuple[float, float, float, float]] = []
    for ring in rings:
        coords = np.asarray(ring.coords, dtype=float)
        for (x0, y0), (x1, y1) in pairwise(coords):
            if x0 != x1 or y0 != y1:
                segments.append((x0, y0, x1, y1))
    if not segments:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(segments, dtype=float)


def closest_point_on_segments(
    segments: FloatArray, px: float, py: float
) -> tuple[float, float, float, int]:
    """Nearest point on a segment set to ``(px, py)``.

    Returns ``(distance, nearest_x, nearest_y, segment_index)``.  For an empty
    segment set the distance is ``inf``.
    """
    if segments.shape[0] == 0:
        return (float("inf"), px, py, -1)
    x0, y0, x1, y1 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    # length_sq is never 0: polygon_segments drops degenerate segments.
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / length_sq, 0.0, 1.0)
    nx, ny = x0 + t * dx, y0 + t * dy
    dist = np.hypot(px - nx, py - ny)
    idx = int(np.argmin(dist))
    return (float(dist[idx]), float(nx[idx]), float(ny[idx]), idx)


def raycast(
    segments: FloatArray,
    origin: tuple[float, float],
    angles: FloatArray,
    max_range: float,
) -> FloatArray:
    """Cast rays from ``origin`` at ``angles`` and return hit distances.

    Rays that hit nothing within ``max_range`` return ``max_range``.  Solves
    ``origin + t*d == p0 + u*(p1 - p0)`` for every (ray, segment) pair at once;
    with a few dozen segments and a handful of rays this is a single small
    matrix operation per tick.
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    if segments.shape[0] == 0:
        return np.full(angles.shape, float(max_range))

    ox, oy = origin
    dx = np.cos(angles)[:, None]  # (R, 1)
    dy = np.sin(angles)[:, None]

    x0, y0 = segments[None, :, 0], segments[None, :, 1]  # (1, S)
    ex, ey = segments[None, :, 2] - x0, segments[None, :, 3] - y0

    denom = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        # t along the ray, u along the segment
        t = ((x0 - ox) * ey - (y0 - oy) * ex) / denom
        u = ((x0 - ox) * dy - (y0 - oy) * dx) / denom

    valid = (np.abs(denom) > 1e-12) & (t >= 0.0) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)
    hits = np.where(valid, t, np.inf)
    nearest = hits.min(axis=1)
    return np.where(np.isfinite(nearest), nearest, float(max_range))
