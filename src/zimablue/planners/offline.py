"""Offline coverage planners: given the map, compute the whole path.

Six mechanisms, which between them are the classical half of the
coverage-path-planning literature.  They are grouped here rather than split
across files because they share one substrate -- a polygon, a swath width, and
the question of what order to do things in -- and separating them would mean
six copies of the same clipping code.

============================  =========================================
planner                       the idea
============================  =========================================
``boustrophedon``             back and forth at a fixed angle
``sweep_optimal``             the same, at the angle that turns least
``trapezoidal``               split at every vertex, sweep each cell
``boustrophedon_cells``       split only where connectivity changes
``morse``                     the same, for curved boundaries
``contour``                   follow the wall inward, offset by offset
``wavefront``                 grid, distance transform, steepest descent
``spanning_tree``             cover the pool by circumnavigating a tree
============================  =========================================

Some decisions are shared and worth stating once.

**Lane spacing is the swath, not a guess.** Every sweep here spaces its lanes
by the cleaning width the robot actually has, with a small overlap. A planner
that spaced them by the chassis width would leave stripes, and one that used a
constant would stop being a planner and start being a shape.

**Cell order is a travelling-salesman problem.** Once a pool is cut into cells,
covering them in the order they happened to be created means driving back and
forth across the pool between them. Ordering them well is a TSP over the
adjacency graph, and the difference is tens of metres on a domestic pool. A
nearest-neighbour tour with 2-opt improvement is used throughout: exact TSP is
not worth it for the twenty-odd cells a pool produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from zimablue.planners.base import PLANNERS, CoveragePath

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.pool import Pool
    from zimablue.robot import Cleaner

__all__ = [
    "Boustrophedon",
    "BoustrophedonCells",
    "Contour",
    "Morse",
    "OptimalSweep",
    "SpanningTree",
    "Trapezoidal",
    "Wavefront",
]

FloatArray = NDArray[np.float64]


def _workspace(pool: Pool, robot: Cleaner) -> Polygon:
    """The area the robot's *centre* can occupy.

    The navigable polygon shrunk by the robot's radius. Planning lanes on the
    un-shrunk polygon puts the outermost lane's centre on the wall, and the
    follower then spends the run grinding along it.
    """
    inner = pool.navigable.buffer(-robot.radius)
    if inner.is_empty:
        # A pool barely wider than the robot. Fall back to the centreline
        # rather than returning nothing: one pass down the middle is the
        # correct plan for a channel this size.
        inner = pool.navigable.buffer(-0.45 * min(robot.chassis.width, robot.chassis.length))
    if isinstance(inner, MultiPolygon):
        inner = max(inner.geoms, key=lambda g: g.area)
    return inner if not inner.is_empty else pool.navigable


def _lane_spacing(robot: Cleaner, overlap: float) -> float:
    """Distance between sweep lines, from the cleaning width."""
    return max(robot.swath_width * (1.0 - overlap), 0.05)


def _sweep_lines(region: Polygon, spacing: float, angle: float) -> list[LineString]:
    """Parallel chords across ``region`` at ``angle``, spaced by ``spacing``.

    Rotate the region flat, cut horizontal lines, rotate the pieces back. A
    concave pool gives several disjoint pieces per line, and keeping them
    separate is the whole point -- joining them would drive the robot straight
    through the wall between two lobes.
    """
    centre = region.centroid
    flat = rotate(region, -angle, origin=centre, use_radians=True)
    minx, miny, maxx, maxy = flat.bounds
    lines: list[LineString] = []
    # Start half a spacing in, so the first and last lanes sit inside the
    # region rather than on its edge.
    y = miny + spacing / 2
    while y <= maxy:
        chord = LineString([(minx - 1.0, y), (maxx + 1.0, y)]).intersection(flat)
        if not chord.is_empty:
            pieces = chord.geoms if isinstance(chord, MultiLineString) else [chord]
            for piece in pieces:
                if isinstance(piece, LineString) and piece.length > 1e-6:
                    lines.append(rotate(piece, angle, origin=centre, use_radians=True))
        y += spacing
    return lines


def _boustrophedon_order(lines: list[LineString], start: FloatArray) -> FloatArray:
    """Walk a set of chords back and forth, nearest end first.

    The chords arrive in sweep order but each has two ends, and taking them
    always left-to-right adds a full lane's width of travel per lane. Choosing
    the near end each time is what makes it boustrophedon rather than a comb.
    """
    waypoints: list[tuple[float, float]] = []
    here = np.asarray(start, dtype=float)
    for line in lines:
        ends = np.array(line.coords)[[0, -1]]
        near = int(np.argmin(np.hypot(*(ends - here).T)))
        ordered = ends[::-1] if near else ends
        waypoints.extend((float(x), float(y)) for x, y in ordered)
        here = ordered[-1]
    return np.asarray(waypoints, dtype=float).reshape(-1, 2)


def _tour(points: FloatArray, start: FloatArray) -> list[int]:
    """Nearest-neighbour order with 2-opt improvement.

    Exact TSP for twenty cells is milliseconds and unnecessary; the difference
    between a greedy tour and an optimal one on a pool decomposition is
    typically under 5%, and 2-opt closes most of that.
    """
    remaining = list(range(len(points)))
    order: list[int] = []
    here = np.asarray(start, dtype=float)
    while remaining:
        nearest = min(remaining, key=lambda i: float(np.hypot(*(points[i] - here))))
        order.append(nearest)
        remaining.remove(nearest)
        here = points[nearest]

    def cost(seq: list[int]) -> float:
        route = np.vstack([start, points[seq]])
        return float(np.hypot(*np.diff(route, axis=0).T).sum())

    best = cost(order)
    improved = True
    while improved and len(order) > 3:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                candidate = order[:i] + order[i:j][::-1] + order[j:]
                value = cost(candidate)
                if value < best - 1e-9:
                    order, best, improved = candidate, value, True
    return order


def _best_angle(cell: Polygon, spacing: float, candidates: int = 12) -> float:
    """The sweep direction that needs the fewest lanes in this cell.

    Hard-coding one direction for every cell of a decomposition throws away
    most of what the decomposition bought: a tall thin strip and a wide flat
    one want their lanes at right angles to each other, and sweeping both the
    same way doubles the turns in one of them.
    """
    best, fewest = 0.0, np.inf
    for angle in np.linspace(0.0, np.pi, candidates, endpoint=False):
        count = len(_sweep_lines(cell, spacing, float(angle)))
        if count and count < fewest:
            best, fewest = float(angle), count
    return best


def _cover_cells(
    cells: list[Polygon], spacing: float, angle: float | None, start: FloatArray
) -> tuple[FloatArray, tuple[Polygon, ...]]:
    """Sweep every cell, visiting them in a good order.

    ``angle=None`` lets each cell choose its own sweep direction, which is
    what a decomposition is for.
    """
    # A sliver narrower than one lane still has to be covered -- dropping it
    # is how trapezoidal decomposition of a smooth boundary returned an empty
    # path, because all three hundred of its cells were slivers. Keep anything
    # a robot could physically enter.
    usable = [cell for cell in cells if cell.area > 1e-6]
    if not usable:
        return np.zeros((0, 2)), ()
    centroids = np.array([[c.centroid.x, c.centroid.y] for c in usable])
    order = _tour(centroids, start)

    waypoints: list[FloatArray] = []
    here = np.asarray(start, dtype=float)
    for index in order:
        cell = usable[index]
        sweep = _best_angle(cell, spacing) if angle is None else angle
        lines = _sweep_lines(cell, spacing, sweep)
        if not lines:
            # Too narrow for even one lane at this spacing. Drive its long
            # axis rather than skipping it: a 20 cm sliver against a wall is
            # exactly the strip a cleaner is judged on.
            lines = [_long_axis(cell)]
        segment = _boustrophedon_order(lines, here)
        if len(segment):
            waypoints.append(segment)
            here = segment[-1]
    if not waypoints:
        return np.zeros((0, 2)), tuple(usable)
    return np.vstack(waypoints), tuple(usable[i] for i in order)


# ----------------------------------------------------------------------
# The sweep family
# ----------------------------------------------------------------------
@PLANNERS.register("boustrophedon")
@dataclass
class Boustrophedon:
    """Back and forth at a fixed angle. The oldest idea in the field.

    Choset & Pignon (1997). Treats the pool as one cell, which is exactly right
    for a convex pool and leaves gaps around obstacles -- which is what the
    cellular decompositions below exist to fix.
    """

    angle: float = 0.0
    """Sweep direction in radians. 0 sweeps along x."""

    overlap: float = 0.10
    name: str = "boustrophedon"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])
        lines = _sweep_lines(region, spacing, self.angle)
        return CoveragePath(
            waypoints=_boustrophedon_order(lines, start),
            planner=self.name,
            cells=(region,),
            notes={"angle": self.angle, "lanes": len(lines), "spacing": spacing},
        )


@PLANNERS.register("sweep_optimal")
@dataclass
class OptimalSweep:
    """Boustrophedon at the angle that needs the fewest lanes.

    Huang (2001) showed the sweep direction is worth optimising: the number of
    lanes is the region's width perpendicular to the sweep divided by the
    spacing, so sweeping along a pool's long axis rather than across it saves a
    turn per lane. Turns are where a tracked machine loses time and traction,
    so this is usually the cheapest real improvement available to a sweeper.

    Searched by brute force over ``samples`` angles rather than solved
    analytically. The analytic answer -- the minimum-width direction of the
    convex hull -- is only optimal for a convex region, and a kidney pool is
    not one.

    What is minimised is ``length + turn_cost * turning``, not lane count.
    Lane count is the right criterion for a convex region, where it *is* the
    turn count; on a concave one a line can break into several chords and the
    two come apart. On the kidney pool the shortest sweep is 161 m with 6480
    degrees of turning and the least-turning one is 189 m with 3600, so the
    weighting is a real choice rather than a formality: ``turn_cost`` is how
    many metres of driving a radian of turning is worth, and 0.35 is roughly
    what a tracked cleaner pays. It picks neither, and lands on 164 m at 4320
    degrees.
    """

    samples: int = 36
    overlap: float = 0.10
    turn_cost: float = 0.35
    name: str = "sweep_optimal"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])

        best_angle, best_cost, best_lines = 0.0, np.inf, []
        for angle in np.linspace(0.0, np.pi, self.samples, endpoint=False):
            lines = _sweep_lines(region, spacing, float(angle))
            if not lines:
                continue
            # Cost the plan the way the robot experiences it: driving and
            # turning, in the same units.
            candidate = CoveragePath(waypoints=_boustrophedon_order(lines, start))
            cost = candidate.length + self.turn_cost * candidate.turns
            if cost < best_cost:
                best_angle, best_cost, best_lines = float(angle), cost, lines

        return CoveragePath(
            waypoints=_boustrophedon_order(best_lines, start),
            planner=self.name,
            cells=(region,),
            notes={
                "angle": best_angle,
                "lanes": len(best_lines),
                "searched": self.samples,
                "cost": best_cost,
            },
        )


# ----------------------------------------------------------------------
# Cellular decompositions
# ----------------------------------------------------------------------
def _slice_at(region: Polygon, xs: FloatArray) -> list[Polygon]:
    """Cut ``region`` into vertical strips at every x in ``xs``."""
    minx, miny, maxx, maxy = region.bounds
    edges = np.unique(np.concatenate([[minx - 1e-6], xs, [maxx + 1e-6]]))
    cells: list[Polygon] = []
    for left, right in pairwise(edges):
        if right - left < 1e-9:
            continue
        strip = region.intersection(
            Polygon([(left, miny - 1), (right, miny - 1), (right, maxy + 1), (left, maxy + 1)])
        )
        if strip.is_empty:
            continue
        pieces = strip.geoms if isinstance(strip, MultiPolygon) else [strip]
        cells.extend(p for p in pieces if isinstance(p, Polygon) and p.area > 1e-9)
    return cells


@PLANNERS.register("trapezoidal")
@dataclass
class Trapezoidal:
    """Cut at every vertex, sweep each cell. The textbook decomposition.

    Simple and wasteful: a smooth boundary has a vertex every few centimetres
    once it has been discretised, so a kidney pool decomposes into hundreds of
    slivers. That is not a bug in the implementation, it is the reason
    boustrophedon cellular decomposition was invented, and running both is the
    clearest way to see why.
    """

    overlap: float = 0.10
    min_width: float = 0.0
    """Merge strips narrower than this, in metres. 0 is the textbook version."""

    name: str = "trapezoidal"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])

        xs = np.array(sorted({round(float(x), 6) for x, _ in _vertices(region)}))
        if self.min_width > 0 and xs.size:
            keep = [xs[0]]
            for value in xs[1:]:
                if value - keep[-1] >= self.min_width:
                    keep.append(value)
            xs = np.asarray(keep)

        cells = _slice_at(region, xs)
        waypoints, ordered = _cover_cells(cells, spacing, None, start)
        return CoveragePath(
            waypoints=waypoints,
            planner=self.name,
            cells=ordered,
            notes={"cells": len(ordered), "cuts": int(xs.size)},
        )


def _long_axis(cell: Polygon) -> LineString:
    """The longest chord through a cell's centroid, for cells too thin to sweep."""
    minx, miny, maxx, maxy = cell.bounds
    centre = cell.centroid
    if maxx - minx >= maxy - miny:
        chord = LineString([(minx - 1.0, centre.y), (maxx + 1.0, centre.y)])
    else:
        chord = LineString([(centre.x, miny - 1.0), (centre.x, maxy + 1.0)])
    clipped = chord.intersection(cell)
    if isinstance(clipped, MultiLineString):
        clipped = max(clipped.geoms, key=lambda g: g.length)
    if isinstance(clipped, LineString) and clipped.length > 1e-9:
        return clipped
    return LineString([(centre.x, centre.y), (centre.x, centre.y)])


def _vertices(region: Polygon) -> FloatArray:
    rings = [region.exterior, *region.interiors]
    return np.vstack([np.asarray(ring.coords) for ring in rings])


def _critical_x(region: Polygon, tolerance: float) -> FloatArray:
    """Where the number of disjoint slices changes as a line sweeps across.

    The boustrophedon decomposition's whole idea, and the difference between it
    and the trapezoidal one. A vertex on a smooth curve does not change
    connectivity and needs no cut; a vertex where the region splits in two, or
    two lobes merge, does. Detected by counting slices either side of each
    candidate rather than by classifying vertex types, which is far shorter and
    works on the discretised boundaries this library actually has.
    """
    minx, _, maxx, _ = region.bounds
    xs = np.unique(np.round(_vertices(region)[:, 0], 6))
    critical: list[float] = []
    for x in xs:
        if x <= minx + tolerance or x >= maxx - tolerance:
            continue
        before = _slice_count(region, x - tolerance)
        after = _slice_count(region, x + tolerance)
        if before != after:
            critical.append(float(x))
    return np.asarray(critical, dtype=float)


def _slice_count(region: Polygon, x: float) -> int:
    _, miny, _, maxy = region.bounds
    chord = LineString([(x, miny - 1.0), (x, maxy + 1.0)]).intersection(region)
    if chord.is_empty:
        return 0
    return len(chord.geoms) if isinstance(chord, MultiLineString) else 1


@PLANNERS.register("boustrophedon_cells")
@dataclass
class BoustrophedonCells:
    """Cut only where the connectivity changes. Choset & Pignon (1997).

    The refinement that makes cellular decomposition practical: cut at the
    *critical points* where a sweeping line splits or merges, not at every
    vertex. On a pool with one obstacle that is two cuts instead of two hundred,
    and the resulting cells are each simply connected, so a back-and-forth
    sweep inside one is complete by construction.
    """

    overlap: float = 0.10
    tolerance: float = 0.05
    """How far either side of a vertex to count slices, in metres."""

    name: str = "boustrophedon_cells"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])

        cuts = _critical_x(region, self.tolerance)
        cells = _slice_at(region, cuts)
        waypoints, ordered = _cover_cells(cells, spacing, None, start)
        return CoveragePath(
            waypoints=waypoints,
            planner=self.name,
            cells=ordered,
            notes={"cells": len(ordered), "critical_points": int(cuts.size)},
        )


@PLANNERS.register("morse")
@dataclass
class Morse:
    """Boustrophedon decomposition with a choice of sweep function.

    Acar & Choset (2002) generalised critical points from a straight sweeping
    line to the level sets of any Morse function, which is what lets the
    decomposition handle curved obstacles rather than only polygonal ones.

    ``function`` picks the level sets: ``"linear"`` reproduces the
    boustrophedon decomposition exactly, and ``"radial"`` sweeps circles
    outward from the pool's centroid, which decomposes a round pool into one
    cell where a linear sweep gives several.
    """

    function: str = "radial"
    overlap: float = 0.10
    tolerance: float = 0.05
    name: str = "morse"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        if self.function == "linear":
            return BoustrophedonCells(overlap=self.overlap, tolerance=self.tolerance).plan(
                pool, robot
            )
        if self.function != "radial":
            raise ValueError(f"function must be 'linear' or 'radial', got {self.function!r}")

        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])
        centre = Point(region.centroid)

        cuts = _critical_radii(region, centre, self.tolerance)
        cells: list[Polygon] = []
        previous = 0.0
        for radius in [*cuts, _far_radius(region, centre)]:
            band = centre.buffer(radius).difference(centre.buffer(previous))
            previous = radius
            piece = region.intersection(band)
            parts = piece.geoms if isinstance(piece, MultiPolygon) else [piece]
            cells.extend(p for p in parts if isinstance(p, Polygon) and p.area > 1e-9)

        waypoints, ordered = _cover_cells(cells, spacing, None, start)
        return CoveragePath(
            waypoints=waypoints,
            planner=self.name,
            cells=ordered,
            notes={
                "function": self.function,
                "cells": len(ordered),
                "critical_radii": len(cuts),
            },
        )


def _far_radius(region: Polygon, centre: Point) -> float:
    return max(centre.distance(Point(p)) for p in _vertices(region)) + 1e-6


def _critical_radii(region: Polygon, centre: Point, step: float) -> list[float]:
    """Radii at which the circular level set changes component count.

    This is what makes it a Morse decomposition rather than contour
    following. The level sets of the radial function are circles; where a
    circle's intersection with the pool splits into two arcs or merges back
    into one, the topology of the sweep has changed and a cell boundary
    belongs there. Between two such radii the region is one connected band
    and can be swept in one go.

    An earlier version cut a band at *every* spacing and drove each one, which
    is a perfectly good coverage path and is also exactly what ``contour``
    does. Two names for one algorithm is worse than either.
    """
    far = _far_radius(region, centre)
    radii = np.arange(step, far, max(step, 1e-3))
    cuts: list[float] = []
    previous = -1
    for radius in radii:
        level = region.intersection(centre.buffer(float(radius)).exterior)
        count = len(getattr(level, "geoms", [level])) if not level.is_empty else 0
        if previous >= 0 and count != previous:
            cuts.append(float(radius))
        previous = count
    return cuts


def _ring_order(rings: list[Polygon], start: FloatArray) -> FloatArray:
    """Walk each annulus around its own centreline, innermost first."""
    waypoints: list[tuple[float, float]] = []
    for ring in rings:
        coords = np.asarray(ring.exterior.coords)
        if len(coords) < 3:
            continue
        # Enter each ring at the point nearest where the last one ended.
        here = np.asarray(waypoints[-1] if waypoints else start, dtype=float)
        offset = int(np.argmin(np.hypot(*(coords - here).T)))
        rolled = np.roll(coords[:-1], -offset, axis=0)
        waypoints.extend((float(x), float(y)) for x, y in rolled)
    return np.asarray(waypoints, dtype=float).reshape(-1, 2)


# ----------------------------------------------------------------------
# Contour-parallel
# ----------------------------------------------------------------------
@PLANNERS.register("contour")
@dataclass
class Contour:
    """Follow the wall, then a swath in from it, and so on inward.

    The offsetting strategy from CNC machining, and the one most real pool
    cleaners visibly do -- a perimeter lap, then another inside it. It has two
    genuine advantages over a sweep: no lane is ever partial, and the wall,
    which is where the dirt collects, is covered first.

    Its weakness is the middle. Offsetting a concave shape inward eventually
    splits it into disconnected islands, and the robot has to cross covered
    ground to reach each one. The count is reported in ``notes["islands"]``.
    """

    overlap: float = 0.10
    outward: bool = False
    """Start at the centre and work out, rather than at the wall and work in."""

    name: str = "contour"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        region = _workspace(pool, robot)
        spacing = _lane_spacing(robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])

        loops: list[FloatArray] = []
        islands = 0
        shrunk: Any = region
        depth = 0
        while not shrunk.is_empty and depth < 200:
            parts = shrunk.geoms if isinstance(shrunk, MultiPolygon) else [shrunk]
            solid = [p for p in parts if isinstance(p, Polygon) and p.area > spacing * spacing]
            if not solid:
                break
            islands = max(islands, len(solid))
            for piece in solid:
                # A buffered kidney has hundreds of vertices per ring, and the
                # unsimplified version of this planner emitted eleven thousand
                # waypoints for one pool. Simplify to a tenth of a lane, which
                # is far below what the follower can track anyway.
                loops.append(np.asarray(piece.exterior.simplify(spacing * 0.1).coords))
            shrunk = shrunk.buffer(-spacing)
            depth += 1

        if self.outward:
            loops.reverse()

        waypoints: list[tuple[float, float]] = []
        here = start
        for loop in loops:
            offset = int(np.argmin(np.hypot(*(loop - here).T)))
            rolled = np.roll(loop[:-1], -offset, axis=0)
            waypoints.extend((float(x), float(y)) for x, y in rolled)
            here = rolled[-1]

        return CoveragePath(
            waypoints=np.asarray(waypoints, dtype=float).reshape(-1, 2),
            planner=self.name,
            cells=(region,),
            notes={"loops": len(loops), "islands": islands, "outward": self.outward},
        )


# ----------------------------------------------------------------------
# Grid methods
# ----------------------------------------------------------------------
def _grid_of(pool: Pool, robot: Cleaner, overlap: float) -> tuple[NDArray[np.bool_], Any, float]:
    """A coarse occupancy grid at the swath's resolution."""
    cell = _lane_spacing(robot, overlap)
    grid = pool.grid(cell)
    free = pool.navigable_mask(cell)
    # Clear the border ring: a cell whose centre is navigable can still be
    # unreachable for a robot with width.
    inner = _workspace(pool, robot)
    xs, ys = grid.cell_centers()
    from shapely import contains_xy

    reachable = np.asarray(contains_xy(inner, xs, ys)).reshape(free.shape)
    return free & reachable, grid, cell


@PLANNERS.register("wavefront")
@dataclass
class Wavefront:
    """Distance transform from a goal, then descend it. Zelinsky et al. (1993).

    Label every cell with its grid distance from a chosen goal, then walk from
    the *furthest* cell down the gradient, preferring unvisited neighbours. The
    result covers everything and finishes at the goal, which is exactly what a
    cleaner wanting to end at its dock needs.

    Its character is different from a sweep: it produces a path with many short
    straight runs rather than a few long ones, so it turns much more often. The
    comparison harness makes that cost visible.
    """

    overlap: float = 0.10
    name: str = "wavefront"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        free, grid, cell = _grid_of(pool, robot, self.overlap)
        start = np.array(pool.start_pose(clearance=robot.radius)[:2])
        goal = _nearest_cell(free, grid, cell, start)
        distance = _bfs(free, goal)

        order = _descend(free, distance)
        waypoints = np.array(
            [[grid.minx + (col + 0.5) * cell, grid.miny + (row + 0.5) * cell] for row, col in order]
        )
        return CoveragePath(
            waypoints=waypoints.reshape(-1, 2),
            planner=self.name,
            cells=(_workspace(pool, robot),),
            notes={"cells": int(free.sum()), "resolution": cell},
        )


def _nearest_cell(
    free: NDArray[np.bool_], grid: Any, cell: float, point: FloatArray
) -> tuple[int, int]:
    rows, cols = np.nonzero(free)
    xs = grid.minx + (cols + 0.5) * cell
    ys = grid.miny + (rows + 0.5) * cell
    index = int(np.argmin(np.hypot(xs - point[0], ys - point[1])))
    return int(rows[index]), int(cols[index])


def _bfs(free: NDArray[np.bool_], goal: tuple[int, int]) -> NDArray[np.int32]:
    """Grid distance from ``goal``, 4-connected. -1 where unreachable."""
    distance = np.full(free.shape, -1, dtype=np.int32)
    distance[goal] = 0
    frontier = [goal]
    while frontier:
        nxt = []
        for row, col in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                inside = 0 <= r < free.shape[0] and 0 <= c < free.shape[1]
                if inside and free[r, c] and distance[r, c] < 0:
                    distance[r, c] = distance[row, col] + 1
                    nxt.append((r, c))
        frontier = nxt
    return distance


def _descend(free: NDArray[np.bool_], distance: NDArray[np.int32]) -> list[tuple[int, int]]:
    """From the furthest cell, always step to the unvisited neighbour with the
    largest distance, jumping to the nearest unvisited cell when stuck."""
    reachable = free & (distance >= 0)
    if not reachable.any():
        return []
    visited = np.zeros_like(reachable)
    flat = int(np.argmax(np.where(reachable, distance, -1)))
    here = (flat // free.shape[1], flat % free.shape[1])
    order = [here]
    visited[here] = True

    while visited[reachable].sum() < reachable.sum():
        best = None
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = here[0] + dr, here[1] + dc
            inside = 0 <= r < free.shape[0] and 0 <= c < free.shape[1]
            free_here = inside and reachable[r, c] and not visited[r, c]
            if free_here and (best is None or distance[r, c] > distance[best]):
                best = (r, c)
        if best is None:
            rows, cols = np.nonzero(reachable & ~visited)
            if not rows.size:
                break
            index = int(np.argmin(np.abs(rows - here[0]) + np.abs(cols - here[1])))
            best = (int(rows[index]), int(cols[index]))
        here = best
        visited[here] = True
        order.append(here)
    return order


@PLANNERS.register("spanning_tree")
@dataclass
class SpanningTree:
    """Circumnavigate a spanning tree of double-width cells.

    Gabriely & Rimon (2001). Build a grid of cells twice the tool width, find a
    spanning tree over it, and walk around the tree keeping it on your left.
    The path visits every quarter-cell exactly once, so coverage is complete
    and *non-overlapping* -- the only planner here with that guarantee.

    The catch is the doubling. Cells are twice the swath, so a pool whose width
    is not a multiple of that loses a strip at the edge, and the coverage
    number reflects it. That is the algorithm, not the implementation.
    """

    overlap: float = 0.10
    name: str = "spanning_tree"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        spacing = _lane_spacing(robot, self.overlap)
        mega = 2.0 * spacing
        region = _workspace(pool, robot)
        grid = pool.grid(mega)
        xs, ys = grid.cell_centers()
        from shapely import contains_xy

        free = np.asarray(contains_xy(region, xs, ys)).reshape(grid.nrows, grid.ncols)
        if not free.any():
            return CoveragePath(np.zeros((0, 2)), planner=self.name, notes={"cells": 0})

        start = np.array(pool.start_pose(clearance=robot.radius)[:2])
        root = _nearest_cell(free, grid, mega, start)
        parents = _spanning_tree(free, root)
        walk = _tree_walk(root, parents)

        waypoints = _hug_tree(walk, grid, mega)
        return CoveragePath(
            waypoints=waypoints,
            planner=self.name,
            cells=(region,),
            notes={"cells": int(free.sum()), "mega_cell": mega, "steps": len(walk)},
        )


def _spanning_tree(
    free: NDArray[np.bool_], root: tuple[int, int]
) -> dict[tuple[int, int], tuple[int, int] | None]:
    """Depth-first spanning tree over the free cells."""
    parents: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    stack = [root]
    while stack:
        row, col = stack.pop()
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbour = (row + dr, col + dc)
            r, c = neighbour
            inside = 0 <= r < free.shape[0] and 0 <= c < free.shape[1]
            if inside and free[r, c] and neighbour not in parents:
                parents[neighbour] = (row, col)
                stack.append(neighbour)
    return parents


def _tree_walk(
    root: tuple[int, int], parents: dict[tuple[int, int], tuple[int, int] | None]
) -> list[tuple[int, int]]:
    """A closed walk that traverses every tree edge exactly twice.

    Not a depth-first *order* -- an Euler tour of the doubled tree. The
    distinction is the whole algorithm: visiting cells in DFS order and
    teleporting between them produces a path that jumps across the pool, while
    walking out and back along each branch produces one continuous route whose
    offset is the covering spiral.
    """
    children: dict[tuple[int, int], list[tuple[int, int]]] = {node: [] for node in parents}
    for node, parent in parents.items():
        if parent is not None:
            children[parent].append(node)

    walk: list[tuple[int, int]] = []

    def descend(node: tuple[int, int]) -> None:
        walk.append(node)
        for child in children[node]:
            descend(child)
            walk.append(node)

    import sys

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(parents) + 1000))
    try:
        descend(root)
    finally:
        sys.setrecursionlimit(limit)
    return walk


def _hug_tree(walk: list[tuple[int, int]], grid: Any, mega: float) -> FloatArray:
    """Trace the boundary of the thickened tree, which is the covering path.

    Gabriely and Rimon's construction says to follow the spanning tree keeping
    it always on the same side, at a quarter-cell offset. The shortest exact
    way to build that curve is to *thicken* the tree by a quarter cell and take
    the outline of the result: the outline of a thickened tree is precisely the
    closed walk that goes out along one side of every branch and back along the
    other, covering both halves of each cell and no quarter twice.

    Two earlier attempts got the character wrong and are worth recording. The
    first emitted the four corners of each cell in a fixed order, which covered
    everything with forty-five thousand degrees of turning because each cell
    was its own little square rather than part of a route. The second offset
    each tree edge to its right, which is the correct rule, but produced a
    disjoint segment per edge -- the offset flips as the direction changes, so
    consecutive segments did not join.
    """
    if not walk:
        return np.zeros((0, 2))

    def centre(cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (grid.minx + (col + 0.5) * mega, grid.miny + (row + 0.5) * mega)

    quarter = mega / 4.0
    edges = [LineString([centre(a), centre(b)]) for a, b in pairwise(walk) if a != b]
    if not edges:
        # A single-cell tree: the path is one lap of that cell.
        x, y = centre(walk[0])
        return np.array(
            [
                (x - quarter, y - quarter),
                (x + quarter, y - quarter),
                (x + quarter, y + quarter),
                (x - quarter, y + quarter),
                (x - quarter, y - quarter),
            ]
        )

    # Square caps and mitred joins, so the corridor is a union of rectangles
    # aligned to the grid rather than a set of rounded sausages -- a rounded
    # offset would round off the corners of every cell and leave the corner
    # quarters uncovered.
    corridor = unary_union(edges).buffer(quarter, cap_style=3, join_style=2)
    if isinstance(corridor, MultiPolygon):
        corridor = max(corridor.geoms, key=lambda g: g.area)
    outline = np.asarray(corridor.exterior.simplify(quarter * 0.05).coords)

    # Enter the loop at the point nearest the root, so the robot does not begin
    # by driving to the far side of the pool.
    root = np.asarray(centre(walk[0]))
    offset = int(np.argmin(np.hypot(*(outline[:-1] - root).T)))
    rolled = np.roll(outline[:-1], -offset, axis=0)
    return np.vstack([rolled, rolled[:1]])
