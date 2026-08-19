"""Dividing a pool between robots.

Most multi-robot coverage in the literature is one idea: **cut the area into
as many pieces as there are robots, then run a single-robot planner in each**.
It is not a dodge. Coverage of a region is a solved problem, the partition is
where the multi-robot content lives, and a good partition turns N robots into
something close to N times one robot -- which is the most anyone has managed.

Four ways to cut, each failing differently:

====================  ====================================================
``voronoi``           nearest robot wins, by straight-line distance
``geodesic``          nearest robot wins, by distance *through the water*
``strips``            equal-area bands across the pool's long axis
``darp``              iterate the assignment until the shares are equal
``forest``            split a spanning tree into balanced subtrees
====================  ====================================================

The failures are the point. ``voronoi`` is instant and hands a robot the far
side of a wall it has to drive around. ``geodesic`` fixes that and still gives
one robot twice the floor of another, because being nearest to a lot of pool
is not the same as deserving it. ``strips`` are equal by construction and
ignore where the robots actually are. ``darp`` is the one that gets both, and
pays for it with an iteration that does not always converge. ``forest``
guarantees each share is connected, because it cuts a tree rather than a map.

Every territory becomes a small :class:`~zimablue.pool.Pool`, so all eight
offline planners work inside one unchanged::

    from zimablue.planners.partition import partitioned

    zb.Fleet(pool="kidney", robots=3,
             controllers=partitioned("darp", "sweep_optimal")).run(minutes=20)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from zimablue.planners.base import PLANNERS, CoveragePath, PathFollower, make_planner
from zimablue.registry import Registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.pool import Pool
    from zimablue.robot import Cleaner

__all__ = [
    "PARTITIONS",
    "Partition",
    "Partitioner",
    "Territory",
    "make_partition",
    "partitioned",
]

BoolArray = NDArray[np.bool_]
Cell = tuple[int, int]
NEIGHBOURS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class Territory:
    """One robot's share of the pool."""

    index: int
    mask: BoolArray
    """Which cells of the pool raster belong to this robot."""

    region: Polygon
    seed: tuple[float, float]
    """Where the robot that owns this was standing when the cut was made."""

    @property
    def cells(self) -> int:
        return int(self.mask.sum())

    @property
    def area(self) -> float:
        return float(self.region.area)

    @property
    def connected(self) -> bool:
        """Whether the share is one piece. A robot handed two islands has to
        drive between them, and that travel is pure loss."""
        return not isinstance(self.region, MultiPolygon)

    def as_pool(self, parent: Pool) -> Pool:
        """The share, as a pool a single-robot planner can be pointed at.

        Depth, material and water are inherited; the features are not, because
        they have already been cut out of the mask this was built from and
        subtracting them twice would put holes in the wrong places.
        """
        from zimablue.pool import Pool as PoolType

        return PoolType(
            boundary=self.region,
            depth=parent.depth_model,
            name=f"{parent.name}#{self.index}",
            material=parent.material,
            water=parent.water,
        )


@dataclass
class Partition:
    """A whole division of the pool, and how good it turned out."""

    territories: list[Territory]
    labels: NDArray[np.int16]
    """Owner of each cell, ``-1`` for cells nobody can reach."""

    cell: float
    method: str
    notes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.territories)

    @property
    def fairness(self) -> float:
        """Smallest share over largest. 1.0 is a perfectly even cut."""
        sizes = [t.cells for t in self.territories]
        if not sizes:
            return 0.0
        largest = max(sizes)
        return min(sizes) / largest if largest else 0.0

    @property
    def connected(self) -> bool:
        return all(t.connected for t in self.territories)

    def describe(self) -> str:
        sizes = "/".join(str(t.cells) for t in self.territories)
        return (
            f"{self.method}: {len(self)} territories, {sizes} cells, "
            f"fairness {self.fairness:.2f}, "
            f"{'all connected' if self.connected else 'SOME SPLIT'}"
        )


@runtime_checkable
class Partitioner(Protocol):
    """Cuts a pool into one region per robot."""

    name: str

    def divide(
        self,
        pool: Pool,
        robots: list[Cleaner],
        poses: list[tuple[float, float, float]],
        *,
        cell: float,
    ) -> Partition: ...


PARTITIONS: Registry[Partitioner] = Registry("partition")


def make_partition(partition: Partitioner | str, **kwargs: Any) -> Partitioner:
    if isinstance(partition, str):
        return PARTITIONS.create(partition, **kwargs)
    return partition


# ----------------------------------------------------------------------
# Shared machinery
# ----------------------------------------------------------------------
def _workspace_grid(pool: Pool, robot: Cleaner, cell: float):
    """The navigable cells, and the raster they sit on.

    Navigable, not navigable-minus-the-robot's-radius. The territories are
    handed to single-robot planners, and every one of those insets by the
    robot radius itself -- so cutting the pool down first insets it twice and
    leaves a two-radius ring along the wall belonging to nobody. Measured on
    the kidney with three robots, that ring was 29 points of coverage: the
    fleet swept its territories perfectly and still missed a third of the
    pool.
    """
    from shapely import contains_xy

    grid = pool.grid(cell)
    xs, ys = grid.cell_centers()
    free = np.asarray(contains_xy(pool.navigable, xs, ys)).reshape(grid.nrows, grid.ncols)
    return free, grid


def _cell_of(grid: Any, cell_size: float, point: tuple[float, float]) -> Cell:
    row = int(np.clip((point[1] - grid.miny) / cell_size, 0, grid.nrows - 1))
    col = int(np.clip((point[0] - grid.minx) / cell_size, 0, grid.ncols - 1))
    return (row, col)


def _nearest_free(free: BoolArray, start: Cell) -> Cell:
    """The free cell closest to ``start``, which may itself be blocked."""
    if free[start]:
        return start
    rows, cols = np.nonzero(free)
    if not len(rows):
        raise ValueError("no free cells: the pool is smaller than the robot")
    best = int(np.argmin((rows - start[0]) ** 2 + (cols - start[1]) ** 2))
    return (int(rows[best]), int(cols[best]))


def _geodesic(free: BoolArray, source: Cell) -> NDArray[np.float64]:
    """Distance from ``source`` to every free cell, *through* the free space.

    Breadth-first in cells, so a wall between two points costs the way round
    it. Straight-line distance is what makes a Voronoi partition hand a robot
    the far lobe of a kidney pool.
    """
    distance = np.full(free.shape, np.inf)
    distance[source] = 0.0
    queue = deque([source])
    while queue:
        row, col = queue.popleft()
        for dr, dc in NEIGHBOURS:
            r, c = row + dr, col + dc
            inside = 0 <= r < free.shape[0] and 0 <= c < free.shape[1]
            if inside and free[r, c] and not np.isfinite(distance[r, c]):
                distance[r, c] = distance[row, col] + 1.0
                queue.append((r, c))
    return distance


def _component(mask: BoolArray, seed: Cell) -> BoolArray:
    """The connected piece of ``mask`` containing ``seed``."""
    out = np.zeros_like(mask)
    if not mask[seed]:
        return out
    out[seed] = True
    queue = deque([seed])
    while queue:
        row, col = queue.popleft()
        for dr, dc in NEIGHBOURS:
            r, c = row + dr, col + dc
            inside = 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]
            if inside and mask[r, c] and not out[r, c]:
                out[r, c] = True
                queue.append((r, c))
    return out


def _polygonise(mask: BoolArray, grid: Any, cell: float) -> Polygon:
    """Turn a set of cells into a polygon a planner can sweep.

    Squares unioned and then simplified by half a cell. The simplification is
    not cosmetic: a raw union of 400 squares has 1600 collinear vertices, and
    the trapezoidal decomposition would cut a cell at every one of them.
    """
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return Polygon()
    boxes = [
        shapely_box(
            grid.minx + c * cell,
            grid.miny + r * cell,
            grid.minx + (c + 1) * cell,
            grid.miny + (r + 1) * cell,
        )
        for r, c in zip(rows, cols, strict=True)
    ]
    merged = unary_union(boxes).buffer(cell * 0.02).buffer(-cell * 0.02)
    if isinstance(merged, MultiPolygon):
        # Keep every piece: a split territory is a real outcome and hiding it
        # by dropping the small islands would flatter the partitioner.
        merged = MultiPolygon([p.simplify(cell * 0.5) for p in merged.geoms if p.area > 1e-9])
        return merged if len(merged.geoms) > 1 else merged.geoms[0]
    return merged.simplify(cell * 0.5)


def _assemble(
    labels: NDArray[np.int16],
    poses: list[tuple[float, float, float]],
    grid: Any,
    cell: float,
    method: str,
    notes: dict[str, Any],
) -> Partition:
    territories = []
    for index, pose in enumerate(poses):
        mask = labels == index
        territories.append(
            Territory(
                index=index,
                mask=mask,
                region=_polygonise(mask, grid, cell),
                seed=(pose[0], pose[1]),
            )
        )
    return Partition(territories=territories, labels=labels, cell=cell, method=method, notes=notes)


def _seed_cells(free: BoolArray, grid: Any, cell: float, poses) -> list[Cell]:
    return [_nearest_free(free, _cell_of(grid, cell, (p[0], p[1]))) for p in poses]


# ----------------------------------------------------------------------
# The partitioners
# ----------------------------------------------------------------------
@PARTITIONS.register("voronoi")
@dataclass
class Voronoi:
    """Nearest robot wins, measured in a straight line.

    The one-liner, and worth having precisely because it is the one everyone
    reaches for first. It takes no iterations and produces shares that are
    neither equal nor necessarily reachable: on a kidney pool the robot in one
    lobe is assigned the tip of the other, because the crow flies across the
    concavity and the robot cannot.
    """

    cell: float = 0.5
    name: str = "voronoi"

    def divide(self, pool, robots, poses, *, cell=None) -> Partition:
        size = cell or self.cell
        free, grid = _workspace_grid(pool, robots[0], size)
        xs, ys = grid.cell_centers()
        xs = np.asarray(xs).reshape(free.shape)
        ys = np.asarray(ys).reshape(free.shape)
        stack = np.stack([np.hypot(xs - p[0], ys - p[1]) for p in poses])
        labels = np.where(free, stack.argmin(axis=0), -1).astype(np.int16)
        return _assemble(labels, poses, grid, size, self.name, {"iterations": 0})


@PARTITIONS.register("geodesic")
@dataclass
class Geodesic:
    """Nearest robot wins, measured through the water.

    One breadth-first search per robot instead of one distance formula, and it
    removes the whole class of Voronoi mistake where a territory is on the
    wrong side of a wall. Shares are still unequal -- a robot parked in a wide
    part of the pool gets a wide territory -- which is what DARP goes on to
    fix.
    """

    cell: float = 0.5
    name: str = "geodesic"

    def divide(self, pool, robots, poses, *, cell=None) -> Partition:
        size = cell or self.cell
        free, grid = _workspace_grid(pool, robots[0], size)
        seeds = _seed_cells(free, grid, size, poses)
        stack = np.stack([_geodesic(free, seed) for seed in seeds])
        reachable = free & np.isfinite(stack).any(axis=0)
        labels = np.where(reachable, np.nan_to_num(stack, posinf=1e18).argmin(axis=0), -1)
        return _assemble(labels.astype(np.int16), poses, grid, size, self.name, {"iterations": 0})


@PARTITIONS.register("strips")
@dataclass
class Strips:
    """Equal-area bands across the pool's long axis.

    The oldest answer, and the one a farmer would give: divide the field into
    N strips and give everyone one. Equal by construction, connected by
    construction on any convex pool, and completely indifferent to where the
    robots are standing -- so the strips are then handed out nearest-first,
    which recovers some of what the cut ignored.
    """

    cell: float = 0.5
    name: str = "strips"

    def divide(self, pool, robots, poses, *, cell=None) -> Partition:
        size = cell or self.cell
        free, grid = _workspace_grid(pool, robots[0], size)
        xs, ys = grid.cell_centers()
        xs = np.asarray(xs).reshape(free.shape)
        ys = np.asarray(ys).reshape(free.shape)

        minx, miny, maxx, maxy = pool.boundary.bounds
        along = xs if (maxx - minx) >= (maxy - miny) else ys
        order = np.argsort(along[free])
        count = len(poses)
        bands = np.zeros(int(free.sum()), dtype=np.int16)
        edges = np.linspace(0, len(order), count + 1).astype(int)
        for band, (lo, hi) in enumerate(pairwise(edges)):
            bands[order[lo:hi]] = band

        strip_labels = np.full(free.shape, -1, dtype=np.int16)
        strip_labels[free] = bands
        assignment = _match_strips(strip_labels, grid, size, poses)
        labels = np.where(free, assignment[np.clip(strip_labels, 0, None)], -1).astype(np.int16)
        axis = "x" if along is xs else "y"
        return _assemble(labels, poses, grid, size, self.name, {"axis": axis})


def _match_strips(strip_labels, grid, cell, poses) -> NDArray[np.int16]:
    """Give each band to the nearest unclaimed robot.

    Greedy on the smallest robot-to-band distance rather than an assignment
    solve. With three or four robots and the same number of bands the two
    agree; the point of the step is to stop robot 0 being sent to the far end
    of the pool because it happened to be listed first.
    """
    count = len(poses)
    centres = []
    for band in range(count):
        rows, cols = np.nonzero(strip_labels == band)
        centres.append(
            (grid.minx + (cols.mean() + 0.5) * cell, grid.miny + (rows.mean() + 0.5) * cell)
            if len(rows)
            else (grid.minx, grid.miny)
        )
    cost = np.array(
        [[np.hypot(c[0] - p[0], c[1] - p[1]) for p in poses] for c in centres], dtype=float
    )
    assignment = np.full(count, -1, dtype=np.int16)
    for _ in range(count):
        flat = np.unravel_index(int(np.argmin(cost)), cost.shape)
        band, robot = int(flat[0]), int(flat[1])
        assignment[band] = robot
        cost[band, :] = np.inf
        cost[:, robot] = np.inf
    return assignment


@PARTITIONS.register("darp")
@dataclass
class DARP:
    """Divide Areas by Robot Positions (Kapoutsis et al., 2017).

    The one that gets equal *and* reachable, by iterating. Each robot has a
    multiplier on its distance field; assign every cell to whoever's weighted
    distance is smallest; measure how far each share is from an equal one;
    nudge the multipliers and go round again. A robot with too much floor gets
    a larger multiplier and loses its outermost cells to a neighbour.

    Connectivity gets its own correction. An assignment by weighted distance
    can leave a robot with an island across the pool, so after each round
    everything outside the piece containing the robot is penalised, which
    pushes it to a neighbour on the next pass.

    This implements the mechanism -- per-robot multipliers driven by area
    error, plus a connectivity penalty -- rather than the paper's exact update
    law. It reports what it achieved in ``notes``, including whether it
    converged, because on an awkward pool it sometimes does not and a
    partition that quietly stopped early is worse than one that says so.
    """

    cell: float = 0.5
    iterations: int = 150
    tolerance: float = 0.03
    gain: float = 0.12
    connectivity_penalty: float = 1.6
    name: str = "darp"

    def divide(self, pool, robots, poses, *, cell=None) -> Partition:
        size = cell or self.cell
        free, grid = _workspace_grid(pool, robots[0], size)
        seeds = _seed_cells(free, grid, size, poses)
        count = len(poses)

        distance = np.stack([_geodesic(free, seed) for seed in seeds])
        reachable = free & np.isfinite(distance).any(axis=0)
        distance = np.where(np.isfinite(distance), distance, 1e12)
        target = float(reachable.sum()) / count

        multiplier = np.ones(count)
        penalty = np.ones_like(distance)
        labels = np.full(free.shape, -1, dtype=np.int16)
        converged = False
        used = 0

        for round_number in range(1, self.iterations + 1):
            used = round_number
            weighted = distance * multiplier[:, None, None] * penalty
            labels = np.where(reachable, weighted.argmin(axis=0), -1).astype(np.int16)

            sizes = np.array([(labels == i).sum() for i in range(count)], dtype=float)
            error = np.abs(sizes - target).max() / max(target, 1.0)

            broken = 0
            for index in range(count):
                mask = labels == index
                keep = _component(mask, seeds[index])
                orphans = mask & ~keep
                if orphans.any():
                    broken += 1
                    penalty[index][orphans] *= self.connectivity_penalty
            if error <= self.tolerance and broken == 0:
                converged = True
                break
            # Too much floor means a bigger multiplier and a smaller share.
            multiplier *= 1.0 + self.gain * (sizes - target) / max(target, 1.0)
            multiplier = np.clip(multiplier, 1e-3, 1e3)

        notes = {
            "iterations": used,
            "converged": converged,
            "target_cells": target,
            "gain": self.gain,
        }
        return _assemble(labels, poses, grid, size, self.name, notes)


@PARTITIONS.register("forest")
@dataclass
class Forest:
    """Cut a spanning tree into balanced subtrees (Zheng et al., 2005).

    Multi-robot Forest Coverage. Rather than dividing the *area* and hoping
    each share is connected, divide a spanning tree of the area, which cannot
    produce a disconnected share: every subtree is connected because it is a
    subtree.

    The cut is greedy on subtree weight -- repeatedly detach the subtree whose
    size is nearest to what one robot should get. That is the standard
    heuristic and it is not optimal; the paper's contribution is the
    approximation bound on a rooted-tree variant, which this does not claim.
    """

    cell: float = 0.5
    name: str = "forest"

    def divide(self, pool, robots, poses, *, cell=None) -> Partition:
        size = cell or self.cell
        free, grid = _workspace_grid(pool, robots[0], size)
        seeds = _seed_cells(free, grid, size, poses)
        count = len(poses)

        parents = _spanning_forest(free, seeds[0])
        pieces = _cut_tree(parents, count)

        # Hand each piece to the nearest robot that has not been given one.
        centres = []
        for piece in pieces:
            rows = np.array([c[0] for c in piece], dtype=float)
            cols = np.array([c[1] for c in piece], dtype=float)
            centres.append(
                (grid.minx + (cols.mean() + 0.5) * size, grid.miny + (rows.mean() + 0.5) * size)
            )
        cost = np.array(
            [[np.hypot(c[0] - p[0], c[1] - p[1]) for p in poses] for c in centres], dtype=float
        )
        labels = np.full(free.shape, -1, dtype=np.int16)
        for _ in range(min(len(pieces), count)):
            piece_index, robot = np.unravel_index(int(np.argmin(cost)), cost.shape)
            for row, col in pieces[piece_index]:
                labels[row, col] = robot
            cost[piece_index, :] = np.inf
            cost[:, robot] = np.inf

        return _assemble(
            labels, poses, grid, size, self.name, {"pieces": len(pieces), "iterations": 0}
        )


def _spanning_forest(free: BoolArray, root: Cell) -> dict[Cell, Cell | None]:
    """Breadth-first spanning tree over the free cells.

    Breadth-first rather than depth-first: a DFS tree of a grid is a long
    snake, and cutting a snake into k pieces gives k long thin territories.
    """
    parents: dict[Cell, Cell | None] = {root: None}
    queue = deque([root])
    while queue:
        row, col = queue.popleft()
        for dr, dc in NEIGHBOURS:
            neighbour = (row + dr, col + dc)
            r, c = neighbour
            inside = 0 <= r < free.shape[0] and 0 <= c < free.shape[1]
            if inside and free[r, c] and neighbour not in parents:
                parents[neighbour] = (row, col)
                queue.append(neighbour)
    return parents


def _cut_tree(parents: dict[Cell, Cell | None], count: int) -> list[list[Cell]]:
    """Split a rooted tree into ``count`` pieces of similar size."""
    children: dict[Cell, list[Cell]] = {node: [] for node in parents}
    root = next(node for node, parent in parents.items() if parent is None)
    for node, parent in parents.items():
        if parent is not None:
            children[parent].append(node)

    order = _post_order(root, children)
    weight = dict.fromkeys(parents, 1)
    for node in order:
        for child in children[node]:
            weight[node] += weight[child]

    remaining = len(parents)
    pieces: list[list[Cell]] = []
    detached: set[Cell] = set()
    for piece in range(count - 1):
        want = remaining / (count - piece)
        best, gap = None, np.inf
        for node in order:
            if node is root or node in detached:
                continue
            live = weight[node]
            if live <= 0:
                continue
            if abs(live - want) < gap:
                best, gap = node, abs(live - want)
        if best is None:
            break
        cut = _subtree(best, children, detached)
        pieces.append(cut)
        detached |= set(cut)
        # The cut subtree no longer counts towards its ancestors' weights.
        ancestor: Cell | None = parents[best]
        while ancestor is not None:
            weight[ancestor] -= len(cut)
            ancestor = parents[ancestor]
        remaining -= len(cut)
    pieces.append([node for node in parents if node not in detached])
    return [p for p in pieces if p]


def _post_order(root: Cell, children: dict[Cell, list[Cell]]) -> list[Cell]:
    order: list[Cell] = []
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        stack.append((node, True))
        for child in children[node]:
            stack.append((child, False))
    return order


def _subtree(root: Cell, children: dict[Cell, list[Cell]], skip: set[Cell]) -> list[Cell]:
    out: list[Cell] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node in skip:
            continue
        out.append(node)
        stack.extend(children[node])
    return out


# ----------------------------------------------------------------------
# Using a partition
# ----------------------------------------------------------------------
@dataclass
class InTerritory:
    """Wraps a single-robot planner so it plans inside one share.

    The whole reason the partition produces polygons rather than cell masks:
    a territory is a pool, and every offline planner already knows what to do
    with a pool.
    """

    planner: Any
    territory: Territory
    parent: Any
    name: str = "in_territory"

    def plan(self, pool: Pool, robot: Cleaner) -> CoveragePath:
        if self.territory.region.is_empty:
            raise ValueError(
                f"robot {self.territory.index} was given an empty territory by the partitioner"
            )
        share = self.territory.as_pool(self.parent)
        plan = self.planner.plan(share, robot)
        plan.notes.update(
            {
                "territory": self.territory.index,
                "territory_area": self.territory.area,
                "connected": self.territory.connected,
            }
        )
        return plan


@dataclass
class PartitionedFleet:
    """What :func:`partitioned` returns: a cut, and a planner for each share.

    A class rather than a closure with attributes stapled on. ``name`` is read
    by :class:`~zimablue.fleet.Fleet` when it writes the manifest and by the
    comparison harness when it labels a row, so it is part of the contract and
    belongs in the signature rather than in an assignment after the ``def``.
    """

    partition: Partitioner | str = "darp"
    planner: Any = "sweep_optimal"
    localisation: str = "odometry"
    cell: float = 0.5
    follower: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        cut = self.partition if isinstance(self.partition, str) else self.partition.name
        return f"{cut}+{self.planner}"

    def __call__(
        self,
        pool: Pool,
        robots: list[Cleaner],
        poses: list[tuple[float, float, float]],
    ) -> list[PathFollower]:
        cut = (
            make_partition(self.partition, cell=self.cell)
            if isinstance(self.partition, str)
            else self.partition
        )
        division = cut.divide(pool, robots, poses, cell=self.cell)
        followers = []
        for index, territory in enumerate(division.territories):
            inner = make_planner(self.planner) if isinstance(self.planner, str) else self.planner
            wrapped = InTerritory(
                planner=inner,
                territory=territory,
                parent=pool,
                name=f"{getattr(inner, 'name', 'planned')}@{division.method}{index}",
            )
            follow = PathFollower(wrapped, localisation=self.localisation, **self.follower)
            follow.name = wrapped.name
            follow.partition = division
            followers.append(follow)
        return followers


def partitioned(
    partition: Partitioner | str = "darp",
    planner: Any = "sweep_optimal",
    *,
    localisation: str = "odometry",
    cell: float = 0.5,
    **follower: Any,
) -> PartitionedFleet:
    """A fleet controller factory: cut the pool, then sweep each share.

    Returns something :class:`~zimablue.fleet.Fleet` calls once it knows where
    the robots are, because every partitioner here takes the start positions
    as input -- that is what "by robot positions" means.

        Fleet(
            pool="kidney",
            robots=3,
            controllers=partitioned("darp", "boustrophedon_cells"),
        )
    """
    return PartitionedFleet(
        partition=partition,
        planner=planner,
        localisation=localisation,
        cell=cell,
        follower=follower,
    )


def _planner_names() -> list[str]:
    return PLANNERS.names()
