"""Comparing coverage planners on every axis at once.

A planner comparison usually reports coverage, sometimes coverage and path
length, and calls it a result. That ranking is not wrong so much as
underdetermined: the planners at the top of it are typically within a
couple of points of each other on coverage while differing by a factor of two
in how far they drove, how much they turned, how early they got most of the
work done, and whether they finish at all on a pool with a waist in it.

So this measures everything that differs, and refuses to collapse it into
one number.
There is no scalar "best planner" here, and the matrix plot exists precisely so
that the shape of a planner is visible instead of its rank.

    from zimablue.planners.compare import compare
    from zimablue.planners.plots import plot_comparison

    result = compare(pools=("rectangular", "kidney"), minutes=20)
    print(result.table())
    plot_comparison(result).savefig("planners.png")

What is measured
----------------

============== =============================================================
coverage       fraction of navigable floor the head passed over
dirt           fraction of the dirt mass removed
possible       dirt removed over what was reachable in the time -- see below
evenness       how uniformly it cleaned, rather than how much
gap            area of the largest patch it never went near, m2
edges          share of the wall area the robot's brushes reached
efficiency     covered area over swept area -- 1.0 is a path with no overlap
turning        degrees of heading change per metre travelled
half           seconds to cover half the pool -- the anytime question
ergodic        how far the time distribution is from uniform, at the end
wasted         share of the run after the ergodic score stopped improving
energy         watt-hours
thrift         grams captured per watt-hour
trouble        collisions per minute
============== =============================================================

``possible`` is the regret column. Its denominator is a physical relaxation
computed by :mod:`zimablue.planners.oracle`: the heaviest cells that fit in
the run's swept-area budget, collected with no travel and no revisits. The
distance from 100% is what the planner cost, in a unit that stays comparable
when the pool, the dirt or the duration changes.

The worst gap is there because 90% coverage means two different things. A
planner that leaves a thin margin everywhere and one that leaves a whole
corner untouched score the same, and only one of them has left a pool with a
visibly dirty end. The largest *connected* uncovered patch tells them apart.

Efficiency and turning are the two that catch what coverage cannot. A planner
can reach 95% by driving over everything three times, and the efficiency column
says so. Turning is the one that costs on real hardware and appears in almost
no published comparison.

The truth-versus-odometry pair
------------------------------

An offline planner is entered twice, as ``name@truth`` and ``name@odometry``.
The first follows the plan from the simulator's true pose and measures the
*route*; the second follows it through the same EKF a real machine would have
and measures the route plus the localisation. Reading only the first is how a
planner comes to look excellent in a paper and disappointing on a floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from zimablue.planners.base import PLANNERS, PathFollower

FloatArray = NDArray[np.float64]

__all__ = [
    "DIMENSIONS",
    "FLEET_DIMENSIONS",
    "Comparison",
    "Dimension",
    "Trial",
    "compare",
    "compare_fleets",
    "coverage_curve",
    "evaluate_fleet",
]


@dataclass(frozen=True)
class Dimension:
    """One measured axis, and which way is good."""

    key: str
    label: str
    better: int
    """``+1`` if larger is better, ``-1`` if smaller is."""

    unit: str = ""
    scale: float = 1.0
    digits: int = 2

    def format(self, value: float) -> str:
        if not np.isfinite(value):
            return "never"
        return f"{value * self.scale:.{self.digits}f}{self.unit}"


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("coverage", "coverage", +1, "%", 100.0, 1),
    Dimension("dirt", "dirt", +1, "%", 100.0, 1),
    Dimension("possible", "of possible", +1, "%", 100.0, 0),
    Dimension("evenness", "evenness", +1, "", 1.0, 2),
    Dimension("gap", "worst gap", -1, "m2", 1.0, 1),
    Dimension("edges", "edges", +1, "%", 100.0, 0),
    Dimension("efficiency", "efficiency", +1, "", 1.0, 2),
    Dimension("turning", "turning", -1, "", 57.29577951308232, 1),
    Dimension("half", "to half", -1, "s", 1.0, 0),
    Dimension("ergodic", "ergodic", -1, "", 1.0, 3),
    Dimension("wasted", "wasted", -1, "%", 100.0, 0),
    Dimension("energy", "energy", -1, "Wh", 1.0, 1),
    Dimension("thrift", "g/Wh", +1, "", 1.0, 1),
    Dimension("trouble", "trouble", -1, "/min", 1.0, 1),
)


@dataclass
class Trial:
    """One planner, one pool, one seed."""

    planner: str
    pool: str
    seed: int
    scores: dict[str, float]
    path: FloatArray
    """The trajectory, decimated for drawing."""

    curve: tuple[FloatArray, FloatArray]
    """Time and coverage, for the anytime plot."""

    plan: Any = None
    """The :class:`~zimablue.planners.base.CoveragePath`, for offline planners."""

    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Comparison:
    """Every trial, and the ways of reading them."""

    trials: list[Trial]
    dimensions: tuple[Dimension, ...] = DIMENSIONS
    minutes: float = 0.0
    label: str = "planner comparison"
    """What the matrix plot calls itself. A fleet comparison scores teams on a
    different set of dimensions and should not claim to be ranking planners."""

    @property
    def planners(self) -> list[str]:
        seen: list[str] = []
        for trial in self.trials:
            if trial.planner not in seen:
                seen.append(trial.planner)
        return seen

    @property
    def pools(self) -> list[str]:
        seen: list[str] = []
        for trial in self.trials:
            if trial.pool not in seen:
                seen.append(trial.pool)
        return seen

    def select(self, planner: str | None = None, pool: str | None = None) -> list[Trial]:
        return [
            t
            for t in self.trials
            if (planner is None or t.planner == planner) and (pool is None or t.pool == pool)
        ]

    def score(self, planner: str, key: str, pool: str | None = None) -> float:
        """Median across seeds and pools.

        Median rather than mean because a planner that fails on one pool of
        five should show as "fine, with a failure" rather than as uniformly
        mediocre -- and because one infinite ``half`` would take a mean with
        it.
        """
        values = [t.scores.get(key, np.nan) for t in self.select(planner, pool)]
        values = [v for v in values if not np.isnan(v)]
        return float(np.median(values)) if values else float("nan")

    def matrix(self, pool: str | None = None) -> FloatArray:
        """Planners by dimensions, rescaled so 1 is the best value in each column.

        Each column is mapped onto ``[0, 1]`` between the worst and best
        planner *in this comparison*, with the sign flipped for the dimensions
        where less is better. It is a ranking aid, not a measurement: the
        numbers in :meth:`table` are the measurement.
        """
        raw = np.array(
            [
                [self.score(planner, dim.key, pool) for dim in self.dimensions]
                for planner in self.planners
            ],
            dtype=float,
        )
        out = np.zeros_like(raw)
        for j, dim in enumerate(self.dimensions):
            column = raw[:, j]
            known = np.isfinite(column)
            if not known.any():
                continue
            low, high = column[known].min(), column[known].max()
            span = high - low
            if span > 1e-12:
                scaled = (np.where(known, column, low) - low) / span
                scaled = scaled if dim.better > 0 else 1.0 - scaled
            else:
                # Everyone who got a number got the same number.
                scaled = np.full_like(column, 1.0 if not known.all() else 0.5)
            # A non-finite entry is a failure to reach the thing at all -- a
            # pool never half covered -- so it takes the bottom of the column
            # rather than being dropped and flattering the planner.
            out[:, j] = np.where(known, scaled, 0.0)
        return out

    def table(self, pool: str | None = None) -> str:
        """The measurements, as text, best in each column marked."""
        width = max(len(p) for p in self.planners) + 2
        header = f"{'':<{width}}" + "".join(f"{d.label:>11}" for d in self.dimensions)
        lines = [header, "-" * len(header)]
        raw = {
            planner: [self.score(planner, d.key, pool) for d in self.dimensions]
            for planner in self.planners
        }
        winners = []
        for j, dim in enumerate(self.dimensions):
            column = [raw[p][j] for p in self.planners]
            finite = [v for v in column if np.isfinite(v)]
            winners.append((max(finite) if dim.better > 0 else min(finite)) if finite else None)
        for planner in self.planners:
            row = f"{planner:<{width}}"
            for j, dim in enumerate(self.dimensions):
                value = raw[planner][j]
                text = dim.format(value)
                best = winners[j]
                if best is not None and np.isclose(value, best):
                    text = "*" + text
                row += f"{text:>11}"
            lines.append(row)
        if pool is None and len(self.pools) > 1:
            lines.append(f"median over {len(self.pools)} pools, {len(self.trials)} runs")
        return "\n".join(lines)

    def to_csv(self, path) -> None:
        import csv
        from pathlib import Path

        with Path(path).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["planner", "pool", "seed", *(d.key for d in self.dimensions)])
            for trial in self.trials:
                writer.writerow(
                    [trial.planner, trial.pool, trial.seed]
                    + [trial.scores.get(d.key, "") for d in self.dimensions]
                )


# ----------------------------------------------------------------------
def coverage_curve(
    recording, pool, *, swath: float, cell: float = 0.1, samples: int = 120
) -> tuple[FloatArray, FloatArray]:
    """Coverage against time, by stamping the swath along the trajectory.

    Recomputed here rather than taken from the simulator because the
    simulator only reports the final number, and the shape of the curve is
    half of what distinguishes these planners -- a sweep and a random walk can
    finish level and have got there completely differently.
    """
    grid = pool.grid(cell)
    navigable = pool.navigable_mask(cell)
    total = int(navigable.sum())
    frames = recording.frames
    time = np.asarray(frames["time"], dtype=float)
    rows = np.clip(
        ((np.asarray(frames["y"], dtype=float) - grid.miny) / cell).astype(int),
        0,
        grid.nrows - 1,
    )
    cols = np.clip(
        ((np.asarray(frames["x"], dtype=float) - grid.minx) / cell).astype(int),
        0,
        grid.ncols - 1,
    )
    reach = max(round(0.5 * swath / cell), 0)
    offsets = [
        (dr, dc)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if np.hypot(dr, dc) * cell <= 0.5 * swath
    ]

    covered = np.zeros_like(navigable)
    edges = np.unique(np.linspace(0, len(time), samples + 1).astype(int))
    times, fractions = [], []
    for start, stop in pairwise(edges):
        block_rows, block_cols = rows[start:stop], cols[start:stop]
        for dr, dc in offsets:
            covered[
                np.clip(block_rows + dr, 0, grid.nrows - 1),
                np.clip(block_cols + dc, 0, grid.ncols - 1),
            ] = True
        times.append(time[stop - 1])
        fractions.append((covered & navigable).sum() / total if total else 0.0)
    return np.asarray(times), np.asarray(fractions)


def largest_gap(spatial, cell: float) -> float:
    """Area of the biggest connected patch the robot never reached, m2.

    Two runs at 90% coverage are not the same run if one left a thin margin
    all the way round and the other left a corner. Connected-component
    labelling by breadth-first search rather than by scipy: the grid is a few
    thousand cells and the dependency is not worth it.
    """
    missed = np.asarray(spatial.missed)
    if not missed.any():
        return 0.0
    seen = np.zeros_like(missed)
    rows, cols = missed.shape
    biggest = 0
    for start in zip(*np.nonzero(missed), strict=True):
        if seen[start]:
            continue
        size = 0
        stack = [start]
        seen[start] = True
        while stack:
            row, col = stack.pop()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and missed[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    stack.append((r, c))
        biggest = max(biggest, size)
    return float(biggest * cell * cell)


def _controller(entry: str):
    """Build the controller an entry names.

    ``"bsa"`` is an online planner and is a controller already. ``"morse@truth"``
    is an offline planner wrapped in a follower.
    """
    if "@" not in entry:
        return entry, False
    name, mode = entry.split("@", 1)
    if name not in PLANNERS:
        raise KeyError(f"unknown planner {name!r}; available: {', '.join(PLANNERS.names())}")
    return PathFollower(name, localisation=mode), True


def evaluate(
    entry: str,
    *,
    pool: str = "rectangular",
    dirt: str = "autumn",
    seed: int = 1,
    minutes: float = 20.0,
    robot: str = "tracked",
    keep_path: int = 4,
) -> Trial:
    """Run one planner once and measure everything."""
    import zimablue as zb
    from zimablue.dynamics import ergodic_score

    controller, needs_truth = _controller(entry)
    if not needs_truth and isinstance(controller, str):
        needs_truth = getattr(zb.CONTROLLERS.create(controller), "needs_truth", False)
    simulation = zb.Simulation(
        pool=pool,
        robot=robot,
        dirt=dirt,
        controller=controller,
        seed=seed,
        expose_truth=needs_truth,
    )
    result = simulation.run(minutes=minutes)
    metrics, recording = result.metrics, result.require_recording()
    geometry = simulation.world.pool
    swath = simulation.robot.swath_width

    times, fractions = coverage_curve(recording, geometry, swath=swath)
    heading = np.asarray(recording.column("heading"), dtype=float)
    turned = float(np.abs(np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))).sum())
    distance = max(metrics.distance_traveled, 1e-6)
    area = float(geometry.navigable.area)

    half = float("inf")
    reached = np.flatnonzero(fractions >= 0.5)
    if reached.size:
        half = float(times[reached[0]])

    try:
        score = ergodic_score(recording, target="uniform", pool=geometry)
        ergodic, wasted = score.value, score.wasted
    except Exception:  # pragma: no cover - a run too short to score
        ergodic, wasted = float("nan"), float("nan")

    from zimablue.planners.oracle import dirt_bound

    bound = dirt_bound(
        recording, geometry, simulation.robot, seconds=metrics.runtime, cell=simulation.world.cell
    )
    scores = {
        "coverage": metrics.coverage,
        "dirt": metrics.dirt_removed_fraction,
        "possible": min(metrics.dirt_removed / bound, 1.0) if bound > 0 else float("nan"),
        "evenness": metrics.cleaning_uniformity,
        "gap": largest_gap(result.spatial, simulation.world.cell),
        "edges": metrics.wall_coverage,
        # Swept area is the path length times the swath; covered area is what
        # that actually reached. The ratio is 1 for a path that never crosses
        # itself and falls as it does.
        "efficiency": min(metrics.coverage * area / (distance * swath), 1.0),
        "turning": turned / distance,
        "half": half,
        "ergodic": ergodic,
        "wasted": wasted,
        "energy": metrics.energy_consumed,
        "thrift": metrics.grams_per_wh,
        "trouble": metrics.collisions / max(metrics.runtime / 60.0, 1e-6),
    }
    path = np.column_stack(
        [
            np.asarray(recording.column("x"), dtype=float),
            np.asarray(recording.column("y"), dtype=float),
        ]
    )[::keep_path]
    plan = getattr(controller, "path", None) if not isinstance(controller, str) else None
    return Trial(
        planner=entry,
        pool=pool,
        seed=seed,
        scores=scores,
        path=path,
        curve=(times, fractions),
        plan=plan,
        notes={"revisits": metrics.revisits, "distance": metrics.distance_traveled},
    )


ONLINE = (
    "spiral_stc",
    "full_stc",
    "bsa",
    "ba_star",
    "brick_and_mortar",
    "binn",
    "epsilon_star",
    "ppcpp",
    "frontier",
    "smc",
    "dirt_seeker",
)
OFFLINE = (
    "boustrophedon",
    "sweep_optimal",
    "trapezoidal",
    "boustrophedon_cells",
    "morse",
    "contour",
    "wavefront",
    "spanning_tree",
)
REFERENCE = ("baseline_coverage", "random_bounce", "systematic")


def default_entries(*, localisation: str = "odometry") -> tuple[str, ...]:
    """Everything in the package, offline planners followed on dead reckoning.

    Odometry rather than truth because that is the comparison a buyer of the
    robot is in. Pass ``localisation="both"`` to get each offline planner
    twice and see what perfect localisation would have been worth.
    """
    modes = ("truth", "odometry") if localisation == "both" else (localisation,)
    offline = tuple(f"{name}@{mode}" for name in OFFLINE for mode in modes)
    return REFERENCE + ONLINE + offline


def compare(
    entries: tuple[str, ...] | None = None,
    *,
    pools: tuple[str, ...] = ("rectangular",),
    seeds: tuple[int, ...] = (1,),
    minutes: float = 20.0,
    dirt: str = "autumn",
    robot: str = "tracked",
    jobs: int = 1,
    localisation: str = "odometry",
    on_result=None,
) -> Comparison:
    """Run every planner on every pool with every seed, and measure them.

    ``jobs`` above one runs the trials in worker processes. They are
    independent by construction -- a trial is a simulation with a fixed seed --
    so this is the one place in the package where parallelism is free.
    """
    entries = entries or default_entries(localisation=localisation)
    work = [(e, p, s) for e in entries for p in pools for s in seeds]

    trials: list[Trial] = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool_executor:
            futures = {
                pool_executor.submit(
                    evaluate, e, pool=p, seed=s, minutes=minutes, dirt=dirt, robot=robot
                ): (e, p, s)
                for e, p, s in work
            }
            for future in futures:
                trial = future.result()
                trials.append(trial)
                if on_result:
                    on_result(trial)
    else:
        for entry, pool_name, seed in work:
            trial = evaluate(
                entry, pool=pool_name, seed=seed, minutes=minutes, dirt=dirt, robot=robot
            )
            trials.append(trial)
            if on_result:
                on_result(trial)

    order = {e: i for i, e in enumerate(entries)}
    trials.sort(key=lambda t: (order.get(t.planner, 99), t.pool, t.seed))
    return Comparison(trials=trials, minutes=minutes)


# ----------------------------------------------------------------------
# Fleets
# ----------------------------------------------------------------------
FLEET_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("coverage", "coverage", +1, "%", 100.0, 1),
    Dimension("dirt", "dirt", +1, "%", 100.0, 1),
    Dimension("possible", "of possible", +1, "%", 100.0, 0),
    Dimension("possible", "of possible", +1, "%", 100.0, 0),
    Dimension("speedup", "speedup", +1, "x", 1.0, 2),
    Dimension("overlap", "overlap", -1, "%", 100.0, 0),
    Dimension("balance", "balance", +1, "", 1.0, 2),
    Dimension("gap", "worst gap", -1, "m2", 1.0, 1),
    Dimension("efficiency", "efficiency", +1, "", 1.0, 2),
    Dimension("turning", "turning", -1, "", 57.29577951308232, 1),
    Dimension("half", "to half", -1, "s", 1.0, 0),
    Dimension("bumps", "bumps", -1, "/min", 1.0, 1),
    Dimension("energy", "energy", -1, "Wh", 1.0, 1),
    Dimension("thrift", "g/Wh", +1, "", 1.0, 1),
)
"""What a *team* is judged on.

Three of these have no single-robot meaning at all. **Speedup** is the team's
coverage over its best member's, and its ceiling is the robot count -- a fleet
scoring 1.2x with three robots has bought two very expensive passengers.
**Overlap** is the floor more than one of them did, which is the cost of
coordinating badly. **Balance** is the shortest robot's distance over the
longest, and it catches the failure a coverage number hides completely: one
robot doing the work while another sits in a corner it was assigned and
finished.
"""


class FleetTrialKwargs(TypedDict):
    """The per-trial arguments :func:`evaluate_fleet` is called with.

    A plain dict literal mixing ints, strs, floats and bools infers as
    ``dict[str, object]``, and unpacking that into a typed signature is an
    error at every argument. Naming the shape once keeps the fan-out readable
    and the call checked.
    """

    robots: int
    pool: str
    seed: int
    minutes: float
    dirt: str
    share: bool


def evaluate_fleet(
    entry: str,
    *,
    robots: int = 3,
    pool: str = "rectangular",
    dirt: str = "autumn",
    seed: int = 1,
    minutes: float = 20.0,
    robot: str = "tracked",
    share: bool = True,
    keep_path: int = 6,
) -> Trial:
    """Run one fleet once and measure the team.

    ``entry`` is a controller name every robot runs (``"bsa"``), a partition
    and planner (``"darp+sweep_optimal"``), or ``"mstc"`` / ``"mstc_nobt"``.
    """
    from zimablue.dynamics import ergodic_score
    from zimablue.fleet import Fleet
    from zimablue.planners.cooperative import mstc
    from zimablue.planners.partition import partitioned

    if entry == "mstc":
        controllers: Any = mstc(backtracking=True)
    elif entry == "mstc_nobt":
        controllers = mstc(backtracking=False)
    elif "+" in entry:
        method, planner = entry.split("+", 1)
        controllers = partitioned(method, planner)
    else:
        controllers = entry

    fleet = Fleet(
        pool=pool, robots=robots, dirt=dirt, controllers=controllers, seed=seed, share=share
    )
    result = fleet.run(minutes=minutes)
    metrics, recording = result.metrics, result.require_recording()
    geometry = fleet.pool
    swath = fleet.robots[0].swath_width

    times, fractions = coverage_curve(recording, geometry, swath=swath)
    turned = 0.0
    for index in range(robots):
        heading = np.asarray(recording.column(f"r{index}.heading"), dtype=float)
        step = np.diff(heading)
        turned += float(np.abs(np.arctan2(np.sin(step), np.cos(step))).sum())
    distance = max(metrics.team.distance_traveled, 1e-6)
    area = float(geometry.navigable.area)

    half = float("inf")
    reached = np.flatnonzero(fractions >= 0.5)
    if reached.size:
        half = float(times[reached[0]])

    try:
        score = ergodic_score(recording, target="uniform", pool=geometry)
        ergodic, wasted = score.value, score.wasted
    except Exception:  # pragma: no cover - a run too short to score
        ergodic, wasted = float("nan"), float("nan")

    from zimablue.planners.oracle import collectable_bound

    initial = recording.dirt_at(0.0)
    navigable = geometry.navigable_mask(fleet.world.cell)
    bound = 0.0
    if initial.size and navigable.shape == initial.shape:
        bound = collectable_bound(
            initial,
            navigable,
            cell=fleet.world.cell,
            # The relaxation extends naturally: a team's swept-area budget is
            # the sum of its members'.
            speed=robots * fleet.robots[0].locomotion.max_speed,
            swath=swath,
            seconds=metrics.team.runtime,
            collectable_total=float(initial.sum()),
        )
    runtime = max(metrics.team.runtime / 60.0, 1e-6)
    scores = {
        "coverage": metrics.team.coverage,
        "dirt": metrics.team.dirt_removed_fraction,
        "possible": min(metrics.team.dirt_removed / bound, 1.0) if bound > 0 else float("nan"),
        "speedup": metrics.speedup,
        "overlap": metrics.overlap,
        "balance": metrics.balance,
        "gap": largest_gap(result.spatial, fleet.world.cell),
        "efficiency": min(metrics.team.coverage * area / (distance * swath), 1.0),
        "turning": turned / distance,
        "half": half,
        "bumps": metrics.encounters / runtime,
        "energy": metrics.team.energy_consumed,
        "thrift": (
            metrics.team.dirt_collected / metrics.team.energy_consumed
            if metrics.team.energy_consumed > 1e-9
            else 0.0
        ),
        "ergodic": ergodic,
        "wasted": wasted,
    }

    # One polyline per robot, separated by a gap so a single plot call draws
    # them as separate strokes rather than joining the last point of one robot
    # to the first of the next.
    strokes = []
    for index in range(robots):
        track = np.column_stack(
            [
                np.asarray(recording.column(f"r{index}.x"), dtype=float),
                np.asarray(recording.column(f"r{index}.y"), dtype=float),
            ]
        )[::keep_path]
        strokes.append(track)
        strokes.append(np.full((1, 2), np.nan))
    return Trial(
        planner=entry,
        pool=pool,
        seed=seed,
        scores=scores,
        path=np.vstack(strokes),
        curve=(times, fractions),
        notes={"robots": robots, "per_robot": [m.coverage for m in metrics.robots]},
    )


FLEET_ENTRIES = (
    "bsa",
    "frontier",
    "binn",
    "epsilon_star",
    "ppcpp",
    "smc",
    "auction",
    "binn_swarm",
    "smc_swarm",
    "mstc",
    "mstc_nobt",
    "voronoi+sweep_optimal",
    "geodesic+sweep_optimal",
    "strips+sweep_optimal",
    "darp+sweep_optimal",
    "forest+sweep_optimal",
    "darp+boustrophedon_cells",
)


def compare_fleets(
    entries: tuple[str, ...] = FLEET_ENTRIES,
    *,
    robots: int = 3,
    pools: tuple[str, ...] = ("rectangular",),
    seeds: tuple[int, ...] = (1,),
    minutes: float = 20.0,
    dirt: str = "autumn",
    share: bool = True,
    jobs: int = 1,
    on_result=None,
) -> Comparison:
    """The same harness, scoring teams instead of individuals."""
    work: list[tuple[str, FleetTrialKwargs]] = [
        (
            entry,
            FleetTrialKwargs(
                robots=robots,
                pool=p,
                seed=s,
                minutes=minutes,
                dirt=dirt,
                share=share,
            ),
        )
        for entry in entries
        for p in pools
        for s in seeds
    ]
    trials: list[Trial] = []

    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        # evaluate_fleet at module level, not a closure: a worker process gets
        # the function by pickling a reference to it, and a local function has
        # no reference to send.
        with ProcessPoolExecutor(max_workers=jobs) as pool_executor:
            futures = [pool_executor.submit(evaluate_fleet, e, **kw) for e, kw in work]
            for future in futures:
                trial = future.result()
                trials.append(trial)
                if on_result:
                    on_result(trial)
    else:
        for entry, kwargs in work:
            trial = evaluate_fleet(entry, **kwargs)
            trials.append(trial)
            if on_result:
                on_result(trial)

    order = {e: i for i, e in enumerate(entries)}
    trials.sort(key=lambda t: (order.get(t.planner, 99), t.pool, t.seed))
    return Comparison(
        trials=trials,
        dimensions=FLEET_DIMENSIONS,
        minutes=minutes,
        label=f"fleet comparison -- {robots} robots",
    )
