"""Online coverage planners: no map, one decision at a time.

An offline planner is handed the pool and computes a route. These are handed a
tick of sensor readings and asked what to do next, which is the situation a
real cleaner is in on a pool it has never seen. Everything they know they built
themselves, in the *estimated* frame, so localisation error and mapping error
compound the way they do on hardware.

They are :class:`~zimablue.controllers.base.Controller` implementations, so
they run with no ceremony::

    zb.Simulation(pool="kidney", controller="bsa").run(minutes=20)

===================  ====================================================
controller           the idea
===================  ====================================================
``spiral_stc``       Spiral-STC: grow a spanning tree, hug its perimeter
``full_stc``         the same, but enter cells the walls only half fill
``bsa``              spiral against a reference wall, then backtrack
``ba_star``          boustrophedon lanes, then A* to the next gap
``brick_and_mortar`` seal cells behind you when it costs no connectivity
``binn``             a neural activity field; drive uphill
``epsilon_star``     a coarse-to-fine potential; ascend when stuck
``ppcpp``            greedy one-step reward with a short lookahead
``frontier``         always drive to the nearest thing you have not done
===================  ====================================================

The shared substrate
--------------------

All nine differ in exactly one method. :class:`OnlineCoverage` owns the EKF,
the occupancy grid, the recovery behaviour and the business of driving to a
cell; a subclass implements :meth:`~OnlineCoverage.choose`, which is handed the
cell the robot is standing in and returns the cells to drive through next.

That is not a convenience. It is the only way the comparison at the end of
:mod:`zimablue.planners` means anything: if each algorithm brought its own
motion layer, a difference in coverage could always be the motion layer's
fault. Here the estimator, the grid resolution, the speeds and the bump
recovery are literally the same code, so a difference in the numbers is a
difference in the decision rule.

**Cells are the swath, minus a little overlap.** The literature usually assumes
a cell the size of the tool. Sizing them any other way makes an algorithm's
completeness guarantee describe a robot other than this one.

**Unknown is treated as free.** An online planner that refused to enter
unobserved space would never move. The robot drives into the unknown, hits
things, and writes down what it hit -- but only cells adjacent to observed
floor are candidate targets, otherwise every one of these would spend the run
charging at the far corner of a grid that is mostly imaginary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

import numpy as np

from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.controllers.systematic import MapCell, OccupancyMap
from zimablue.estimation import EstimatorConfig, PoseEstimator
from zimablue.geometry import wrap_angle
from zimablue.robot import Cleaner, DriveCommand

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "BSA",
    "BAStar",
    "BrickAndMortar",
    "EpsilonStar",
    "EvidenceMap",
    "Frontier",
    "FullSTC",
    "NeuralField",
    "OnlineCoverage",
    "OnlineTuning",
    "Predictive",
    "SpiralSTC",
]

Cell = tuple[int, int]

# (row, col) steps, counterclockwise, with row = +y and col = +x.
EAST, NORTH, WEST, SOUTH = (0, 1), (1, 0), (0, -1), (-1, 0)
CCW: tuple[Cell, ...] = (EAST, NORTH, WEST, SOUTH)


def _turn(direction: Cell, quarters: int) -> Cell:
    """Rotate a step by ``quarters`` counterclockwise right angles."""
    return CCW[(CCW.index(direction) + quarters) % 4]


def _order_from(back: Cell) -> tuple[Cell, ...]:
    """The four directions counterclockwise, starting at ``back``."""
    start = CCW.index(back)
    return tuple(CCW[(start + i) % 4] for i in range(4))


@dataclass
class OnlineTuning:
    """Everything about *driving* that these controllers share."""

    cruise: float = 0.9
    """Fraction of top speed on a straight run."""

    turn_gain: float = 2.6
    arrive: float = 0.45
    """How close to a cell centre counts as arrived, as a fraction of a cell.
    Below about a third the robot overshoots and circles back; above about
    two-thirds it cuts corners and misses the cell it was aiming at."""

    wall_threshold: float = 0.3
    recover_time: float = 1.8
    wedged_time: float = 2.0
    settle: float = 0.35
    """Seconds held still at each decision point. This is the zero-velocity
    window that makes the gyro bias observable; without it the grid shears."""

    overlap: float = 0.1
    """Cell size as a fraction below the swath."""


class EvidenceMap(OccupancyMap):
    """An occupancy grid that can change its mind about a wall.

    :class:`~zimablue.controllers.systematic.OccupancyMap` writes a wall the
    first time it sees one and never takes it back, which is fine for a
    controller that uses the map only to find frontiers. It is fatal for these
    planners, because the map *is* the plan: three minutes of sonar echoes
    scattered by a drifting pose estimate turned an eight-metre pool into 552
    wall cells around 108 of floor, and the robot declared the job finished
    having cleaned an eighth of it.

    Two changes fix that, and both are things the robot actually knows:

    * a wall needs corroboration -- ``votes`` sightings before the cell counts
      as blocked, and a beam that passes *through* a cell takes a vote away;
    * the robot's own footprint is proof. Wherever the hull has been is floor,
      whatever the sonar said about it earlier.

    Bump switches are worth more than echoes, which is what the ``weight``
    argument on :meth:`mark_wall` carries.
    """

    def __init__(self, *, votes: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.votes_needed = int(votes)
        self.votes = np.zeros((self.size, self.size), dtype=np.int16)

    def mark_wall(self, x: float, y: float, weight: int = 1) -> None:
        row, col = self.to_index(x, y)
        tally = min(int(self.votes[row, col]) + weight, 2 * self.votes_needed)
        self.votes[row, col] = tally
        if tally >= self.votes_needed:
            self.grid[row, col] = MapCell.WALL

    def _set_free(self, x: float, y: float) -> None:
        row, col = self.to_index(x, y)
        if self.votes[row, col] > 0:
            self.votes[row, col] -= 1
        if self.votes[row, col] < self.votes_needed:
            self.grid[row, col] = MapCell.FREE

    def mark_free(self, x: float, y: float, radius: float) -> None:
        rows, cols = self._disk_indices(x, y, radius)
        if rows.size:
            self.votes[rows, cols] = 0
            self.grid[rows, cols] = MapCell.FREE


class OnlineCoverage:
    """Estimation, mapping, recovery and motion. Subclasses decide where to go.

    The one method to override is :meth:`choose`. It is called when the robot
    has finished the cells it was given and needs more, and it returns a route
    -- a list of adjacent cells -- or ``None`` to declare the pool finished.
    Most of these algorithms return a single cell; the ones that backtrack
    return a whole path across the pool.
    """

    name = "online_coverage"

    def __init__(
        self,
        *,
        tuning: OnlineTuning | None = None,
        estimator: EstimatorConfig | None = None,
        cell: float | None = None,
        extent: float = 30.0,
        votes: int = 3,
    ) -> None:
        self.tuning = tuning or OnlineTuning()
        self.estimator_config = estimator
        self._cell = cell
        self.extent = extent
        self.votes = int(votes)
        self.estimator = PoseEstimator(self.estimator_config)
        self.map = EvidenceMap(cell=cell or 0.4, extent=extent)
        self.done: set[Cell] = set()
        self.sealed: set[Cell] = set()
        self._peer_done: set[Cell] = set()
        self._peer_cells: set[Cell] = set()
        self.finished = False
        self.index = 0
        self.origin: tuple[float, float, float] | None = None
        self.blackboard: Any = None
        self.share = True
        self.fleet_size = 1

    # ------------------------------------------------------------------
    def attach_fleet(
        self,
        *,
        index: int,
        blackboard: Any,
        origin: tuple[float, float, float],
        fleet_size: int = 1,
        share: bool = True,
    ) -> None:
        """Join a fleet. Called by :class:`~zimablue.fleet.Fleet` before reset.

        Two things arrive with the invitation. The **origin** is where this
        robot starts in the frame the fleet shares, and starting the estimator
        there is what makes one robot's grid cells mean the same as another's
        -- up to the drift each of them accumulates afterwards, which is the
        interesting part rather than a caveat. The **blackboard** is the radio.

        ``share`` off gives a fleet of strangers: they still collide, still
        clean the same dirt, and still get in each other's way, but no robot
        knows another exists. It is the baseline every cooperative method has
        to beat, and it is not as far behind as the literature implies.
        """
        self.index = int(index)
        self.blackboard = blackboard
        self.origin = origin
        self.fleet_size = int(fleet_size)
        self.share = bool(share)

    # ------------------------------------------------------------------
    def reset(self, robot: Cleaner) -> None:
        self.radius = robot.radius
        self.swath = robot.swath_width
        cell = self._cell or max(self.swath * (1.0 - self.tuning.overlap), 0.15)
        self.estimator = PoseEstimator(self.estimator_config, origin=self.origin or (0.0, 0.0, 0.0))
        self.map = EvidenceMap(cell=cell, extent=self.extent, votes=self.votes)
        self.cell = self.map.cell
        self.done = set()
        self.sealed = set()
        self.finished = False
        self.facing: Cell = EAST
        start = self.origin or (0.0, 0.0, 0.0)
        self.here: Cell = self.map.to_index(start[0], start[1])
        self._peer_done: set[Cell] = set()
        self._peer_cells: set[Cell] = set()
        self._route: list[Cell] = []
        self._last_time = 0.0
        self._contact_since: float | None = None
        self._recovering_until = -1e9
        self._recover_turn = 1.0
        self._hold_until = -1e9
        self._decisions = 0
        self._backtracks = 0
        self._stalled = 0
        self._empty = 0
        self.begin()

    def begin(self) -> None:
        """Hook for per-run subclass state. Called at the end of ``reset``."""

    # ------------------------------------------------------------------
    def step(self, ctl: ControlInput) -> DriveCommand:
        dt = max(ctl.time - self._last_time, 0.0)
        self._last_time = ctl.time

        speed, gyro, wheel = self._proprioception(ctl)
        self.estimator.predict(speed, gyro, dt)
        self.estimator.zero_velocity_update(
            gyro, dt, moving=wheel > self.estimator.config.zupt_speed
        )
        pose = self.estimator.estimate
        self.map.absorb(ctl, pose, radius=self.radius, swath=self.swath)

        if ctl.battery <= ctl.robot.power.battery.cutoff or self.finished:
            return DriveCommand.stop()

        self._sync(ctl, pose)
        blocked, wedged = self._obstruction(ctl)
        if blocked:
            # Whatever is in front of us is not floor. Writing it down is what
            # turns a bump into information; without this the algorithms would
            # keep routing through the same wall.
            ahead = self._neighbour(self.here, self._heading_step(pose.heading))
            if ahead is not None:
                self.map.mark_wall(*self.map.to_world(*ahead), weight=2)
                if not self.passable(ahead) and ahead in self._route:
                    self._route = []
        if wedged and ctl.time > self._recovering_until:
            self._recover_turn *= -1.0
            self._recovering_until = ctl.time + self.tuning.recover_time
            self._contact_since = None
            self._route = []
        if ctl.time < self._recovering_until:
            return self._recover(ctl)

        return self._act(ctl, pose)

    def _sync(self, ctl: ControlInput, pose) -> None:
        """Say where we are and what we have done; hear the same back.

        The covered set goes over the wire by reference rather than by copy.
        That models a robot broadcasting its map continuously and keeps the
        tick cheap; what it does *not* model is bandwidth, so a result here
        about a fleet with a small ``comms_range`` is about range and nothing
        else.
        """
        if self.blackboard is None:
            return
        self.blackboard.publish(
            self.index,
            pose.x,
            pose.y,
            pose.heading,
            covered=self.done if self.share else set(),
            time=ctl.time,
            extras={"cell": self.cell},
        )
        if not self.share:
            self._peer_done = set()
            self._peer_cells = set()
            return
        peers = self.blackboard.peers(self.index)
        self._peer_done = set().union(*(p.covered for p in peers)) if peers else set()
        # Where they are *now*, as a cell each. Enough to make the routing
        # steer round a team-mate instead of into one; the collision resolver
        # is the backstop, not the plan.
        self._peer_cells = {self.map.to_index(p.x, p.y) for p in peers}

    # -- the motion layer -------------------------------------------------
    def _act(self, ctl: ControlInput, pose) -> DriveCommand:
        top = ctl.robot.locomotion.max_speed
        cell = self.map.to_index(pose.x, pose.y)
        if cell != self.here:
            step = (cell[0] - self.here[0], cell[1] - self.here[1])
            if step in CCW:
                self.facing = step
            self.here = cell
        self.done.add(cell)

        if ctl.time < self._hold_until:
            return DriveCommand(0.0, 0.0, brush=True, pump=1.0)

        while self._route and self._route[0] == self.here:
            self._route.pop(0)

        if not self._route:
            route = self.choose(self.here)
            self._decisions += 1
            if not route:
                return self._nothing_left(ctl)
            self._empty = 0
            if len(route) > 1:
                self._backtracks += 1
            self._route = list(route)
            step = (route[0][0] - self.here[0], route[0][1] - self.here[1])
            if step in CCW and step != self.facing:
                # Stop for the turn. It costs nothing -- a differential drive
                # has to slow down to turn anyway -- and it buys the
                # zero-velocity update that keeps the gyro bias observable.
                # Holding at *every* decision instead cost a third of the run.
                self._hold_until = ctl.time + self.tuning.settle
                return DriveCommand(0.0, 0.0, brush=True, pump=1.0)

        target = self._route[0]
        tx, ty = self.map.to_world(*target)
        if float(np.hypot(tx - pose.x, ty - pose.y)) < self.tuning.arrive * self.cell:
            self.done.add(target)
            self._route.pop(0)
            if not self._route:
                return DriveCommand(0.0, 0.0, brush=True, pump=1.0)
            target = self._route[0]
            tx, ty = self.map.to_world(*target)

        bearing = float(np.arctan2(ty - pose.y, tx - pose.x))
        error = float(wrap_angle(bearing - pose.heading))
        forward = top * self.tuning.cruise * float(np.clip(np.cos(error), 0.0, 1.0))
        return DriveCommand.from_body(forward, self.tuning.turn_gain * error, ctl.robot.locomotion)

    def _nothing_left(self, ctl: ControlInput) -> DriveCommand:
        """Called when :meth:`choose` finds nowhere to go.

        Not believed the first time. A pocket that looks finished usually
        opens up the moment the sonar sees round the corner, so the robot
        turns on the spot once and asks again; two empty answers in a row is
        the pool being done.
        """
        self._empty += 1
        if self._empty >= 2:
            self.finished = True
            return DriveCommand.stop()
        self._recovering_until = ctl.time + self.tuning.recover_time
        return self._recover(ctl)

    def _recover(self, ctl: ControlInput) -> DriveCommand:
        top = ctl.robot.locomotion.max_speed
        left = ctl.time - (self._recovering_until - self.tuning.recover_time)
        if left < self.tuning.recover_time * 0.45:
            return DriveCommand(-top * 0.5, -top * 0.5, brush=True, pump=1.0)
        turn = top * 0.55 * self._recover_turn
        return DriveCommand(-turn, turn, brush=True, pump=1.0)

    # -- what the subclass overrides ---------------------------------------
    def choose(self, here: Cell) -> list[Cell] | None:
        raise NotImplementedError

    # -- the grid, as the algorithms see it ---------------------------------
    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.map.size and 0 <= cell[1] < self.map.size

    def passable(self, cell: Cell) -> bool:
        """Not a known wall, not sealed off, and not where a team-mate is.

        A team-mate is treated as a wall for one tick at a time rather than
        written into the map: they move, and a robot that remembered where its
        colleagues used to be would fill the pool with obstacles that are not
        there any more.
        """
        return (
            self.inside(cell)
            and self.map.grid[cell] != MapCell.WALL
            and cell not in self.sealed
            and cell not in self._peer_cells
        )

    def observed(self, cell: Cell) -> bool:
        return self.inside(cell) and self.map.grid[cell] == MapCell.FREE

    def candidate(self, cell: Cell) -> bool:
        """Worth driving to: passable, not done, and next to observed floor.

        The last clause is what keeps these controllers inside the pool. The
        grid is thirty metres across and almost all of it is unobserved; every
        unobserved cell is trivially "not done", so without an adjacency
        requirement the nearest-unvisited rule would point at open nothing.
        """
        if cell in self.done or cell in self._peer_done or not self.passable(cell):
            return False
        if self.observed(cell):
            return True
        return any(self.observed(self._step(cell, d)) for d in CCW)

    def _step(self, cell: Cell, direction: Cell) -> Cell:
        return (cell[0] + direction[0], cell[1] + direction[1])

    def _neighbour(self, cell: Cell, direction: Cell) -> Cell | None:
        nxt = self._step(cell, direction)
        return nxt if self.inside(nxt) else None

    def neighbours(self, cell: Cell) -> list[Cell]:
        return [self._step(cell, d) for d in CCW if self.passable(self._step(cell, d))]

    def walls_around(self, cell: Cell) -> int:
        """How many of the eight neighbours are wall, sealed, or off the grid."""
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                if not self.passable((cell[0] + dr, cell[1] + dc)):
                    count += 1
        return count

    # -- routing ------------------------------------------------------------
    def route_to(self, goal: Cell) -> list[Cell] | None:
        """Shortest passable route from ``here`` to ``goal``, exclusive of here."""
        return self._search(lambda c: c == goal)

    def nearest(self, wanted=None) -> list[Cell] | None:
        """Route to the closest cell satisfying ``wanted`` (default: candidate)."""
        return self._search(wanted or self.candidate)

    def _search(self, wanted) -> list[Cell] | None:
        """Breadth-first over passable cells. Returns the route, not the cell.

        Breadth-first rather than A*: the goal is a *predicate* here, not a
        point, so there is nothing to write a heuristic against.
        """
        start = self.here
        came: dict[Cell, Cell] = {start: start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell != start and wanted(cell):
                return self._unwind(came, cell)
            for direction in CCW:
                nxt = self._step(cell, direction)
                if nxt not in came and self.passable(nxt):
                    came[nxt] = cell
                    queue.append(nxt)
        return None

    def astar(self, goal: Cell) -> list[Cell] | None:
        """Shortest route to a known cell, with a Manhattan heuristic."""
        start = self.here
        came: dict[Cell, Cell] = {start: start}
        cost = {start: 0}
        heap = [(0, start)]
        while heap:
            _, cell = heappop(heap)
            if cell == goal:
                return self._unwind(came, cell)
            for direction in CCW:
                nxt = self._step(cell, direction)
                if not self.passable(nxt):
                    continue
                fresh = cost[cell] + 1
                if fresh < cost.get(nxt, 1 << 30):
                    cost[nxt] = fresh
                    came[nxt] = cell
                    guess = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    heappush(heap, (fresh + guess, nxt))
        return None

    def _unwind(self, came: dict[Cell, Cell], cell: Cell) -> list[Cell]:
        route = [cell]
        while came[cell] != cell:
            cell = came[cell]
            route.append(cell)
        route.reverse()
        return route[1:]

    def backtrack(self) -> list[Cell] | None:
        """Give up locally and drive to the nearest unfinished cell.

        Every algorithm here needs this, including the ones whose papers prove
        they do not: the proofs assume a known grid, and a robot that finds its
        walls by hitting them will sometimes seal itself into a pocket it has
        already finished.
        """
        self._stalled += 1
        return self.nearest()

    # ------------------------------------------------------------------
    def _proprioception(self, ctl: ControlInput) -> tuple[float, float, float]:
        encoder = ctl.reading("encoder")
        if encoder is not None and encoder.valid:
            left, right = float(encoder[0]), float(encoder[1])
        else:
            left = right = 0.0
        imu = ctl.reading("imu")
        gyro = float(imu[2]) if imu is not None and imu.valid else 0.0
        return 0.5 * (left + right), gyro, max(abs(left), abs(right))

    def _obstruction(self, ctl: ControlInput) -> tuple[bool, bool]:
        contact = ctl.reading("contact")
        front = side = False
        if contact is not None and contact.valid:
            front = bool(contact[0] > 0.5)
            side = bool(contact[1] > 0.5 or contact[2] > 0.5)
        sonar = ctl.reading("sonar")
        ahead = float(sonar[0]) if sonar is not None and sonar.valid else float("inf")
        if not np.isfinite(ahead):
            ahead = float("inf")
        blocked = front or ahead <= self.tuning.wall_threshold
        if not (front or side):
            self._contact_since = None
        elif self._contact_since is None:
            self._contact_since = ctl.time
        wedged = (
            self._contact_since is not None
            and ctl.time - self._contact_since > self.tuning.wedged_time
        ) or ctl.extras.get("stuck", 0.0) > 0.5
        return blocked, wedged

    def _heading_step(self, heading: float) -> Cell:
        """The grid direction closest to a continuous heading."""
        return CCW[round(heading / (np.pi / 2)) % 4]

    # ------------------------------------------------------------------
    def telemetry(self) -> dict[str, float]:
        pose = self.estimator.estimate
        return {
            "est_x": pose.x,
            "est_y": pose.y,
            "est_heading": pose.heading,
            "est_sigma": pose.position_sigma,
            "cells_done": float(len(self.done)),
            "decisions": float(self._decisions),
            "backtracks": float(self._backtracks),
            "stalled": float(self._stalled),
            "mapped": float(self.map.explored_cells),
            "peer_cells": float(len(self._peer_done)),
        }


# ======================================================================
# Spanning-tree coverage
# ======================================================================
class SpiralSTC(OnlineCoverage):
    """Spiral-STC (Gabriely & Rimon, 2001).

    Group the cells into 2x2 *mega-cells*, grow a spanning tree over the
    mega-cells depth-first, and drive around the outside of that tree. Because
    a tree has no cycles, its thickened perimeter passes through every sub-cell
    exactly once, which is the algorithm's whole claim: complete coverage with
    no repetition, and the proof is a property of trees rather than of the
    robot.

    Implemented as the paper states it, incrementally. At each sub-cell the
    robot looks at the four directions counterclockwise from the one it came
    from and takes the first that is unfinished and reachable, growing the tree
    when that means entering a new mega-cell. Nothing here plans ahead: the
    spiral is what the local rule produces.

    The original has a well-known limitation and it is kept rather than
    quietly patched. A mega-cell is only entered if *all four* of its
    sub-cells are clear, so anything the wall clips is skipped entirely --
    which along a curved pool wall is a lot. :class:`FullSTC` is the fix, and
    running the two against each other is the cheapest way to see what the
    limitation costs.
    """

    name = "spiral_stc"
    partial = False

    def begin(self) -> None:
        self._tree: set[Cell] = {self.mega(self.here)}

    @staticmethod
    def mega(cell: Cell) -> Cell:
        return (cell[0] >> 1, cell[1] >> 1)

    @staticmethod
    def sub_cells(mega: Cell) -> list[Cell]:
        return [(2 * mega[0] + r, 2 * mega[1] + c) for r in (0, 1) for c in (0, 1)]

    def _admits(self, mega: Cell, entry: Cell) -> bool:
        if self.partial:
            return True
        return all(self.passable(cell) for cell in self.sub_cells(mega))

    def _untouched(self, mega: Cell) -> bool:
        """No sub-cell visited yet -- entering a touched mega-cell would close
        a loop in the tree, and the no-repetition guarantee dies with it."""
        return all(cell not in self.done for cell in self.sub_cells(mega))

    def choose(self, here: Cell) -> list[Cell] | None:
        home = self.mega(here)
        for direction in _order_from(_turn(self.facing, 2)):
            step = self._step(here, direction)
            if not self.candidate(step):
                continue
            target = self.mega(step)
            if target == home or target in self._tree:
                return [step]
            if self._untouched(target) and self._admits(target, step):
                self._tree.add(target)
                return [step]

        route = self.backtrack()
        if route:
            # The perimeter walk restarts from wherever we land, so that cell's
            # mega-cell becomes the root of a new tree.
            self._tree.add(self.mega(route[-1]))
        return route


class FullSTC(SpiralSTC):
    """Full Spiral-STC: the same walk, into partially blocked cells too.

    Spiral-STC's completeness holds over the mega-cells it accepts, and it
    accepts only the ones the obstacles miss entirely. On a rectangular room
    with rectangular furniture that is nearly everything; on a kidney-shaped
    pool it is a ring of skipped cells all the way round the wall.

    Full-STC enters a mega-cell as long as the sub-cell it is stepping into is
    clear, and the perimeter walk then does the right thing by itself -- the
    blocked sub-cells simply never come up as candidates. The cost is that the
    walk is no longer a clean spiral and needs to backtrack more.
    """

    name = "full_stc"
    partial = True


# ======================================================================
# Spiral and boustrophedon with backtracking
# ======================================================================
class BSA(OnlineCoverage):
    """Backtracking Spiral Algorithm (Gonzalez et al., 2005).

    Wall-following, made into a coverage strategy: keep a reference side --
    here the right -- and always turn towards it if you can. Against an actual
    wall that traces the boundary; on the second lap the first lap *is* the
    wall, so the path spirals inward on its own with no lane bookkeeping at
    all.

    Spirals end enclosed, so the other half of the algorithm is the
    backtracking: when every direction is finished, drive to the nearest cell
    that is not, and start a new spiral there.
    """

    name = "bsa"

    def choose(self, here: Cell) -> list[Cell] | None:
        # Right, straight, left, back. Reversing the first two would spiral the
        # other way; nothing else about the algorithm changes.
        for quarters in (-1, 0, 1, 2):
            step = self._step(here, _turn(self.facing, quarters))
            if self.candidate(step):
                return [step]
        return self.backtrack()


class BAStar(OnlineCoverage):
    """BA* (Viet et al., 2013): boustrophedon motion, A* backtracking.

    The motion half is a lane sweep with a fixed direction priority, which is
    the oldest idea in the field and still the one that turns least. The
    contribution is the other half: when the lane sweep runs out, BA* does not
    take the nearest unfinished cell, it takes the one that is *cheapest to
    reach*, evaluated by running A* to each candidate.

    Those differ whenever the pool is not convex. The nearest gap as the crow
    flies can be on the far side of a wall, and BSA -- which measures distance
    by breadth-first search through free space, so it never makes that mistake
    -- will pick the same cell for a different reason. Where BA* actually wins
    is that it prefers backtracking points adjacent to ground it has already
    done, so the next sweep starts against a known edge.
    """

    name = "ba_star"

    def begin(self) -> None:
        self._lane: Cell = NORTH

    def choose(self, here: Cell) -> list[Cell] | None:
        for direction in (self._lane, EAST, _turn(self._lane, 2), WEST):
            step = self._step(here, direction)
            if self.candidate(step):
                if direction in (EAST, WEST):
                    # Stepped across to the next lane: run it the other way.
                    self._lane = _turn(self._lane, 2)
                return [step]
        return self._nearest_by_cost()

    def _nearest_by_cost(self, limit: int = 12) -> list[Cell] | None:
        """A* to each nearby backtracking point; keep the cheapest route."""
        points = [
            cell
            for cell in self.done
            for direction in CCW
            if self.candidate(self._step(cell, direction))
        ]
        if not points:
            return self.backtrack()
        points.sort(key=lambda c: abs(c[0] - self.here[0]) + abs(c[1] - self.here[1]))
        best: list[Cell] | None = None
        for point in points[:limit]:
            route = self.astar(point)
            if route is not None and (best is None or len(route) < len(best)):
                best = route
        self._backtracks += 1
        return best or self.backtrack()


class BrickAndMortar(OnlineCoverage):
    """Brick and mortar (Ferranti, Trigoni & Levene, 2007).

    The others avoid re-covering ground by remembering where they have been.
    This one avoids it by *walling the ground off*: a finished cell is turned
    into an obstacle, so it stops being somewhere the robot can drive through.
    The name is the picture -- the cleaned region is masonry, and it grows from
    the edges inward.

    The trick is the test before each brick is laid. Sealing a cell is only
    allowed if it does not cut the unfinished region in two, checked locally by
    counting how many separate groups the cell's eight neighbours fall into. It
    is the connectivity number from thinning algorithms, and one cheap test per
    cell is what keeps the robot from bricking itself into a corner.
    """

    name = "brick_and_mortar"

    def choose(self, here: Cell) -> list[Cell] | None:
        self._seal(here)
        best: Cell | None = None
        rank: tuple[int, int] | None = None
        for quarters in (0, -1, 1, 2):
            direction = _turn(self.facing, quarters)
            step = self._step(here, direction)
            if not self.candidate(step):
                continue
            # Hug the masonry: the cell with the most walls around it is the
            # one most likely to be orphaned if we leave it for later.
            score = (self.walls_around(step), 1 if quarters == 0 else 0)
            if rank is None or score > rank:
                rank, best = score, step
        if best is not None:
            return [best]
        return self.backtrack()

    def _seal(self, cell: Cell) -> None:
        """Lay a brick, if the cell is a simple point of the free space.

        Two conditions, both from the thinning literature: the eight
        neighbours must form a single group, and at least one of the four
        edge-adjacent neighbours must already be blocked. The second is what
        stops the robot bricking over the middle of an open floor, which
        satisfies the first condition trivially and puts an island in the
        pool.
        """
        edged = any(not self.passable(self._step(cell, d)) for d in CCW)
        if edged and self._groups(cell) <= 1:
            self.sealed.add(cell)

    def _groups(self, cell: Cell) -> int:
        """Connected groups of passable cells in the eight-neighbour ring.

        Two or more means the cell is a bridge between parts of the free space
        and must stay open. Walking the ring in order and counting the runs is
        the whole test.
        """
        ring = [
            self.passable((cell[0] + dr, cell[1] + dc))
            for dr, dc in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
        ]
        if all(ring):
            return 1
        return sum(1 for i, open_ in enumerate(ring) if open_ and not ring[i - 1])


# ======================================================================
# Field methods
# ======================================================================
class NeuralField(OnlineCoverage):
    """BINN (Luo & Yang, 2008): a shunting neural field over the grid.

    Every cell is a neuron. Unfinished floor excites, walls inhibit, and
    neighbours pass activity to each other, so attraction spreads outward from
    the parts of the pool still to do and decays with distance. The robot
    always steps to the most active neighbour. No path is ever planned, no
    graph is ever searched, and the robot still leaves a dead end and crosses
    the pool because the far side is shouting louder than the near side is.

    The dynamics are Grossberg's shunting equation, one Euler step per cell per
    iteration::

        dx/dt = -A x + (B - x)([I]+ + mu * sum_j [x_j]+) - (D + x)[I]-

    which is bounded in ``[-D, B]`` by construction -- the excitatory term
    vanishes as ``x`` approaches ``B``. The field persists between decisions
    and is relaxed a fixed number of iterations each time, so it is genuinely
    a dynamical system reaching for equilibrium rather than a distance
    transform in disguise.
    """

    name = "binn"

    def __init__(
        self,
        *,
        decay: float = 8.0,
        upper: float = 1.0,
        lower: float = 1.0,
        coupling: float = 0.7,
        drive: float = 100.0,
        relax: int = 12,
        heading_weight: float = 0.12,
        rate: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.decay = decay
        self.upper = upper
        self.lower = lower
        self.coupling = coupling
        self.drive = drive
        self.relax = relax
        self.heading_weight = heading_weight
        self.rate = rate

    def begin(self) -> None:
        self.activity = np.zeros((self.map.size, self.map.size), dtype=float)

    def _input(self) -> np.ndarray:
        wall = self.map.grid == MapCell.WALL
        observed = self.map.grid == MapCell.FREE
        touching = np.zeros_like(observed)
        touching[1:, :] |= observed[:-1, :]
        touching[:-1, :] |= observed[1:, :]
        touching[:, 1:] |= observed[:, :-1]
        touching[:, :-1] |= observed[:, 1:]

        done = np.zeros_like(observed)
        if self.done:
            rows, cols = np.array(sorted(self.done)).T
            done[rows, cols] = True

        field = np.zeros_like(self.activity)
        field[(observed | touching) & ~wall & ~done] = self.drive
        field[wall] = -self.drive
        return field

    def _relax(self) -> None:
        """Iterate the field towards the equilibrium of the shunting equation.

        Setting ``dx/dt = 0`` and solving for ``x`` gives

            x* = (B E - D I) / (A + E + I)

        with ``E`` the excitation reaching the cell and ``I`` the inhibition,
        and the update is a damped step towards it. Integrating the ODE
        forward instead is what the equation looks like it wants, and it does
        not work: the excitatory input is 100 and the decay is 8, so any Euler
        step large enough to propagate activity in a few iterations overshoots
        the bound and rings between the ceiling and the floor. The field then
        reports its *lowest* value at the cells that should be shouting
        loudest, and the robot paces between two cells for the whole run --
        which is exactly what it did.
        """
        field = self._input()
        for _ in range(self.relax):
            positive = np.maximum(self.activity, 0.0)
            spread = np.zeros_like(positive)
            spread[1:, :] += positive[:-1, :]
            spread[:-1, :] += positive[1:, :]
            spread[:, 1:] += positive[:, :-1]
            spread[:, :-1] += positive[:, 1:]
            excite = np.maximum(field, 0.0) + self.coupling * spread
            inhibit = np.maximum(-field, 0.0)
            equilibrium = (self.upper * excite - self.lower * inhibit) / (
                self.decay + excite + inhibit
            )
            self.activity += self.rate * (equilibrium - self.activity)

    def choose(self, here: Cell) -> list[Cell] | None:
        self._relax()
        best: Cell | None = None
        score = -np.inf
        for quarters in (0, -1, 1, 2):
            direction = _turn(self.facing, quarters)
            step = self._step(here, direction)
            if not self.passable(step):
                continue
            turn = abs(((CCW.index(direction) - CCW.index(self.facing) + 2) % 4) - 2) / 2.0
            value = float(self.activity[step]) - self.heading_weight * turn
            if value > score:
                score, best = value, step
        # A flat neighbourhood means the excitation has not reached here yet.
        # The paper relaxes the field to equilibrium before every step and so
        # cannot stall; a fixed dozen iterations can, hence the fallback.
        if best is None or score <= 1e-3:
            return self.backtrack()
        return [best]


class EpsilonStar(OnlineCoverage):
    """epsilon* (Song & Gupta, 2018): one potential per scale.

    A greedy potential field covers what is in front of it and then gets stuck,
    and the usual repair is to search the whole map for somewhere else to go --
    which is not a potential field any more. epsilon* keeps it local by keeping
    several maps: the grid, the grid in 2x2 blocks, in 4x4 blocks, and so on.

    While the finest level has an unfinished neighbour, follow it. When it does
    not, step up a level and ask the coarser map the same question, and again,
    until some scale has an answer. The robot then drops back to the fine
    level inside whichever block that was.

    This is not the same as "go to the nearest unfinished cell". A coarse block
    with forty cells left in it outranks a single stray cell that happens to be
    closer, so the robot leaves the crumbs and goes to the loaf.
    """

    name = "epsilon_star"

    def __init__(self, *, levels: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.levels = levels

    def choose(self, here: Cell) -> list[Cell] | None:
        for quarters in (0, -1, 1, 2):
            step = self._step(here, _turn(self.facing, quarters))
            if self.candidate(step):
                return [step]
        return self._ascend() or self.backtrack()

    def _ascend(self) -> list[Cell] | None:
        remaining = list(self._pending())
        if not remaining:
            return None
        for level in range(1, self.levels + 1):
            size = 1 << level
            blocks: dict[Cell, int] = {}
            for cell in remaining:
                key = (cell[0] // size, cell[1] // size)
                blocks[key] = blocks.get(key, 0) + 1
            mine = (self.here[0] // size, self.here[1] // size)
            others = {k: v for k, v in blocks.items() if k != mine}
            if not others:
                continue
            # Most work first, distance as the tie-break: the point of the
            # coarse level is to stop chasing single cells.
            for block in sorted(
                others, key=lambda k: (-others[k], abs(k[0] - mine[0]) + abs(k[1] - mine[1]))
            )[:6]:
                route = self._search(
                    lambda c, b=block, s=size: (c[0] // s, c[1] // s) == b and self.candidate(c)
                )
                if route:
                    self._backtracks += 1
                    return route
        return None

    def _pending(self):
        observed = np.argwhere(self.map.grid == MapCell.FREE)
        for row, col in observed:
            cell = (int(row), int(col))
            if self.candidate(cell):
                yield cell


class Predictive(OnlineCoverage):
    """PPCPP (Hassan & Liu, 2019): greedy on a reward, with a short lookahead.

    Everything else here optimises coverage and treats turning as a
    consequence. This one puts them in the same objective and weighs them
    against each other, one step at a time:

    * **coverage** -- one point for reaching a cell not yet done;
    * **smoothness** -- a penalty proportional to the turn required, because a
      turn costs time and slip that a metre of straight line does not;
    * **boundary** -- a bonus for cells against a wall, which sounds cosmetic
      and is not. Cells next to walls are the ones that get orphaned, and an
      orphaned cell costs a whole traverse to come back for.

    Greedy on that reward alone walks into dead ends, so each candidate is
    scored with a short discounted rollout of the same rule. Three steps is
    enough to see a wall coming and cheap enough to run every cell.
    """

    name = "ppcpp"

    def __init__(
        self,
        *,
        smoothness: float = 0.35,
        boundary: float = 0.25,
        horizon: int = 3,
        discount: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.smoothness = smoothness
        self.boundary = boundary
        self.horizon = horizon
        self.discount = discount

    def choose(self, here: Cell) -> list[Cell] | None:
        best: Cell | None = None
        score = -np.inf
        for direction in CCW:
            step = self._step(here, direction)
            if not self._open(step):
                continue
            value = self._reward(
                step, direction, self.facing, set()
            ) + self.discount * self._rollout(step, direction, {step}, self.horizon - 1)
            if value > score:
                score, best = value, step
        if best is None:
            return self.backtrack()
        return [best]

    def _open(self, cell: Cell) -> bool:
        """Somewhere the robot may step, covered or not.

        Unlike the others, PPCPP is allowed onto ground it has already done.
        It has to be: the coverage term is what distinguishes a new cell from
        an old one, and if every candidate were new by construction the term
        would be a constant and the reward would be smoothness and boundary
        alone.
        """
        return self.candidate(cell) or (cell in self.done and self.passable(cell))

    def _rollout(self, cell: Cell, facing: Cell, taken: set[Cell], left: int) -> float:
        if left <= 0:
            return 0.0
        best = 0.0
        for direction in CCW:
            step = self._step(cell, direction)
            if step in taken or not self._open(step):
                continue
            value = self._reward(step, direction, facing, taken) + self.discount * self._rollout(
                step, direction, taken | {step}, left - 1
            )
            best = max(best, value)
        return best

    def _reward(self, cell: Cell, direction: Cell, facing: Cell, taken: set[Cell]) -> float:
        fresh = 1.0 if cell not in self.done and cell not in taken else 0.0
        turn = abs(((CCW.index(direction) - CCW.index(facing) + 2) % 4) - 2) / 2.0
        return fresh - self.smoothness * turn + self.boundary * (self.walls_around(cell) / 8.0)


class Frontier(OnlineCoverage):
    """Nearest-frontier coverage (Yamauchi, 1997), as the control.

    Drive to the closest cell you have not done, every time. No sweep, no
    spiral, no field -- and it is here because it is the thing every other
    method has to beat. A surprising amount of published coverage work performs about
    this well, and reading any of these numbers without this row next to them
    would be reading them without a baseline.
    """

    name = "frontier"

    def choose(self, here: Cell) -> list[Cell] | None:
        for quarters in (0, -1, 1, 2):
            step = self._step(here, _turn(self.facing, quarters))
            if self.candidate(step):
                return [step]
        return self.nearest()


for _class in (
    SpiralSTC,
    FullSTC,
    BSA,
    BAStar,
    BrickAndMortar,
    NeuralField,
    EpsilonStar,
    Predictive,
    Frontier,
):

    def _factory(_class=_class, **kwargs):
        return _class(**kwargs)

    CONTROLLERS.register(_class.name)(_factory)
