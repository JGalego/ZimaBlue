"""Coverage algorithms that only make sense with more than one robot.

Partitioning (:mod:`zimablue.planners.partition`) is the half of multi-robot
coverage that reuses single-robot planners. This is the other half: methods
whose *decision rule* refers to the other robots.

======================  ==================================================
``mstc``                one spanning tree, cut into arcs by robot position
``mstc_backtracking``   the same, but a finished robot takes over a tail
``auction``             bid for the next cell, cheapest robot wins
``binn_swarm``          the neural field, with team-mates as inhibition
``smc_swarm``           one ergodic average, shared across the fleet
======================  ==================================================

There is a sixth that needs no code at all. Every online planner in
:mod:`zimablue.planners.online` becomes cooperative when a fleet hands it a
blackboard: it publishes what it has covered and skips what its team-mates
say they have done. ``Fleet(..., share=True)`` is that, it is the default, and
it is the baseline the five above have to beat.

What they can rely on
---------------------

The blackboard carries **estimates**, not truth. Two robots agree about which
cell is which only to the extent that their dead reckoning agrees, and it
degrades all run. So a method here is allowed to know roughly where its
team-mates are and roughly what they have done -- which is what a real fleet
has -- and any method that needed better than that would be describing a
different machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zimablue.controllers.base import CONTROLLERS
from zimablue.planners.base import CoveragePath, PathFollower, make_planner
from zimablue.planners.ergodic import SpectralCoverage
from zimablue.planners.online import NeuralField, OnlineCoverage, _turn

__all__ = [
    "AuctionFrontier",
    "MSTCFollower",
    "SwarmField",
    "SwarmSpectral",
    "mstc",
]


# ======================================================================
# Multi-robot spanning tree coverage
# ======================================================================
class MSTCFollower(PathFollower):
    """One robot's arc of a shared spanning-tree circuit (Hazon & Kaminka, 2005).

    Spiral-STC's path is a closed loop: circumnavigating a spanning tree
    returns you to where you started, having passed through every cell exactly
    once. MSTC's observation is that a closed loop cut into ``k`` arcs is ``k``
    tours, each complete on its own piece, with no partitioning of the *area*
    required at all -- the tree did it.

    That gives the property the paper is actually about, which is not speed but
    **robustness**: a robot that dies leaves a gap in one arc, and the
    neighbours can absorb it, because the arcs are contiguous stretches of one
    curve rather than separately planned routes through separate regions.

    ``backtracking`` turns on the second half of the paper. Without it a robot
    that finishes its arc stops, and the fleet's makespan is set by whoever
    drew the worst arc. With it, a finished robot takes the back half of the
    busiest team-mate's remaining stretch and announces it; the team-mate reads
    the announcement and stops early. Nothing here is negotiated -- the taker
    decides and the loser complies -- which is enough because the arcs are
    ordered and two robots cannot claim the same tail in the same tick.
    """

    def __init__(
        self,
        circuit: CoveragePath,
        begin: int,
        end: int,
        *,
        backtracking: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(_FixedRoute(_arc(circuit.waypoints, begin, end)), **kwargs)
        self.circuit = circuit
        self.begin = int(begin)
        self.end = int(end)
        self.backtracking = backtracking
        self.name = "mstc_backtracking" if backtracking else "mstc"
        self.handovers = 0
        self._claimed: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    def step(self, control_input: Any) -> Any:
        command = super().step(control_input)
        if self.blackboard is None or self.path is None:
            return command

        remaining = max(len(self.path) - self.target, 0)
        self.blackboard.publish(
            self.index,
            *self._pose,
            time=control_input.time,
            extras={
                "remaining": float(remaining),
                "begin": float(self.begin),
                "end": float(self.end),
                "position": float(self.begin + self.target),
            },
        )
        self._honour_claims()
        if self.backtracking and remaining <= 1:
            self._take_over()
        return command

    def _honour_claims(self) -> None:
        """Give up the tail of our arc if a team-mate has announced it."""
        claim = getattr(self.blackboard, "claims", {}).get(self.index)
        if claim is None or claim == self._claimed:
            return
        self._claimed = claim
        _, split = claim
        if split <= self.begin:
            return
        self.end = split
        self._rebuild()

    def _take_over(self) -> None:
        """Adopt the back half of whoever has the most left to do."""
        peers = self.blackboard.peers(self.index)
        busiest, most = None, 0.0
        for peer in peers:
            left = peer.extras.get("remaining", 0.0)
            if left > most:
                busiest, most = peer, left
        # Below a handful of cells the drive over costs more than the help.
        if busiest is None or most < 8:
            return
        here = int(busiest.extras.get("position", 0.0))
        finish = int(busiest.extras.get("end", 0.0))
        split = here + max(int((finish - here) // 2), 1)
        if split >= finish:
            return
        claims = getattr(self.blackboard, "claims", None)
        if claims is None:
            claims = self.blackboard.claims = {}
        claims[busiest.index] = (self.index, split)
        self.begin, self.end = split, finish
        self.handovers += 1
        self._rebuild()

    def _rebuild(self) -> None:
        self.planner = _FixedRoute(_arc(self.circuit.waypoints, self.begin, self.end))
        self.path = None
        self._planned_for = None
        self.target = 0
        self._last_target = -1

    def telemetry(self) -> dict[str, float]:
        base = super().telemetry()
        base["handovers"] = float(self.handovers)
        base["arc"] = float(max(self.end - self.begin, 0))
        return base


@dataclass
class _FixedRoute:
    """A planner that has already made up its mind."""

    waypoints: Any
    name: str = "arc"

    def plan(self, pool: Any, robot: Any) -> CoveragePath:
        return CoveragePath(waypoints=self.waypoints, planner=self.name)


def _arc(waypoints: Any, begin: int, end: int) -> Any:
    """The stretch of a closed circuit between two indices, wrapping."""
    count = len(waypoints)
    if count == 0:
        return waypoints
    begin %= count
    if end <= begin:
        end += count
    index = np.arange(begin, min(end, begin + count)) % count
    return waypoints[index]


def mstc(
    *,
    backtracking: bool = True,
    planner: Any = "spanning_tree",
    localisation: str = "odometry",
    **follower: Any,
):
    """Fleet factory: one spanning-tree circuit, cut into arcs.

        Fleet(pool="kidney", robots=3, controllers=mstc())

    The circuit comes from the single-robot ``spanning_tree`` planner, so the
    cells, the tree and the perimeter walk are the same code -- what MSTC adds
    is where to cut it.
    """

    def build(pool: Any, robots: list[Any], poses: list[tuple[float, float, float]]):
        circuit = make_planner(planner).plan(pool, robots[0])
        if len(circuit) < len(poses) * 2:
            raise ValueError(
                f"{planner} produced {len(circuit)} waypoints on {pool.name}, "
                f"which is too few to cut into {len(poses)} arcs"
            )
        way = circuit.waypoints
        entry = sorted(
            (int(np.argmin(np.hypot(*(way - np.array(pose[:2])).T))), index)
            for index, pose in enumerate(poses)
        )
        followers: list[Any] = [None] * len(poses)
        for slot, (start, robot_index) in enumerate(entry):
            finish = entry[(slot + 1) % len(entry)][0]
            if finish <= start:
                finish += len(way)
            followers[robot_index] = MSTCFollower(
                circuit,
                start,
                finish,
                backtracking=backtracking,
                localisation=localisation,
                **follower,
            )
        return followers

    build.name = "mstc_backtracking" if backtracking else "mstc"
    return build


# ======================================================================
# Market-based allocation
# ======================================================================
@CONTROLLERS.register("auction")
class AuctionFrontier(OnlineCoverage):
    """Bid for the next cell; the cheapest robot gets it (Zlot et al., 2002).

    Frontier coverage, with the conflict resolved by price instead of by luck.
    Each robot works out what the nearby unfinished cells would cost it -- in
    cells driven, through the free space it knows about -- and announces the
    one it wants along with its bid. A robot that sees a cheaper announcement
    for the cell it wanted moves on to its next choice.

    The auction is decentralised and single-round: nobody clears the market,
    every robot just declines to chase a target somebody else can reach more
    cheaply. That is the practical version of the idea and the one that
    survives a radio with range limits, which a centralised auctioneer does
    not.

    Ties break on index, which matters more than it sounds. Two robots
    equidistant from the same cell will otherwise both defer, both re-bid, and
    do it again next tick.
    """

    name = "auction"

    def __init__(self, *, shortlist: int = 6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shortlist = int(shortlist)
        self.outbid = 0

    def begin(self) -> None:
        self.bid: tuple[Any, float] | None = None

    def choose(self, here):
        for quarters in (0, -1, 1, 2):
            step = self._step(here, _turn(self.facing, quarters))
            if self.candidate(step) and not self._contested(step, 1.0):
                self._announce(step, 1.0)
                return [step]

        for route in self._shortlist():
            cost = float(len(route))
            if not self._contested(route[-1], cost):
                self._announce(route[-1], cost)
                return route
            self.outbid += 1
        # Everything nearby is somebody else's. Take the best of a bad lot
        # rather than stopping: a robot that refuses every contested cell in a
        # small pool refuses to move at all.
        fallback = self.nearest()
        if fallback:
            self._announce(fallback[-1], float(len(fallback)))
        return fallback

    def _shortlist(self):
        """A few of the cheapest reachable targets, cheapest first."""
        found: list[list] = []
        blocked: set = set()
        for _ in range(self.shortlist):
            route = self._search(lambda c: self.candidate(c) and c not in blocked)
            if not route:
                break
            found.append(route)
            blocked.add(route[-1])
        return found

    def _contested(self, cell, cost: float) -> bool:
        """Whether a team-mate has announced a cheaper bid for this cell."""
        if self.blackboard is None:
            return False
        for peer in self.blackboard.peers(self.index):
            want = peer.extras.get("bid_row"), peer.extras.get("bid_col")
            if want[0] is None or (int(want[0]), int(want[1])) != cell:
                continue
            theirs = peer.extras.get("bid_cost", np.inf)
            if theirs < cost or (theirs == cost and peer.index < self.index):
                return True
        return False

    def _announce(self, cell, cost: float) -> None:
        self.bid = (cell, cost)

    def _sync(self, ctl, pose) -> None:
        super()._sync(ctl, pose)
        if self.blackboard is None or self.bid is None:
            return
        cell, cost = self.bid
        post = self.blackboard.posts.get(self.index)
        if post is not None:
            post.extras.update(
                {"bid_row": float(cell[0]), "bid_col": float(cell[1]), "bid_cost": float(cost)}
            )

    def telemetry(self) -> dict[str, float]:
        base = super().telemetry()
        base["outbid"] = float(self.outbid)
        return base


# ======================================================================
# Field methods, extended to a fleet
# ======================================================================
@CONTROLLERS.register("binn_swarm")
class SwarmField(NeuralField):
    """BINN with the team-mates wired in as inhibition (Luo & Yang, 2008).

    The single-robot field already handles obstacles by driving their cells
    strongly negative. A team-mate is an obstacle that moves and that you would
    rather not follow, so it goes in the same way -- and because activity
    diffuses, the inhibition reaches further than the robot does. The fleet
    spreads out without anyone deciding that it should.

    This is the multi-robot version as published, and it is a five-line change
    to the single-robot one, which is the strongest argument for the field
    formulation there is.
    """

    name = "binn_swarm"

    def __init__(self, *, peer_weight: float = 0.6, peer_reach: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.peer_weight = float(peer_weight)
        self.peer_reach = int(peer_reach)

    def _input(self) -> np.ndarray:
        field = super()._input()
        if not self._peer_cells:
            return field
        reach = self.peer_reach
        for row, col in self._peer_cells:
            lo_r, hi_r = max(row - reach, 0), min(row + reach + 1, self.map.size)
            lo_c, hi_c = max(col - reach, 0), min(col + reach + 1, self.map.size)
            patch = field[lo_r:hi_r, lo_c:hi_c]
            # Only where there is no wall already: a wall is more inhibiting
            # than a colleague and should stay that way.
            patch[patch >= 0.0] = -self.peer_weight * self.drive
        return field


@CONTROLLERS.register("smc_swarm")
class SwarmSpectral(SpectralCoverage):
    """Ergodic coverage with one time-average for the whole fleet.

    Mathew and Mezic's metric is defined over a trajectory; for N agents it is
    defined over all N at once, and the control law each agent follows is the
    gradient of that same joint quantity. So the extension is not an
    approximation or a heuristic -- it is the original statement of the
    problem, and the single-robot case is the special one.

    In practice each robot publishes the running sum of its own basis
    coefficients and adds up what it hears. Two robots that have drifted apart
    contribute coefficients computed in slightly different frames, which blurs
    the joint spectrum rather than breaking it: an ergodic metric is about
    where time was spent, and being a few centimetres wrong about that is a
    small error in a smooth functional.
    """

    name = "smc_swarm"

    def _descent(self, pose) -> float:
        peers = self.blackboard.peers(self.index) if self.blackboard is not None else []
        shared = self._sum.copy()
        samples = max(len(self._trace), 1)
        for peer in peers:
            contribution = peer.extras.get("spectrum")
            weight = peer.extras.get("samples", 0.0)
            if contribution is None or not weight:
                continue
            if np.shape(contribution) != np.shape(shared):
                continue
            shared = shared + np.asarray(contribution)
            samples += int(weight)

        coefficients = shared / samples
        residual = self._weight * (coefficients - self._mu)
        self._phi = float((self._weight * (coefficients - self._mu) ** 2).sum())
        dx, dy = self._gradient(pose.x, pose.y)
        bx, by = float((residual * dx).sum()), float((residual * dy).sum())
        if abs(bx) < 1e-12 and abs(by) < 1e-12:
            return pose.heading
        return float(np.arctan2(-by, -bx))

    def _sync(self, ctl, pose) -> None:
        super()._sync(ctl, pose)
        if self.blackboard is None or self._domain is None:
            return
        post = self.blackboard.posts.get(self.index)
        if post is not None:
            # By reference, like the covered set: this models a robot
            # broadcasting its coefficients, and there are only sixty-four.
            post.extras["spectrum"] = self._sum
            post.extras["samples"] = float(len(self._trace))


def _fleet_ready() -> tuple[str, ...]:
    """Controllers whose decision rule refers to the other robots."""
    return ("auction", "binn_swarm", "smc_swarm")
