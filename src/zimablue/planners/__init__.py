"""Coverage path planning: the classical literature, implemented.

Coverage path planning asks how to move a robot so that a sensor or tool
passes over every point of a region.  It is the oldest question in cleaning
robotics and the field is large -- the most recent survey (Shen et al., 2026)
reviews 125 works and names over a hundred algorithms.

This module implements **the 2D branch of that literature**, and it implements
every distinct *mechanism* in it rather than every published variant. The
single-robot half is :mod:`~zimablue.planners.offline` and
:mod:`~zimablue.planners.online`; the multi-robot half is
:mod:`~zimablue.planners.partition` (divide the pool, then reuse a single-robot
planner in each share) and :mod:`~zimablue.planners.cooperative` (methods whose
decision rule refers to the other robots). What is deliberately absent, and
why:

* 3D and visual/inspection coverage -- the backend is planar and there is no
  camera in the loop;
* learning-based coverage -- that is :mod:`zimablue.rl`, which is the same idea
  with the policy learned rather than written.

Offline and online
------------------

The split matters because it decides what a method is allowed to know.

:mod:`~zimablue.planners.offline` gets the map and computes a whole route
before the robot moves. Each returns a
:class:`~zimablue.planners.base.CoveragePath`, and
:class:`~zimablue.planners.base.PathFollower` drives it::

    zb.Simulation(pool="kidney", controller=PathFollower("sweep_optimal"),
                  expose_truth=True).run(minutes=20)

:mod:`~zimablue.planners.online` gets sensor readings and decides as it goes.
Those are :class:`~zimablue.controllers.base.Controller` implementations
directly, and need nothing special::

    zb.Simulation(pool="kidney", controller="spiral_stc").run(minutes=20)

Reading the map is not the same as cheating. You could survey a pool once and
load the result, which is why an offline planner is a legitimate design and not
an oracle. But a plan still has to be *followed*, and ``PathFollower`` can do
that on the true pose or on dead reckoning. Comparing the two is the most
informative thing in this package: it separates "is this a good route" from
"can this robot drive it".
"""

from __future__ import annotations

from zimablue.planners.base import (
    PLANNERS,
    CoveragePath,
    CoveragePlanner,
    PathFollower,
    make_planner,
)
from zimablue.planners.cooperative import (
    AuctionFrontier,
    MSTCFollower,
    SwarmField,
    SwarmSpectral,
    mstc,
)
from zimablue.planners.ergodic import SpectralCoverage
from zimablue.planners.offline import (
    Boustrophedon,
    BoustrophedonCells,
    Contour,
    Morse,
    OptimalSweep,
    SpanningTree,
    Trapezoidal,
    Wavefront,
)
from zimablue.planners.online import (
    BSA,
    BAStar,
    BrickAndMortar,
    EpsilonStar,
    Frontier,
    FullSTC,
    NeuralField,
    OnlineCoverage,
    OnlineTuning,
    Predictive,
    SpiralSTC,
)
from zimablue.planners.partition import (
    PARTITIONS,
    Partition,
    Partitioner,
    Territory,
    make_partition,
    partitioned,
)

__all__ = [
    "BSA",
    "PARTITIONS",
    "PLANNERS",
    "AuctionFrontier",
    "BAStar",
    "Boustrophedon",
    "BoustrophedonCells",
    "BrickAndMortar",
    "Contour",
    "CoveragePath",
    "CoveragePlanner",
    "EpsilonStar",
    "Frontier",
    "FullSTC",
    "MSTCFollower",
    "Morse",
    "NeuralField",
    "OnlineCoverage",
    "OnlineTuning",
    "OptimalSweep",
    "Partition",
    "Partitioner",
    "PathFollower",
    "Predictive",
    "SpanningTree",
    "SpectralCoverage",
    "SpiralSTC",
    "SwarmField",
    "SwarmSpectral",
    "Territory",
    "Trapezoidal",
    "Wavefront",
    "make_partition",
    "make_planner",
    "mstc",
    "partitioned",
]
