"""Shared fixtures.

Tests use short runs and coarse rasters deliberately: the suite has to stay
fast enough that people actually run it before pushing.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from zimablue.dirt import make_dirt
from zimablue.pool import make_pool
from zimablue.rng import RngTree
from zimablue.robot import make_robot
from zimablue.simulation import Simulation
from zimablue.world import World


@pytest.fixture
def pool():
    return make_pool("rectangular")


@pytest.fixture
def robot():
    return make_robot("tracked")


@pytest.fixture
def rng():
    return RngTree(1234)


@pytest.fixture
def world(pool, rng):
    return World.build(pool, make_dirt("light_sediment"), rng.stream("dirt"))


@pytest.fixture
def short_run():
    """A complete two-minute recorded run, reused by several test modules."""
    sim = Simulation(pool="rectangular", dirt="light_sediment", seed=7, scenario_name="test")
    return sim.run(seconds=120)
