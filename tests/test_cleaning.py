"""Cleaning interaction -- the coverage-is-not-cleanliness machinery."""

from __future__ import annotations

import pytest

from zimablue.dirt import DirtSpec, LayerSpec, make_dirt
from zimablue.physics.cleaning import apply_cleaning
from zimablue.pool import make_pool
from zimablue.rng import RngTree
from zimablue.robot import make_robot
from zimablue.simulation import Simulation


def _setup(dirt_type: str, grams: float = 40.0):
    pool = make_pool("rectangular")
    spec = DirtSpec(name="test", layers=(LayerSpec(dirt_type, grams_per_m2=grams),))
    state = spec.build(pool, RngTree(0).stream("dirt"))
    return pool, state, make_robot("tracked")


def _clean(pool, state, robot, *, brush_on, seconds=2.0, dt=0.02):
    load = 0.0
    for _ in range(int(seconds / dt)):
        outcome = apply_cleaning(
            pool,
            state,
            robot,
            x=5.0,
            y=2.5,
            heading=0.0,
            speed=0.0,
            brush_on=brush_on,
            pump_duty=1.0,
            filter_load=load,
            dt=dt,
            cell=0.10,
        )
        load = outcome.filter_load
    return load


def test_cleaning_removes_loose_dirt():
    pool, state, robot = _setup("sediment")
    before = state.field.total()
    _clean(pool, state, robot, brush_on=True)
    assert state.field.total() < before


def test_brush_advantage_rises_with_adhesion():
    """The central claim, stated as the relationship it actually is.

    The brush only matters for dirt that is *bonded* to the surface. So the
    ratio of (removed with brush) to (removed without) should be about 1 for
    loose sand and grow with adhesion, reaching several times for biofilm.

    Testing the ordering rather than one magic number keeps this honest: it
    asserts the physics the model claims, not a coefficient someone tuned.
    """
    advantage = {}
    for kind in ("sand", "sediment", "algae", "biofilm"):
        pool_a, state_a, robot = _setup(kind)
        pool_b, state_b, _ = _setup(kind)
        under = lambda s: float(s.field.layers[kind][20:30, 45:55].sum())  # noqa: E731, B023
        before = under(state_a)
        _clean(pool_a, state_a, robot, brush_on=True)
        _clean(pool_b, state_b, robot, brush_on=False)
        with_brush = before - under(state_a)
        without = before - under(state_b)
        advantage[kind] = with_brush / max(without, 1e-9)

    assert advantage["sand"] == pytest.approx(1.0, abs=0.15), (
        "the brush should make almost no difference to loose sand"
    )
    assert advantage["algae"] > 1.8, "adhered algae should need agitation"
    assert (
        advantage["biofilm"]
        > advantage["algae"]
        > advantage["sediment"]
        >= advantage["sand"] - 0.15
    )


def test_loose_dirt_barely_cares_about_the_brush():
    """The converse: sand is not bonded, so suction does most of the work."""
    pool_a, state_a, robot = _setup("sand")
    pool_b, state_b, _ = _setup("sand")
    under = lambda s: float(s.field.layers["sand"][20:30, 45:55].sum())  # noqa: E731

    before = under(state_a)
    _clean(pool_a, state_a, robot, brush_on=True)
    _clean(pool_b, state_b, robot, brush_on=False)
    assert (before - under(state_b)) > 0.5 * (before - under(state_a))


def test_biofilm_is_harder_than_algae():
    removed = {}
    for kind in ("algae", "biofilm"):
        pool, state, robot = _setup(kind)
        before = state.field.total()
        _clean(pool, state, robot, brush_on=True, seconds=1.0)
        removed[kind] = before - state.field.total()
    assert removed["biofilm"] < removed["algae"]


def test_fine_dirt_passes_through_a_coarse_filter():
    """Lifted is not the same as captured: fines below the mesh resettle."""
    pool, state, robot = _setup("sediment")
    outcome = apply_cleaning(
        pool,
        state,
        robot,
        x=5.0,
        y=2.5,
        heading=0.0,
        speed=0.1,
        brush_on=True,
        pump_duty=1.0,
        filter_load=0.0,
        dt=0.02,
        cell=0.10,
    )
    assert outcome.passed_through > 0
    assert outcome.captured < outcome.total_removed


def test_a_full_filter_stops_collecting():
    pool, state, robot = _setup("sand", grams=200.0)
    capacity = robot.cleaning.filter.capacity
    outcome = apply_cleaning(
        pool,
        state,
        robot,
        x=5.0,
        y=2.5,
        heading=0.0,
        speed=0.0,
        brush_on=True,
        pump_duty=1.0,
        filter_load=capacity,
        dt=0.02,
        cell=0.10,
    )
    assert outcome.captured == pytest.approx(0.0, abs=1e-9)


def test_pump_off_removes_nothing():
    pool, state, robot = _setup("sand")
    before = state.field.total()
    apply_cleaning(
        pool,
        state,
        robot,
        x=5.0,
        y=2.5,
        heading=0.0,
        speed=0.0,
        brush_on=False,
        pump_duty=0.0,
        filter_load=0.0,
        dt=0.02,
        cell=0.10,
    )
    assert state.field.total() == pytest.approx(before)


def test_oversized_debris_is_not_swallowed():
    pool = make_pool("rectangular")
    state = make_dirt("autumn").build(pool, RngTree(2).stream("dirt"))
    robot = make_robot("tracked")
    # Park one huge leaf right in front of the intake.
    state.debris.x[0], state.debris.y[0] = 5.3, 2.5
    state.debris.size[0] = robot.cleaning.pump.max_debris_size * 3
    outcome = apply_cleaning(
        pool,
        state,
        robot,
        x=5.0,
        y=2.5,
        heading=0.0,
        speed=0.1,
        brush_on=True,
        pump_duty=1.0,
        filter_load=0.0,
        dt=0.02,
        cell=0.10,
    )
    assert outcome.debris_blocked >= 1
    assert not state.debris.collected[0]


def test_debris_is_never_pushed_out_of_the_pool():
    pool = make_pool("rectangular")
    state = make_dirt("autumn").build(pool, RngTree(2).stream("dirt"))
    state.debris.x[0], state.debris.y[0] = 9.8, 2.5
    state.debris.size[0] = 0.5
    robot = make_robot("tracked")
    for _ in range(200):
        apply_cleaning(
            pool,
            state,
            robot,
            x=9.5,
            y=2.5,
            heading=0.0,
            speed=0.3,
            brush_on=True,
            pump_duty=1.0,
            filter_load=0.0,
            dt=0.02,
            cell=0.10,
        )
    assert bool(pool.contains(state.debris.x[0], state.debris.y[0]))


def test_dirt_mass_is_conserved_over_a_run():
    """Nothing vanishes: what leaves the pool is exactly what the filter holds."""
    sim = Simulation(pool="rectangular", dirt="light_sediment", seed=4, record=False)
    initial = sim.world.dirt.total_mass
    result = sim.run(seconds=90)
    remaining = result.world.dirt.total_mass
    assert remaining + result.state.dirt_collected == pytest.approx(initial, rel=1e-6)


def test_a_dirty_pool_gets_cleaner():
    sim = Simulation(pool="rectangular", dirt="light_sediment", seed=8, record=False)
    before = sim.world.dirt.total_mass
    result = sim.run(seconds=180)
    assert result.world.dirt.total_mass < before
    assert result.metrics.dirt_removed > 0
    assert 0.0 < result.metrics.dirt_removed_fraction <= 1.0
