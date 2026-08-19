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


# ======================================================================
# Water transport: where the dirt goes when nobody is cleaning it
# ======================================================================
def _field(rows=9, cols=41, grams=0.18):
    """A uniform field on a rectangular mask, for transport tests."""
    import numpy as np

    from zimablue.dirt.field import DirtField
    from zimablue.geometry import Grid

    grid = Grid.covering((0.0, 0.0, cols * 0.1, rows * 0.1), 0.1)
    mask = np.ones(grid.shape, dtype=bool)
    field = DirtField(grid, mask)
    field.add_layer("sediment", np.full(grid.shape, grams))
    return field


def _converging(field):
    """Flow that runs inward from both ends and meets in the middle."""
    import numpy as np

    cols = field.grid.shape[1]
    x = np.arange(cols)
    vx = np.tile(0.3 * np.tanh((cols / 2 - x) / 6.0), (field.grid.shape[0], 1))
    return vx, np.zeros_like(vx)


def test_drift_conserves_mass_exactly():
    """No renormalisation, so this has to hold by construction.

    The version this replaced lost mass at the mask and put it back by scaling
    the whole field, which is a multiplicative correction: the cell with the
    most dirt gets the most of it back.
    """
    import numpy as np

    field = _field()
    before = field.total()
    rng = np.random.default_rng(0)
    vx = rng.normal(0.0, 0.2, field.grid.shape)
    vy = rng.normal(0.0, 0.2, field.grid.shape)
    for _ in range(50):
        field.drift(vx, vy, 1.0)
    assert field.total() == pytest.approx(before, rel=1e-12)


def test_drift_does_not_wrap_around_the_pool():
    """``np.roll`` would teleport dirt out of one wall and in at the other."""
    import numpy as np

    field = _field(grams=0.0)
    layer = field.layers["sediment"]
    layer[:, 0] = 1.0
    vx = np.full(field.grid.shape, -0.3)
    field.drift(vx, np.zeros_like(vx), 1.0, spread=0.0)
    assert field.layers["sediment"][:, -1].sum() == 0.0, "dirt came in at the far wall"
    assert field.total() == pytest.approx(9.0, rel=1e-12), "and it was not lost either"


def test_the_upwind_gate_reads_the_cell_the_mass_leaves():
    """Where the flow changes sign, the destination's velocity is the wrong test.

    Two cells: the left one flows right, the right one flows left. The mass
    leaving the left cell must arrive in the right one. Gating on the
    destination -- which is what this used to do -- asks whether the *arrival*
    cell is flowing right, sees that it is not, and drops the mass.
    """
    import numpy as np

    field = _field(rows=1, cols=2, grams=0.0)
    field.layers["sediment"][0, 0] = 1.0
    vx = np.array([[0.3, -0.3]])
    field.drift(vx, np.zeros_like(vx), 1.0, spread=0.0)
    assert field.layers["sediment"][0, 1] > 0.0, "the mass never arrived"
    assert field.total() == pytest.approx(1.0, rel=1e-12)


def test_a_convergent_flow_makes_a_dirt_line_not_a_knife_edge():
    """The regression for the brown stripe down the middle of the kidney.

    Pure advection into a stagnation line concentrates without bound: four
    hundred steps put twenty times a cell's initial load onto one cell, and
    with a faster-suspending dirt type it reaches sixty. Diffusion is what
    turns that into the dirt line a real pool gets rather than a knife edge.
    """
    peaks = {}
    for spread in (0.0, 0.12):
        field = _field()
        vx, vy = _converging(field)
        start = field.layers["sediment"].max()
        for _ in range(400):
            field.drift(vx, vy, 1.0, spread=spread)
        peaks[spread] = field.layers["sediment"].max() / start
    assert peaks[0.0] > 15.0, "the test flow is not convergent enough to prove anything"
    assert peaks[0.12] < 0.6 * peaks[0.0], (
        f"diffusion barely helped: {peaks[0.12]:.0f}x against {peaks[0.0]:.0f}x without"
    )
    assert peaks[0.12] < 12.0


def test_diffusion_is_refused_where_it_would_be_unstable():
    import numpy as np

    field = _field()
    zero = np.zeros(field.grid.shape)
    with pytest.raises(ValueError, match="stability"):
        field.drift(zero, zero, 1.0, spread=0.3)


def test_every_pool_sweeps_its_dirt_towards_a_sink():
    """Where the flow piles dirt up says whether the plumbing makes sense.

    Left alone, dirt should end up on the drain or under the skimmer -- that
    is what they are for. Anywhere else is a dead spot, and a dead spot in
    open water is a plumbing mistake rather than a fact about pools.

    The kidney used to fail this: it had a return at each end aimed at the
    other, the jets met in the middle, and the pile landed 4 m from the drain
    in the middle of the floor. It looked exactly like coverage the robot had
    missed, which is how it was found.
    """
    import numpy as np

    from zimablue.dirt import DirtSpec, LayerSpec
    from zimablue.pool import POOL_PRESETS, make_pool
    from zimablue.rng import RngTree

    for name in POOL_PRESETS.names():
        pool = make_pool(name)
        sinks = [f.position for f in pool.features if type(f).__name__ in ("Drain", "Skimmer")]
        if not sinks:
            continue
        spec = DirtSpec(name="flat", layers=(LayerSpec("sediment", grams_per_m2=10.0),))
        field = spec.build(pool, RngTree(0).stream("dirt")).field
        field.layers["sediment"] = np.where(field.mask, 0.1, 0.0)
        vx, vy = pool.flow_grid(field.grid.cell)
        for _ in range(400):
            field.drift(vx, vy, 1.0)
        layer = field.layers["sediment"]
        row, col = np.unravel_index(int(layer.argmax()), layer.shape)
        px = field.grid.minx + (col + 0.5) * field.grid.cell
        py = field.grid.miny + (row + 0.5) * field.grid.cell
        gap = min(float(np.hypot(px - sx, py - sy)) for sx, sy in sinks)
        # Loose on purpose. A single return on an eleven-metre oval throws its
        # pile a metre past the skimmer and into the end wall, which is a sink
        # region rather than a dead spot. The failure this catches was four
        # metres out, in open water, in the middle of the floor.
        assert gap < 1.2, (
            f"{name} piles its dirt at ({px:.1f}, {py:.1f}), {gap:.1f} m from any sink"
        )


# ======================================================================
# Debris the intake cannot take
# ======================================================================
def test_the_ceiling_counts_exactly_the_debris_that_is_too_big():
    """Oversize items can never be collected, so they cap dirt removed.

    A run that reports 41% has done 44% of what this robot could possibly do,
    and the difference is not the controller's fault. Before this was
    reported, seven and a half percent of an autumn pool was permanently out
    of reach with nothing anywhere saying so.
    """
    import numpy as np

    result = Simulation(
        pool="kidney", robot="tracked", dirt="autumn", controller="baseline_coverage", seed=42
    ).run(minutes=2)
    metrics = result.metrics
    debris = result.world.dirt.debris
    limit = make_robot("tracked").cleaning.pump.max_debris_size

    # The skimmer may take an oversize floater; those stop being "out of
    # reach", so the ceiling counts oversize items still in the water.
    too_big = (np.asarray(debris.size) > limit) & ~np.asarray(debris.skimmed)
    assert metrics.debris_oversize == int(too_big.sum())
    assert metrics.uncollectable_dirt == pytest.approx(
        float(np.asarray(debris.mass)[too_big].sum())
    )
    assert 0.0 < metrics.dirt_ceiling < 1.0
    assert metrics.dirt_removed_fraction <= metrics.dirt_ceiling + 1e-9

    # And nothing under the limit was swallowed *by the robot* -- the skimmer
    # answers to a different size limit than the intake.
    swallowed = np.asarray(debris.collected) & ~np.asarray(debris.skimmed)
    assert not (swallowed & too_big).any(), "an oversize item was somehow swallowed"


def test_a_clean_pool_has_no_ceiling():
    result = Simulation(
        pool="rectangular", dirt="clean", controller="baseline_coverage", seed=1
    ).run(seconds=30)
    assert result.metrics.dirt_ceiling == 1.0
    assert result.metrics.debris_oversize == 0


def test_a_wider_intake_raises_the_ceiling_and_collects_more():
    """The trade the model can now express: intake against autonomy.

    Real cleaners split on exactly this -- most have a 3 to 4 inch mouth and
    are documented as prone to clogging on leaves and acorns, and the models
    sold for leafy gardens advertise a much wider one.
    """
    from dataclasses import replace

    from zimablue.robot import Cleaner

    narrow = make_robot("tracked")
    wide = Cleaner(
        chassis=narrow.chassis,
        locomotion=narrow.locomotion,
        power=narrow.power,
        cleaning=replace(narrow.cleaning, pump=replace(narrow.cleaning.pump, max_debris_size=0.2)),
        sensors=narrow.sensors,
    )

    runs = {}
    for name, robot in (("narrow", narrow), ("wide", wide)):
        runs[name] = (
            Simulation(
                pool="kidney", robot=robot, dirt="autumn", controller="baseline_coverage", seed=42
            )
            .run(minutes=8)
            .metrics
        )

    assert runs["wide"].dirt_ceiling > runs["narrow"].dirt_ceiling
    assert runs["wide"].debris_oversize < runs["narrow"].debris_oversize
    assert runs["wide"].debris_collected > runs["narrow"].debris_collected


def test_the_summary_says_what_it_could_not_reach():
    result = Simulation(
        pool="kidney", robot="tracked", dirt="autumn", controller="baseline_coverage", seed=42
    ).run(minutes=2)
    text = result.metrics.summary()
    assert "too big for the intake" in text
    assert "dirt ceiling" in text
