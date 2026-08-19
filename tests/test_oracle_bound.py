"""The collectable bound: every run sits under it, whatever the planner."""

from __future__ import annotations

import numpy as np

import zimablue as zb
from zimablue.planners import collectable_bound, dirt_bound
from zimablue.planners.compare import DIMENSIONS, evaluate


def test_no_run_beats_the_bound():
    for controller in ("baseline_coverage", "random_bounce", "dirt_seeker"):
        result = zb.Simulation(
            pool="rectangular", dirt="autumn", controller=controller, seed=4
        ).run(minutes=5)
        recording = result.require_recording()
        bound = dirt_bound(
            recording,
            zb.make_pool("rectangular"),
            zb.make_robot("tracked"),
            seconds=result.metrics.runtime,
            cell=0.1,
        )
        assert result.metrics.dirt_removed <= bound + 1e-6, controller


def test_the_bound_grows_with_time_and_saturates():
    result = zb.Simulation(pool="rectangular", dirt="autumn", seed=4).run(minutes=1)
    recording = result.require_recording()
    pool, robot = zb.make_pool("rectangular"), zb.make_robot("tracked")
    short = dirt_bound(recording, pool, robot, seconds=30.0, cell=0.1)
    longer = dirt_bound(recording, pool, robot, seconds=300.0, cell=0.1)
    forever = dirt_bound(recording, pool, robot, seconds=1e7, cell=0.1)
    assert 0 < short < longer <= forever
    initial = recording.dirt_at(0.0).sum()
    assert forever <= initial + recording.debris_at(0.0)[:, 2].sum() + 1e-6


def test_the_bound_is_the_top_cells_not_the_mean():
    """A time-starved bound takes the heaviest cells, so a concentrated field
    bounds higher than a uniform one of the same mass."""
    navigable = np.ones((10, 10), dtype=bool)
    uniform = np.full((10, 10), 1.0)
    piled = np.zeros((10, 10))
    piled[0, :10] = 10.0
    kwargs = {"cell": 0.1, "speed": 0.4, "swath": 0.3, "seconds": 2.0}
    assert collectable_bound(piled, navigable, **kwargs) > collectable_bound(
        uniform, navigable, **kwargs
    )


def test_oversize_debris_is_not_held_against_a_planner():
    navigable = np.ones((4, 4), dtype=bool)
    grid = np.full((4, 4), 2.0)
    capped = collectable_bound(
        grid,
        navigable,
        cell=0.1,
        speed=1.0,
        swath=1.0,
        seconds=1e6,
        collectable_debris=0.0,
        collectable_total=20.0,
    )
    assert capped == 20.0


def test_compare_reports_the_possible_column():
    assert any(dim.key == "possible" for dim in DIMENSIONS)
    trial = evaluate("random_bounce", pool="rectangular", dirt="autumn", seed=3, minutes=1.0)
    value = trial.scores["possible"]
    assert 0.0 < value <= 1.0
