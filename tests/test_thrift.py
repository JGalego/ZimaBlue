"""Grams per watt-hour, and the filter that finally matters."""

from __future__ import annotations

import zimablue as zb
from zimablue.robot import Cleaner
from zimablue.robot.components import CleaningSystem, Filter
from zimablue.scenarios import Scenario


def test_grams_per_wh_is_reported_and_consistent():
    result = zb.Simulation(pool="rectangular", dirt="autumn", seed=5).run(minutes=2)
    m = result.metrics
    assert m.energy_consumed > 0
    assert m.grams_per_wh == m.dirt_collected / m.energy_consumed
    assert 0.0 <= m.filter_load_fraction <= 1.0
    assert m.filter_load_fraction > 0, "an autumn run puts something in the bag"


def tiny_bag_robot():
    base = zb.make_robot("tracked")
    cleaning = CleaningSystem(
        brush=base.cleaning.brush,
        pump=base.cleaning.pump,
        filter=Filter(capacity=3.0, mesh=base.cleaning.filter.mesh),
    )
    return Cleaner(
        name="tiny_bag",
        chassis=base.chassis,
        locomotion=base.locomotion,
        cleaning=cleaning,
        power=base.power,
        sensors=list(base.sensors.values()) if hasattr(base.sensors, "values") else None,
    )


def test_a_full_filter_can_end_the_run():
    sim = zb.Simulation(
        pool="rectangular",
        robot=tiny_bag_robot(),
        dirt="autumn",
        seed=5,
        stop_on_full_filter=True,
    )
    result = sim.run(minutes=20)
    assert result.metrics.termination == "filter_full"
    assert result.metrics.runtime < 20 * 60


def test_running_on_with_a_full_filter_stays_the_default():
    result = zb.Simulation(pool="rectangular", robot=tiny_bag_robot(), dirt="autumn", seed=5).run(
        minutes=5
    )
    assert result.metrics.termination != "filter_full"
    assert result.metrics.filter_load_fraction == 1.0


def test_the_scenario_key_round_trips():
    scenario = Scenario.from_dict({"name": "t", "termination": {"stop_on_full_filter": True}})
    assert scenario.stop_on_full_filter
    assert scenario.to_dict()["termination"]["stop_on_full_filter"] is True


def test_compare_scores_thrift():
    from zimablue.planners.compare import DIMENSIONS, evaluate

    assert any(d.key == "thrift" for d in DIMENSIONS)
    trial = evaluate("random_bounce", pool="rectangular", dirt="autumn", seed=3, minutes=1.0)
    assert trial.scores["thrift"] > 0
