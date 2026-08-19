"""The dirt seeker: chases grams it can sense, not area it can count."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb
from zimablue.controllers.base import CONTROLLERS, ControlInput
from zimablue.planners import DirtSeeker
from zimablue.planners.compare import default_entries
from zimablue.sensors import Reading


def fake_input(t, turbidity, robot, dt=0.2):
    return ControlInput(
        time=t,
        dt=dt,
        readings={"turbidity": Reading(name="turbidity", time=t, values=np.array([turbidity]))},
        battery=1.0,
        filter_load=0.0,
        robot=robot,
    )


def test_it_is_registered_and_compared():
    assert "dirt_seeker" in CONTROLLERS
    assert "dirt_seeker" in default_entries()


def test_it_needs_no_truth():
    assert not getattr(DirtSeeker(), "needs_truth", False)


def test_a_spike_over_ambient_triggers_a_scrub():
    robot = zb.make_robot("tracked")
    seeker = DirtSeeker(trigger_ratio=1.7)
    seeker.reset(robot)
    t = 0.0
    for _ in range(50):  # learn the ambient level: a steady 8 g/m2
        seeker.step(fake_input(t, 8.0, robot))
        t += 0.2
    assert seeker.mode == "wander"
    seeker.step(fake_input(t, 30.0, robot))  # the pile
    assert seeker.mode == "scrub"
    assert seeker.hotspots, "the find should be remembered as well as acted on"


def test_a_steady_level_never_triggers_however_dirty():
    """Dirt everywhere is ambient, not a find: the trigger is relative."""
    robot = zb.make_robot("tracked")
    seeker = DirtSeeker()
    seeker.reset(robot)
    t = 0.0
    for _ in range(120):
        seeker.step(fake_input(t, 25.0, robot))
        t += 0.2
    assert seeker.mode == "wander"
    assert seeker._scrubs == 0


def test_it_out_collects_the_bounce_while_covering_less():
    """The thesis, run: same pool, same dirt, same seed, same duration.

    The seeker ends with more of the dirt and less of the floor -- the two
    metrics rank the two controllers in opposite orders, which is the point
    of the whole package.
    """
    seeker = zb.Simulation(pool="kidney", dirt="autumn", controller="dirt_seeker", seed=7).run(
        minutes=10
    )
    bounce = zb.Simulation(pool="kidney", dirt="autumn", controller="random_bounce", seed=7).run(
        minutes=10
    )

    assert seeker.metrics.dirt_removed_fraction > bounce.metrics.dirt_removed_fraction
    assert seeker.metrics.coverage < bounce.metrics.coverage


def test_its_telemetry_lands_in_the_recording():
    result = zb.Simulation(pool="rectangular", dirt="autumn", controller="dirt_seeker", seed=7).run(
        minutes=3
    )
    recording = result.require_recording()
    modes = np.asarray(recording.column("ctl.mode"), dtype=float)
    assert (modes == 2.0).any(), "it never once scrubbed an autumn pool"
    assert float(recording.column("ctl.scrubs")[-1]) > 0


def test_deterministic_across_runs():
    def run():
        return zb.Simulation(
            pool="rectangular", dirt="autumn", controller="dirt_seeker", seed=9
        ).run(minutes=2)

    first, second = run(), run()
    assert first.metrics.dirt_removed == pytest.approx(second.metrics.dirt_removed, abs=0)
    assert first.metrics.distance_traveled == pytest.approx(second.metrics.distance_traveled, abs=0)
